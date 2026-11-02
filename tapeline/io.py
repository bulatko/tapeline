"""Artifact reading/writing: PLY, poses from external formats, COLMAP model."""
import json
from pathlib import Path
import numpy as np

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_ply(path, xyz, rgb=None):
    xyz = np.ascontiguousarray(xyz, np.float32)
    rgb = np.zeros((len(xyz), 3), np.uint8) if rgb is None else np.asarray(rgb, np.uint8)
    arr = np.empty(len(xyz), PLY_DTYPE)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n" % len(arr)).encode())
        f.write(arr.tobytes())
    return Path(path)


def read_ply(path):
    d = np.fromfile(path, dtype=np.uint8)
    head = d[:512].tobytes().split(b"end_header\n")[0] + b"end_header\n"
    b = d[len(head):].view(PLY_DTYPE)
    return (np.stack([b["x"], b["y"], b["z"]], 1).astype(np.float64),
            np.stack([b["red"], b["green"], b["blue"]], 1))


def write_obj_mesh(path, verts, faces, rgb=None):
    """Mesh from marching cubes; color is per-vertex (the obj format allows it via an extension)."""
    with open(path, "w") as f:
        for i, v in enumerate(verts):
            if rgb is None:
                f.write("v %.4f %.4f %.4f\n" % tuple(v))
            else:
                c = rgb[i] / 255.0
                f.write("v %.4f %.4f %.4f %.3f %.3f %.3f\n" % (*v, *c))
        for t in faces + 1:
            f.write("f %d %d %d\n" % tuple(t))
    return Path(path)


# ---------------------------------------------------------------- external poses
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _to_w2c_cv(M, convention, axes, order):
    """Any external pose -> w2c in the CV convention (same as COLMAP)."""
    M = np.asarray(M, np.float64)
    if M.size == 16:
        M = M.reshape(4, 4, order=order)
    if axes == "gl":                 # ARKit/OpenGL: Y up, Z backward
        M = M @ ARKIT_TO_CV if convention == "c2w" else ARKIT_TO_CV @ M
    if convention == "c2w":
        M = np.linalg.inv(M)
    return M


def read_poses(path, fmt="auto", convention="w2c", axes="cv", order="C"):
    """-> dict(names, extrinsics (N,4,4) w2c CV, intrinsics (N,3,3) | None).

    The convention is given explicitly and never guessed: swapped axes produce plausible-looking
    garbage that only shows up in the dense cloud.
    """
    p = Path(path)
    if fmt == "auto":
        fmt = ("colmap" if p.is_dir() else
               "npz" if p.suffix == ".npz" else
               "jsonl" if p.suffix in (".jsonl", ".json") else "colmap")

    if fmt == "npz":
        z = np.load(p, allow_pickle=True)
        E = np.asarray(z["extrinsics"], np.float64)
        names = [str(s) for s in z["image_names"]]
        K = np.asarray(z["intrinsics"], np.float64) if "intrinsics" in z else None
        E = np.stack([_to_w2c_cv(m, convention, axes, order) for m in E])
        return dict(names=names, extrinsics=E, intrinsics=K)

    if fmt == "jsonl":
        names, E, K = [], [], []
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            names.append(r.get("name") or r.get("file") or r["frame"])
            E.append(_to_w2c_cv(r.get("transform") or r["pose"], convention, axes, order))
            if "intrinsics" in r:
                K.append(np.asarray(r["intrinsics"], np.float64).reshape(3, 3, order=order))
        return dict(names=names, extrinsics=np.stack(E),
                    intrinsics=np.stack(K) if K else None)

    import pycolmap
    rec = pycolmap.Reconstruction(str(p))
    names, E, K = [], [], []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        pose = im.cam_from_world()
        M = np.eye(4)
        M[:3, :3] = pose.rotation.matrix()
        M[:3, 3] = pose.translation
        names.append(im.name)
        E.append(_to_w2c_cv(M, convention, axes, order))
        cam = rec.cameras[im.camera_id]
        f = cam.params
        K.append(np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1]])
                 if cam.model.name == "OPENCV" else
                 np.array([[f[0], 0, f[1]], [0, f[0], f[2]], [0, 0, 1]]))
    return dict(names=names, extrinsics=np.stack(E), intrinsics=np.stack(K))
