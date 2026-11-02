"""Pictures and videos of a finished run (`tapeline viz <work>`), rendered on the CPU without OpenGL.

    tapeline viz runs/room                    # -> runs/room/viz/
    tapeline viz runs/room --out docs/assets --gif-seconds 12

Outputs (all in the world frame of the run: metres, z up, floor at z = 0):

* `triptych.mp4` - per selected frame: the input frame | the fused point cloud rendered from the
  refined pose of that frame | a top view (cloud slice below `slice_top`, camera trajectory, the
  current camera with its horizontal field of view);
* `triptych.gif` - the same, shorter and smaller (frames sampled evenly over the whole video);
* `hero.jpg` - the cloud seen obliquely from above, ceiling cut away, with camera frustums;
* `topview.png` - the cloud slice from above with the full trajectory;
* `visibility.png` - the visibility map of `wallfield.npz`: observed free space versus cells nobody
  looked at, with the cloud slice inside the map's height band (needs matplotlib: `pip install "tapeline[viz]"`).

Points are drawn by z-buffer splatting in numpy; videos are encoded with PyAV, GIFs with Pillow.
The renderer is a check of the poses as much as an illustration: if a pose is wrong, the middle
panel does not line up with the frame.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG = (24, 24, 28)
TRAJ = (150, 150, 160)
CAM = (255, 80, 60)


# ---------------------------------------------------------------------------------------------- scene
def load_run(work, max_points=2_000_000):
    """-> scene dict from a finished work directory (needs `sfm`, `depth`, `fuse` artefacts):
    xyz, rgb (cloud), R, t (world-to-camera, world frame of the cloud), K, wh (image size of K), names,
    images (frame directory), free (visibility map or None), cell (its cell size), band."""
    import pycolmap
    from .io import read_ply
    from .stages.world import build_world, poses_to_world

    work = Path(work)
    man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    st = man.get("stages", {})
    # the scale the cloud was fused with; the world frame is rebuilt exactly as in the fuse stage
    scale = float(st.get("fuse", {}).get("scale") or st.get("scale", {}).get("scale") or 1.0)
    cfg = man.get("config", {})
    z = np.load(work / "depth.npz")
    names = [str(s) for s in z["names"]]
    rec = pycolmap.Reconstruction(str(work / "sfm" / "model"))
    X = np.array([p.xyz for p in rec.points3D.values()])
    w = build_world(rec, X, scale)
    R, t = poses_to_world(w, np.asarray(z["poses"], np.float64), scale)
    xyz, rgb = read_ply(work / "cloud.ply")
    xyz, rgb = _subsample(xyz, rgb, max_points)
    free, cell, band = None, 2 * float(cfg.get("voxel", 0.02)), tuple(cfg.get("free_band", (0.5, 2.0)))
    if (work / "wallfield.npz").exists():
        free = np.load(work / "wallfield.npz")["free"].astype(np.float32)
    order = np.argsort(names, kind="stable")
    return dict(xyz=xyz.astype(np.float32), rgb=rgb, R=R[order], t=t[order], K=np.asarray(z["K"], np.float64),
                wh=tuple(int(v) for v in z["img_wh"]), names=[names[i] for i in order],
                images=work / "scan" / "images", free=free, cell=cell, band=band, voxel=cell / 2)


def _subsample(xyz, rgb, n, seed=0):
    if len(xyz) <= n:
        return xyz, rgb
    i = np.sort(np.random.default_rng(seed).choice(len(xyz), n, replace=False))
    return xyz[i], rgb[i]


def centres(R, t):
    return -np.einsum("nji,nj->ni", R, t)


# ------------------------------------------------------------------------------------------- renderer
def splat(xyz, rgb, R, t, K, size, point=2, bg=BG, near=0.05, size_m=None, max_point=6):
    """Perspective z-buffer render of coloured points. R, t world-to-camera (OpenCV axes), K for
    `size` = (w, h). Each point covers a square of `point` x `point` pixels or, with `size_m`, the
    footprint of a `size_m`-metre square at its depth (at least `point`, at most `max_point` pixels),
    so that near surfaces stay closed. The nearest point wins.
    -> (uint8 image h x w x 3, coverage = fraction of pixels hit)."""
    w, h = size
    pc = np.asarray(xyz, np.float32) @ np.asarray(R, np.float32).T + np.asarray(t, np.float32)
    zc = pc[:, 2]
    m = zc > near
    pc, zc, col = pc[m], zc[m], np.asarray(rgb)[m]
    u = np.floor(K[0, 0] * pc[:, 0] / zc + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * pc[:, 1] / zc + K[1, 2]).astype(np.int64)
    if size_m:
        ps = np.clip(np.ceil(K[0, 0] * size_m / zc), point, max(point, max_point)).astype(np.int64)
    else:
        ps = np.full(len(zc), point, np.int64)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    us, vs, zs, cs = [], [], [], []
    for du in range(int(ps.max(initial=point))):
        for dv in range(int(ps.max(initial=point))):
            uu, vv = u - ps // 2 + du, v - ps // 2 + dv
            k = (du < ps) & (dv < ps) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            us.append(uu[k]); vs.append(vv[k]); zs.append(zc[k]); cs.append(col[k])
    if not sum(len(x) for x in us):
        return img, 0.0
    idx = np.concatenate(vs) * w + np.concatenate(us)
    zz, cc = np.concatenate(zs), np.concatenate(cs)
    o = np.lexsort((zz, idx))                          # by pixel, then by depth: the first per pixel is nearest
    idx, cc = idx[o], cc[o]
    first = np.r_[True, idx[1:] != idx[:-1]]
    img.reshape(-1, 3)[idx[first]] = cc[first]
    return img, float(first.sum()) / (w * h)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """-> R, t (world-to-camera, OpenCV axes: x right, y down, z forward) of a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(a, np.float64) for a in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ eye


def project(P, R, t, K):
    pc = np.asarray(P, np.float64) @ R.T + t
    zc = np.maximum(pc[:, 2], 1e-6)
    return np.stack([K[0, 0] * pc[:, 0] / zc + K[0, 2], K[1, 1] * pc[:, 1] / zc + K[1, 2]], 1), pc[:, 2]


def ramp(s):
    """Blue -> green -> yellow colour ramp for s in [0, 1] (trajectory time, map values)."""
    s = np.clip(np.asarray(s, np.float64), 0, 1)[..., None]
    a = np.array([68, 1, 84.0]); b = np.array([33, 145, 140.0]); c = np.array([253, 231, 37.0])
    return np.where(s < 0.5, a + (b - a) * s * 2, b + (c - b) * (s - 0.5) * 2).astype(np.uint8)


# ------------------------------------------------------------------------------------------- top view
class TopView:
    """Orthographic view from above: fixed extent from the cloud and the cameras, metres -> pixels."""

    def __init__(self, scene, size, slice_top=1.9, margin=0.3, point=2):
        xyz, C = scene["xyz"], centres(scene["R"], scene["t"])
        lo = np.minimum(np.percentile(xyz[:, :2], 0.5, axis=0), C[:, :2].min(0)) - margin
        hi = np.maximum(np.percentile(xyz[:, :2], 99.5, axis=0), C[:, :2].max(0)) + margin
        self.w, self.h = size
        self.ppm = min(self.w / (hi[0] - lo[0]), self.h / (hi[1] - lo[1]))
        self.c = (lo + hi) / 2
        keep = xyz[:, 2] < slice_top                    # cut the ceiling away, keep floor, walls and furniture
        P, col = xyz[keep], scene["rgb"][keep]
        u, v = self.px(P[:, :2]).T.astype(np.int64)
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = BG
        k = (u >= 0) & (u < self.w - point + 1) & (v >= 0) & (v < self.h - point + 1)
        o = np.argsort(P[k, 2], kind="stable")          # the highest point of each pixel is drawn last
        for du in range(point):
            for dv in range(point):
                img[v[k][o] + dv, u[k][o] + du] = col[k][o]
        self.base = img

    def px(self, xy):
        """world XY (metres) -> pixel (x right = +X, up = +Y)."""
        xy = np.asarray(xy, np.float64)
        return np.stack([(xy[..., 0] - self.c[0]) * self.ppm + self.w / 2,
                         self.h / 2 - (xy[..., 1] - self.c[1]) * self.ppm], -1)

    def draw(self, scene, upto=None, cone=1.0, dim=0.75):
        """Top view with the trajectory up to frame `upto` (None: the whole trajectory) and the camera
        `upto` with its horizontal field of view drawn as a cone of length `cone` metres."""
        R, t, K = scene["R"], scene["t"], scene["K"]
        C = centres(R, t)
        im = Image.fromarray((self.base.astype(np.float32) * dim + np.array(BG) * (1 - dim)).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        lw = max(1, round(self.w / 300))
        q = [tuple(p) for p in self.px(C[:, :2])]
        if len(q) > 1:
            d.line(q, fill=TRAJ + (140,), width=lw)
            n = len(q) if upto is None else upto + 1
            col = ramp(np.linspace(0, 1, len(q)))
            for i in range(1, n):
                d.line([q[i - 1], q[i]], fill=tuple(int(c) for c in col[i]) + (255,), width=lw + 1)
        if upto is not None:
            i = upto
            wimg, himg = scene["wh"]
            rays = np.linalg.inv(K) @ np.array([[0, himg / 2, 1.0], [wimg, himg / 2, 1.0]]).T
            dirs = (R[i].T @ rays).T[:, :2]
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
            tri = self.px(np.stack([C[i, :2], C[i, :2] + cone * dirs[0], C[i, :2] + cone * dirs[1]]))
            d.polygon([tuple(p) for p in tri], fill=CAM + (90,), outline=CAM + (255,))
            r = lw * 2 + 1
            x, y = q[i]
            d.ellipse([x - r, y - r, x + r, y + r], fill=CAM + (255,))
        return im


# ---------------------------------------------------------------------------------------------- views
def _frame(scene, i, size):
    p = Path(scene["images"]) / scene["names"][i]
    if p.exists():
        return Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
    return Image.new("RGB", size, BG)


def _scaled_K(scene, size):
    K = scene["K"].copy()
    sx, sy = size[0] / scene["wh"][0], size[1] / scene["wh"][1]
    K[0] *= sx
    K[1] *= sy
    return K


def triptych_frame(scene, i, height, top, labels=True):
    """One triptych frame (PIL image): frame i | cloud rendered from pose i | top view."""
    w = int(round(height * scene["wh"][0] / scene["wh"][1] / 2)) * 2
    size = (w, height)
    frame = _frame(scene, i, size)
    pt = max(1, round(height / 360))
    ren, _ = splat(scene["xyz"], scene["rgb"], scene["R"][i], scene["t"][i], _scaled_K(scene, size), size, point=pt,
                   size_m=1.5 * scene.get("voxel", 0.02))
    tv = top.draw(scene, upto=i)
    im = Image.new("RGB", (2 * w + top.w, height), BG)
    im.paste(frame, (0, 0)); im.paste(Image.fromarray(ren), (w, 0)); im.paste(tv, (2 * w, 0))
    if labels:
        d = ImageDraw.Draw(im)
        for x, s in ((0, "input frame"), (w, "point cloud, this camera"), (2 * w, "top view")):
            d.text((x + 6, 4), s, fill=(255, 255, 255))
    return im


def write_mp4(frames, path, fps):
    """Frames (iterable of PIL images of one size) -> H.264 MP4 (PyAV); MPEG-4 if libx264 is missing."""
    import av
    out = av.open(str(path), "w")
    st = None
    try:
        for im in frames:
            if st is None:
                try:
                    st = out.add_stream("libx264", rate=fps, options={"crf": "23", "preset": "medium"})
                except Exception:
                    st = out.add_stream("mpeg4", rate=fps)
                st.width, st.height, st.pix_fmt = im.width // 2 * 2, im.height // 2 * 2, "yuv420p"
            vf = av.VideoFrame.from_image(im.crop((0, 0, st.width, st.height)))
            for pkt in st.encode(vf):
                out.mux(pkt)
        if st is not None:
            for pkt in st.encode():
                out.mux(pkt)
    finally:
        out.close()
    return Path(path)


def write_gif(frames, path, fps, colors=64):
    """Frames -> looping GIF with one palette for all frames (taken from 8 frames spread over the clip),
    no dithering: several times smaller than per-frame palettes on rendered point clouds."""
    frames = list(frames)
    k = min(8, len(frames))
    strip = Image.new("RGB", (frames[0].width, frames[0].height * k))
    for j in range(k):
        strip.paste(frames[j * len(frames) // k], (0, j * frames[0].height))
    pal = strip.quantize(colors=colors, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
    return Path(path)


def hero(scene, size=(1600, 900), slice_top=1.9, elev=50.0, azim=None, fov=45.0, every=None, ss=2, fill=0.9):
    """Oblique view of the cloud from above with the ceiling cut at `slice_top` and camera frustums.
    `azim` (degrees, None: the long horizontal axis of the cloud runs across the image, seen 30 degrees off
    its normal); the distance is chosen so that the cloud fills `fill` of the image. Rendered at `ss` x the
    size and downsampled."""
    W, H = size[0] * ss, size[1] * ss
    keep = scene["xyz"][:, 2] < slice_top
    P, col = scene["xyz"][keep], scene["rgb"][keep]
    lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    target = (lo + hi) / 2
    radius = 0.5 * np.linalg.norm(hi - lo)
    f = 0.5 * H / np.tan(np.radians(fov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    if azim is None:
        xy = P[:, :2] - P[:, :2].mean(0)
        e1 = np.linalg.eigh(xy.T @ xy)[1][:, -1]
        azim = np.degrees(np.arctan2(e1[1], e1[0])) - 90.0 + 30.0
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    S = P[np.random.default_rng(0).choice(len(P), min(len(P), 20000), replace=False)]
    dist = radius / np.tan(np.radians(fov) / 2)
    for _ in range(6):        # fit the projected cloud (0.5-99.5 percentiles) into the image, centred by the principal point
        R, t = look_at(target + dist * view, target)
        K[0, 2], K[1, 2] = W / 2, H / 2
        uv, _ = project(S, R, t, K)
        ulo, uhi = np.percentile(uv, [0.5, 99.5], axis=0)
        dist *= max((uhi[0] - ulo[0]) / (fill * W), (uhi[1] - ulo[1]) / (fill * H))
    R, t = look_at(target + dist * view, target)
    uv, _ = project(S, R, t, K)
    ulo, uhi = np.percentile(uv, [0.5, 99.5], axis=0)
    K[0, 2], K[1, 2] = W - (ulo[0] + uhi[0]) / 2, H - (ulo[1] + uhi[1]) / 2
    img, _ = splat(P, col, R, t, K, (W, H), point=ss, size_m=1.5 * scene.get("voxel", 0.02), max_point=4 * ss)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    C = centres(scene["R"], scene["t"])
    n = len(C)
    every = every or max(1, n // 40)
    wimg, himg = scene["wh"]
    Kinv = np.linalg.inv(scene["K"])
    corners = Kinv @ np.array([[0, 0, 1.0], [wimg, 0, 1], [wimg, himg, 1], [0, himg, 1]]).T
    depth = 0.04 * radius + 0.1
    cols = ramp(np.linspace(0, 1, n))
    q, _ = project(C, R, t, K)
    d.line([tuple(p) for p in q], fill=TRAJ + (200,), width=ss * 2)
    for i in range(0, n, every):
        pts = C[i] + depth * (scene["R"][i].T @ corners).T
        uv, _ = project(np.vstack([C[i][None], pts]), R, t, K)
        c = tuple(int(x) for x in cols[i]) + (255,)
        a0 = tuple(uv[0])
        for j in range(4):
            d.line([a0, tuple(uv[1 + j])], fill=c, width=ss)
            d.line([tuple(uv[1 + j]), tuple(uv[1 + (j + 1) % 4])], fill=c, width=ss)
    return im.resize(size, Image.LANCZOS)


def visibility(scene, path, slice_band=None, traj=True, dpi=110):
    """Visibility map of wallfield.npz -> PNG: observed free space (colour = mean number of observed free
    voxels per column of the cell), cells nobody looked at (dark), the cloud inside the map's height band
    (white) and the camera trajectory. Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as e:
        raise SystemExit('the visibility map needs matplotlib: pip install "tapeline[viz]"') from e
    F = scene["free"]
    if F is None or not len(F):
        raise ValueError("no visibility map in the run (wallfield.npz missing or empty)")
    cell = scene["cell"]
    band = slice_band or scene["band"]
    xyz = scene["xyz"]
    occ = xyz[(xyz[:, 2] >= band[0]) & (xyz[:, 2] <= band[1])]
    C = centres(scene["R"], scene["t"])
    allxy = np.vstack([F[:, :2], occ[:, :2], C[:, :2]])
    # the grid is aligned to the cell centres of the map (rounding, not flooring: the centres are float16)
    lo = F[:, :2].min(0) - cell * (np.ceil((F[:, :2].min(0) - allxy.min(0)) / cell) + 2.5)
    hi = allxy.max(0) + 2 * cell
    nx, ny = (np.ceil((hi - lo) / cell).astype(int) + 1)
    grid = np.full((ny, nx), np.nan)
    ix = np.clip(np.round((F[:, 0] - lo[0]) / cell - 0.5).astype(int), 0, nx - 1)
    iy = np.clip(np.round((F[:, 1] - lo[1]) / cell - 0.5).astype(int), 0, ny - 1)
    grid[iy, ix] = F[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx + 0.6), dpi=dpi)
    ax.set_facecolor("#202024")
    ext = (lo[0], lo[0] + nx * cell, lo[1], lo[1] + ny * cell)
    vmax = float(np.nanpercentile(grid, 95)) if np.isfinite(grid).any() else 1.0
    cmap = ListedColormap(ramp(np.linspace(0, 1, 256)) / 255.0)
    m = ax.imshow(np.ma.masked_invalid(grid), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6),
                  interpolation="nearest")
    if len(occ):
        o = occ[np.random.default_rng(0).choice(len(occ), min(len(occ), 300_000), replace=False)]
        ax.scatter(o[:, 0], o[:, 1], s=0.05, c="white", alpha=0.35, linewidths=0, rasterized=True)
    if traj:
        ax.plot(C[:, 0], C[:, 1], color="#ff503c", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m")
    ax.set_title("visibility map: observed free space (colour) vs. unobserved (dark); cloud in "
                 f"{band[0]:g}-{band[1]:g} m (white)", fontsize=9)
    cb = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("observed free voxels per column", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


# ------------------------------------------------------------------------------------------------ all
def render(scene, out, fps=10, height=360, gif_fps=8, gif_seconds=12, gif_width=800, hero_size=(1600, 900),
           what=("triptych", "gif", "hero", "topview", "visibility"), log=print):
    """Render the selected outputs of a scene into directory `out`; -> {name: path}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(scene["names"])
    res = {}
    if "triptych" in what:
        top = TopView(scene, (height, height))
        res["triptych"] = write_mp4((triptych_frame(scene, i, height, top) for i in range(n)), out / "triptych.mp4", fps)
        log(f"triptych.mp4: {n} frames at {fps} fps")
    if "gif" in what:
        k = max(1, min(n, int(gif_fps * gif_seconds)))
        idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        # panel height so that frame | render | square top view fit into gif_width
        ar = scene["wh"][0] / scene["wh"][1]
        gh = int(gif_width / (2 * ar + 1)) // 2 * 2
        top = TopView(scene, (gh, gh))
        res["gif"] = write_gif([triptych_frame(scene, i, gh, top, labels=False) for i in idx], out / "triptych.gif",
                               gif_fps)
        log(f"triptych.gif: {len(idx)} frames at {gif_fps} fps, panel height {gh}")
    if "hero" in what:
        hero(scene, hero_size).save(out / "hero.jpg", quality=88)
        res["hero"] = out / "hero.jpg"
    if "topview" in what:
        ext = np.ptp(scene["xyz"][:, :2], axis=0) + 1e-6   # canvas follows the aspect of the scene, long side 1000 px
        size = (1000, max(200, int(1000 * ext[1] / ext[0]))) if ext[0] >= ext[1] else (max(200, int(1000 * ext[0] / ext[1])), 1000)
        TopView(scene, size, point=2).draw(scene, upto=None, dim=1.0).save(out / "topview.png", optimize=True)
        res["topview"] = out / "topview.png"
    if "visibility" in what and scene.get("free") is not None and len(scene["free"]):
        res["visibility"] = visibility(scene, out / "visibility.png")
    return {k: str(v) for k, v in res.items()}
