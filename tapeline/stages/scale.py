"""Metric scale of the scale-free reconstruction.

Estimate: DA3 in its own scale (align_to_input_ext_scale=False) vs. our depths at the
pixels of triangulated points. Plus an independent plausibility check from the room
geometry - ceiling height and camera height. A mismatch is not silently fixed but reported:
the scale stays a separate factor, and if it is wrong everything is recomputed with that one factor.
"""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pycolmap

from .world import build_world, camera_centres


# DA3's metric depth (NestedDepthAnything3._apply_metric_scaling) is multiplied by ((fx+fy)/2)/300 of the focal length
# PREDICTED by the network (cam_dec), not of our K; the api then replaces pred.intrinsics with our K, so f_pred is not
# visible from outside. Any error of the predicted focal length goes into the scale by the same factor.
# Correction: s = s_raw · median(f_known/f_pred) over views (equivalent to running DA3 with f_known).
# Disable: env TAPELINE_SCALE_FOCAL_FIX=0 (or cfg.scale_focal_fix = False).
@contextmanager
def _focal_probe():
    """During inference, intercepts f_pred (what goes into apply_metric_scaling) and f_known (our K in process_res
    pixels, the model input). The originals are restored in finally - including on exception. Does not change the
    computation. -> rec: dict(f_pred, f_known, error)."""
    rec = {"f_pred": [], "f_known": [], "error": None}
    try:
        import depth_anything_3.api as api
        import depth_anything_3.model.da3 as da3m
        orig_ams = da3m.apply_metric_scaling
        orig_fwd = api.DepthAnything3._run_model_forward
    except Exception as e:                                   # name/module changed - no correction will be applied
        rec["error"] = f"hook not installed: {type(e).__name__}: {e}"
        yield rec
        return
    stash = {}

    def run_fwd(self, imgs, ex_t, in_t, *a, **k):
        stash["in_t"] = in_t
        return orig_fwd(self, imgs, ex_t, in_t, *a, **k)

    def ams(depth, intrinsics, *a, **k):
        try:
            fp = ((intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2).detach().float().cpu().numpy().ravel()
            kin = stash.get("in_t")
            fk = ((kin[:, :, 0, 0] + kin[:, :, 1, 1]) / 2).detach().float().cpu().numpy().ravel()
            rec["f_pred"].extend(fp.tolist()); rec["f_known"].extend(fk.tolist())
        except Exception as e:                               # recording must not break inference
            rec["error"] = f"recording f: {type(e).__name__}: {e}"
        return orig_ams(depth, intrinsics, *a, **k)

    da3m.apply_metric_scaling = ams
    api.DepthAnything3._run_model_forward = run_fwd
    try:
        yield rec
    finally:
        da3m.apply_metric_scaling = orig_ams
        api.DepthAnything3._run_model_forward = orig_fwd


def _focal_fix_enabled(cfg):
    env = os.environ.get("TAPELINE_SCALE_FOCAL_FIX")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "no", "")
    return bool(getattr(cfg, "scale_focal_fix", True))


def _observations(rec, im):
    pose = im.cam_from_world()
    R, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
    uv, z = [], []
    for p2 in im.points2D:
        if p2.has_point3D():
            Xc = R @ rec.points3D[p2.point3D_id].xyz + t
            if Xc[2] > 1e-6:
                uv.append(p2.xy); z.append(Xc[2])
    return np.asarray(uv), np.asarray(z)


def run(cfg, n_take=24):
    p = cfg.paths()
    rec = pycolmap.Reconstruction(str(p["sfm"]))
    X = np.array([q.xyz for q in rec.points3D.values()])

    if cfg.scale is not None:
        s = float(cfg.scale)
        src = "config"
    elif cfg.poses_metric:
        s, src = 1.0, "metric_poses"
    else:
        cam = list(rec.cameras.values())[0]
        fx, fy, cx, cy = cam.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        imgs = sorted(rec.images.values(), key=lambda i: i.name)
        take = imgs[::max(1, len(imgs) // n_take)][:n_take]
        paths, exts, obs = [], [], []
        for im in take:
            pose = im.cam_from_world()
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = pose.rotation.matrix(); E[:3, 3] = pose.translation
            paths.append(str(Path(p["frames_sel"]) / im.name)); exts.append(E)
            obs.append(_observations(rec, im))
        from .depth import get_da3
        model = get_da3(cfg, getattr(cfg, "scale_model", None))  # optional separate model for the scale stage
        with _focal_probe() as frec:
            pred = model.inference(paths, extrinsics=np.stack(exts), intrinsics=np.stack([K] * len(paths)),
                                   infer_gs=False, align_to_input_ext_scale=False,
                                   process_res=cfg.depth_res)
        D = np.asarray(pred.depth, np.float32)
        h, w = D.shape[-2:]
        sx, sy = w / cam.width, h / cam.height
        r = []
        for k, (uv, z) in enumerate(obs):
            if not len(z):
                continue
            xi = np.clip((uv[:, 0] * sx).astype(int), 0, w - 1)
            yi = np.clip((uv[:, 1] * sy).astype(int), 0, h - 1)
            d = D[k][yi, xi]
            m = np.isfinite(d) & (d > 1e-6)
            if m.sum():
                r.append(d[m] / z[m])
        r = np.concatenate(r)
        s, src = float(np.median(r)), "da3"
        focal = _focal_correction(s, frec, len(paths), _focal_fix_enabled(cfg))
        s = focal["scale"]

    w = build_world(rec, X, s)
    Y = (X * s - w["ctr"]) @ w["A"].T - np.array([0, 0, w["floor"]])
    hist, edges = np.histogram(Y[:, 2], bins=200,
                               range=(np.percentile(Y[:, 2], 0.5), np.percentile(Y[:, 2], 99.5)))
    sig = np.flatnonzero(hist > 0.15 * hist.max())
    ceiling = float(edges[sig[-1] + 1])
    C = camera_centres(rec, s)
    cam_h = float(np.median((C - w["ctr"]) @ w["A"][2] - w["floor"]))
    plausible = (2.2 <= ceiling <= 3.6) and (1.1 <= cam_h <= 1.9)
    out = dict(scale=s, source=src, ceiling_m=ceiling, camera_height_m=cam_h,
               plausible=bool(plausible),
               scale_if_ceiling_260=float(2.6 / (ceiling / s)) if ceiling else None)
    if src == "da3":
        out.update({k: v for k, v in focal.items() if k != "scale"})
    return out


def _focal_correction(s_raw, frec, n_views, enabled):
    """-> dict(scale, s_raw, focal_fix, f_ratio_median/p10/p90 = f_pred/f_known over views, focal_note).
    If interception failed (not a nested model, renamed function, view count mismatch) - scale = s_raw, focal_fix False,
    a warning is logged, no failure."""
    out = dict(scale=s_raw, s_raw=s_raw, focal_fix=False, f_ratio_median=None, f_ratio_p10=None, f_ratio_p90=None,
               focal_note=None)
    fp, fk = np.asarray(frec["f_pred"], float), np.asarray(frec["f_known"], float)
    if frec["error"] or len(fp) != n_views or len(fk) != n_views or not (np.all(fp > 0) and np.all(fk > 0)):
        out["focal_note"] = (frec["error"] or
                             f"f_pred not intercepted: {len(fp)} of {n_views} views (model is not nested or the name changed)")
        print(f"[scale] WARNING: DA3 focal correction not applied - {out['focal_note']}; scale = s_raw", flush=True)
        return out
    q = fp / fk
    out.update(f_ratio_median=float(np.median(q)), f_ratio_p10=float(np.percentile(q, 10)),
               f_ratio_p90=float(np.percentile(q, 90)))
    if enabled:
        out["scale"] = float(s_raw * np.median(fk / fp))
        out["focal_fix"] = True
    else:
        out["focal_note"] = "disabled (TAPELINE_SCALE_FOCAL_FIX=0 / cfg.scale_focal_fix=False)"
    return out
