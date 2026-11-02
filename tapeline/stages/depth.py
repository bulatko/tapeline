"""Depth: DA3 in posed mode on the model's poses, a single pass over all frames.

Chunking is unnecessary and harmful: memory grows only linearly with the frame count, while
splitting introduces seams between independently aligned pieces.
It is enabled only as a response to OOM.
"""
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch


def _load_da3(name, fast_init=False):
    from depth_anything_3.api import DepthAnything3
    if fast_init:  # build the model directly on the GPU (much faster than CPU init + transfer)
        with torch.device("cuda"):
            return DepthAnything3.from_pretrained(name).to("cuda")
    return DepthAnything3.from_pretrained(name).to("cuda")


def get_da3(cfg, name=None):
    """With model_cache/prefetch DA3 is loaded once per process (depth + scale), otherwise on every call."""
    name = name or cfg.depth_model
    if getattr(cfg, "model_cache", False) or getattr(cfg, "prefetch", False):
        from . import _models
        return _models.get(("da3", name), lambda: _load_da3(name, getattr(cfg, "fast_init", False)))
    return _load_da3(name, getattr(cfg, "fast_init", False))


def _batch(model, paths, exts, ixts, res, align=True):
    if not align:
        # DA3 ALWAYS calls evo Umeyama (api._align_to_input_extrinsics_intrinsics) when extrinsics are passed;
        # align_to_input_ext_scale=False only changes how the result is applied, so the fallback would fail with the same
        # exception. Without poses there is no alignment; the chunk's scale is then brought to model units via SfM observations (see run).
        return model.inference(paths, infer_gs=False, process_res=res)
    return model.inference(paths, extrinsics=np.stack(exts), intrinsics=np.stack(ixts),
                           infer_gs=False, process_res=res)


def _to_model_units(rec, imgs, D):
    """Factor "DA3 depth -> model units": median of z_SfM / d_DA3 over the observations (as in the scale stage)."""
    from .scale import _observations
    cam = list(rec.cameras.values())[0]
    h, w = D.shape[-2:]
    sx, sy = w / cam.width, h / cam.height
    r = []
    for k, im in enumerate(imgs):
        uv, z = _observations(rec, im)
        if not len(z):
            continue
        d = D[k][np.clip((uv[:, 1] * sy).astype(int), 0, h - 1), np.clip((uv[:, 0] * sx).astype(int), 0, w - 1)]
        m = np.isfinite(d) & (d > 1e-6)
        if m.any():
            r.append(z[m] / d[m])
    return float(np.median(np.concatenate(r)))


def run(cfg):
    p = cfg.paths()
    rec = pycolmap.Reconstruction(str(p["sfm"]))
    cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy = cam.params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    # frames without 3D observations (fragments cut off by sfm.prune_stray, or frames without matches) are not
    # anchored in the main map - their poses are unverified, so they are excluded from depth and fusion
    n_all = len(imgs)
    imgs = [im for im in imgs if any(p2.has_point3D() for p2 in im.points2D)]
    skipped = n_all - len(imgs)

    paths, exts, names = [], [], []
    for im in imgs:
        pose = im.cam_from_world()
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = pose.rotation.matrix(); E[:3, 3] = pose.translation
        paths.append(str(Path(p["frames_sel"]) / im.name))
        exts.append(E); names.append(im.name)
    ixts = [K] * len(paths)

    model = get_da3(cfg)

    t0 = time.time()
    chunk = cfg.depth_chunk or len(paths)
    depth, conf = [], []
    has_conf = None
    # posed DA3 aligns its poses to ours via evo Umeyama and fails with "Degenerate covariance rank" on a nearly
    # degenerate trajectory. Fallback: no alignment, then the depth is converted to model units with a single
    # factor from SfM observations - the output is in the same units as usual.
    align, unit, units = True, 1.0, []
    while True:
        try:
            # Chunks are EQUAL. range(0, n, chunk) after OOM halving can leave a tail of 1–2 frames (e.g. 297+297+1),
            # and Umeyama on 1–2 frames is always degenerate. Hence ceil(n/chunk) chunks of n//k..n//k+1 frames each.
            n_ch = -(-len(paths) // chunk)
            bounds = [len(paths) * i // n_ch for i in range(n_ch + 1)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                from ._models import tf32
                with tf32(cfg):
                    pred = _batch(model, paths[s:e], exts[s:e], ixts[s:e], cfg.depth_res, align=align)
                d = np.asarray(pred.depth, np.float32)
                if not align:       # pose-free DA3: each chunk has its own scale -> its own factor
                    u = _to_model_units(rec, imgs[s:e], d)
                    units.append(u)
                    d = d * u
                c = getattr(pred, "conf", None)
                has_conf = c is not None if has_conf is None else has_conf
                depth.append(d.astype(np.float16))
                if has_conf:
                    conf.append(np.asarray(c, np.float32).astype(np.float16))
            break
        except torch.cuda.OutOfMemoryError:
            if chunk <= 24:
                raise
            chunk = max(24, chunk // 2)
            depth, conf, has_conf, units = [], [], None, []
            torch.cuda.empty_cache()
        except Exception as e:                      # evo.core.geometry.GeometryException etc. from the alignment
            if not align or "Degenerate covariance" not in str(e):
                raise
            align = False
            depth, conf, has_conf, units = [], [], None, []
            torch.cuda.empty_cache()

    D = np.concatenate(depth)
    if units:
        unit = float(np.median(units))
    np.savez(p["depth"], depth=D, conf=np.concatenate(conf) if has_conf else np.zeros(0, np.float16),
             poses=np.stack(exts), names=np.array(names), K=K, has_conf=bool(has_conf),
             img_wh=np.array([cam.width, cam.height]))
    return dict(frames=len(paths), skipped_no_obs=skipped, shape=list(D.shape), chunk=chunk, has_conf=bool(has_conf),
                align="umeyama" if align else "fallback_noalign", unit_factor=unit, unit_factors=units,
                wall=time.time() - t0, peak_gb=float(torch.cuda.max_memory_allocated() / 1e9))
