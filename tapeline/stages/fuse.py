"""Cloud assembly: TSDF fusion (default) or plain point concatenation.

Why TSDF: it averages the field value along each ray rather than point positions, so the noise
of tens of observations is reduced by roughly the square root of their count, and "free space"
erases flying pixels that concatenation can only accumulate.

Per-observation weights (conf, range, incidence angle) have a negligible effect on accuracy,
so the default weight is 1; conf weighting remains an option for setups with few
observations per voxel.

Two optional mechanisms act on the depth before integration:
  confidence cut-off (cfg.conf_min; 0 disables it): a pixel whose DA3 confidence is below the
       threshold is not an observation at all - neither surface nor free space. The lowest-confidence
       pixels have a much larger depth error than the rest.
  per-frame depth scale (cfg.frame_scale; frame_scale.json from the frame_scale stage): the depth
       of frame i is multiplied by s_i on top of the global scale.
Plus a visibility map (visibility.free_map) - the "free space" input for a downstream wall model.
"""
import json
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image

from .. import io as bio
from .visibility import free_map
from .world import build_world, camera_centres, poses_to_world, to_world


def _load(cfg):
    p = cfg.paths()
    z = np.load(p["depth"], allow_pickle=True)
    rec = pycolmap.Reconstruction(str(p["sfm"]))
    X = np.array([q.xyz for q in rec.points3D.values()])
    return z, rec, X


def _common(cfg, z, rec, X, scale):
    w = build_world(rec, X, scale)
    poses = np.asarray(z["poses"], np.float64)
    R, t = poses_to_world(w, poses, scale)
    K = np.asarray(z["K"], np.float64)
    imw, imh = [int(v) for v in z["img_wh"]]
    D = np.asarray(z["depth"], np.float32)
    hd, wd = D.shape[-2:]
    sx, sy = wd / imw, hd / imh
    intr = (K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy)
    return w, R, t, D, intr, (hd, wd), (imw, imh), (sx, sy)


def _frame_scales(cfg, names):
    """Per-frame depth multipliers from frame_scale.json; 1.0 if the stage is disabled, the file is missing
    or the frame has no points (its scale is left untouched). Second output: mask of frames that go into
    fusion: a frame whose scale deviates from the median by more than frame_scale_max_dev (in log) is dropped -
    there DA3 and SfM disagree not by percent but by a large factor (typically mirrors/glass), and rescaling
    the depth does not fix such a frame."""
    fs = np.ones(len(names), np.float64)
    keep = np.ones(len(names), bool)
    p = cfg.paths()["frame_scale"]
    if cfg.frame_scale and Path(p).exists():
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        by = dict(zip(j["names"], j["scales"]))
        fs = np.array([by.get(n, 1.0) for n in names], np.float64)
        if cfg.frame_scale_max_dev > 0:
            scaled = np.array([by.get(n, 1.0) != 1.0 for n in names])
            if scaled.any():
                med = np.median(np.log(fs[scaled]))
                keep = np.abs(np.log(fs) - med) <= cfg.frame_scale_max_dev
                keep |= ~scaled                      # a frame without its own scale (too few points) is not judged by it
    return fs, keep


def run(cfg, scale=1.0):
    return (tsdf if cfg.fuse == "tsdf" else concat)(cfg, scale)


def concat(cfg, scale):
    z, rec, X = _load(cfg)
    w, R, t, D, (fx, fy, cx, cy), (hd, wd), (imw, imh), (sx, sy) = _common(cfg, z, rec, X, scale)
    CF = np.asarray(z["conf"], np.float32) if bool(z["has_conf"]) else None
    names = [str(s) for s in z["names"]]
    fs, keep = _frame_scales(cfg, names)
    dev = "cuda"
    yy, xx = torch.meshgrid(torch.arange(hd, device=dev, dtype=torch.float32),
                            torch.arange(wd, device=dev, dtype=torch.float32), indexing="ij")
    dirs = torch.stack([(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], -1)
    gthr = (float(np.percentile(CF, cfg.conf_percentile))
            if (CF is not None and cfg.conf_percentile > 0 and cfg.conf_mode == "global") else None)
    acc = rgb_acc = None
    for i in range(len(D)):
        if not keep[i]:
            continue
        d = torch.as_tensor(D[i], device=dev) * (scale * fs[i])
        m = torch.isfinite(d) & (d > 0.05) & (d < cfg.max_depth)
        if CF is not None and cfg.conf_min > 0:
            m &= torch.as_tensor(CF[i], device=dev) >= cfg.conf_min
        if CF is not None and cfg.conf_percentile > 0:
            thr = gthr if gthr is not None else float(np.percentile(CF[i], cfg.conf_percentile))
            m &= torch.as_tensor(CF[i], device=dev) >= thr
        if not bool(m.any()):
            continue
        Xc = dirs[m] * d[m][:, None]
        Rt = torch.as_tensor(R[i], dtype=torch.float32, device=dev)
        tt = torch.as_tensor(t[i], dtype=torch.float32, device=dev)
        Xw = (Xc - tt) @ Rt
        img = np.asarray(Image.open(Path(cfg.paths()["frames_sel"]) / names[i]).convert("RGB"))
        ti = torch.as_tensor(img, device=dev)
        nz = torch.nonzero(m)
        cols = ti[(nz[:, 0].float() / sy).long().clamp(0, imh - 1),
                  (nz[:, 1].float() / sx).long().clamp(0, imw - 1)]
        v = torch.floor(Xw / cfg.voxel).to(torch.int64)
        key = v[:, 0] * 4_000_003 + v[:, 1] * 65_537 + v[:, 2]
        _, first = np.unique(key.cpu().numpy(), return_index=True)
        ft = torch.as_tensor(first, device=dev)
        acc = v[ft] if acc is None else torch.cat([acc, v[ft]])
        rgb_acc = cols[ft] if rgb_acc is None else torch.cat([rgb_acc, cols[ft]])
    key = acc[:, 0] * 4_000_003 + acc[:, 1] * 65_537 + acc[:, 2]
    _, first = np.unique(key.cpu().numpy(), return_index=True)
    ft = torch.as_tensor(first, device=acc.device)
    P = ((acc[ft].float() + 0.5) * cfg.voxel).cpu().numpy()
    bio.write_ply(cfg.paths()["cloud"], P, rgb_acc[ft].cpu().numpy())
    return dict(method="concat", points=int(len(P)), voxel=cfg.voxel, scale=scale,
                floor=w["floor"], cos_up=w["cos_agree"],
                conf_min=cfg.conf_min if CF is not None else 0.0, frames_scaled=int((fs != 1.0).sum()),
                frames_dropped=int((~keep).sum()))


def isolated_cameras(Cxy, rho, min_nb):
    """Outlier-camera mask: a camera with fewer than min_nb other cameras within rho m horizontally. A real trajectory
    is dense (adjacent frames 0.15–0.3 s apart are typically 0.01–0.3 m apart), while a false BA registration or
    a refill from a false anchor shows up as an isolated jump of 10–40 m."""
    C = np.asarray(Cxy, np.float64)
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    nb = (d2 < rho * rho).sum(1) - 1
    return nb < min_nb


def tsdf_box(cfg, X, w, Cw, scale):
    """-> lo, hi, dims, info. TSDF box: cloud percentiles & (camera bbox +/- max_depth). The camera bbox is built
    WITHOUT isolated cameras (isolated_cameras) if their share is at most box_cam_max_frac; otherwise all cameras
    are used. Without isolated cameras the box is identical. box_cam_rho = 0 disables the filtering."""
    Y = (X * scale - w["ctr"]) @ w["A"].T - np.array([0, 0, w["floor"]])
    # bounds from percentiles: SfM can produce outlier points at 10^8 m, and a min/max grid
    # would have billions of voxels
    lo = np.percentile(Y, 0.3, axis=0) - cfg.box_pad; lo[2] = -0.25
    hi = np.percentile(Y, 99.7, axis=0) + cfg.box_pad; hi[2] = cfg.z_max
    # ...and on a long walk even a 0.3 % tail (behind glass, sky) can stretch the box to ~10^9 voxels
    # (torch.nonzero fails at INT_MAX). TSDF integrates nothing beyond max_depth from a camera,
    # so camera bbox +/- max_depth is a tight bound; take the intersection
    rho = float(getattr(cfg, "box_cam_rho", 2.0))
    min_nb = int(getattr(cfg, "box_cam_min_nb", 3))
    max_frac = float(getattr(cfg, "box_cam_max_frac", 0.05))
    info = dict(cams=int(len(Cw)), cams_dropped=0, rho=rho, min_nb=min_nb)
    if rho > 0 and len(Cw) > min_nb + 1:
        iso = isolated_cameras(Cw[:, :2], rho, min_nb)
        n_iso = int(iso.sum())
        if 0 < n_iso <= max_frac * len(Cw):
            Cw = Cw[~iso]
            info["cams_dropped"] = n_iso
        elif n_iso:
            info["cams_isolated_not_dropped"] = n_iso
    lo[:2] = np.maximum(lo[:2], Cw[:, :2].min(0) - cfg.max_depth - cfg.box_pad)
    hi[:2] = np.minimum(hi[:2], Cw[:, :2].max(0) + cfg.max_depth + cfg.box_pad)
    dims = np.ceil((hi - lo) / cfg.voxel).astype(int)
    return lo, hi, dims, info


def tsdf(cfg, scale):
    t_start = time.time()
    z, rec, X = _load(cfg)
    w, R, t, D, (fx, fy, cx, cy), (hd, wd), (imw, imh), (sx, sy) = _common(cfg, z, rec, X, scale)
    CF = np.asarray(z["conf"], np.float32) if bool(z["has_conf"]) else None
    names = [str(s) for s in z["names"]]
    fs, keep = _frame_scales(cfg, names)
    dev = "cuda"
    lo, hi, dims, box_info = tsdf_box(cfg, X, w, to_world(w, camera_centres(rec), scale), scale)
    if int(np.prod(dims)) > 1_500_000_000:
        raise RuntimeError(f"TSDF grid {dims.tolist()} = {np.prod(dims) / 1e9:.1f} billion voxels at voxel {cfg.voxel} m; "
                           f"increase voxel or decrease max_depth (box {np.round(hi - lo, 1).tolist()} m)")

    vol = torch.ones(tuple(dims), dtype=torch.float32, device=dev)
    wacc = torch.zeros(tuple(dims), dtype=torch.float32, device=dev)
    racc = torch.zeros(tuple(dims) + (3,), dtype=torch.float32, device=dev)
    rwacc = torch.zeros(tuple(dims), dtype=torch.float32, device=dev)
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=dev)
    Dg = torch.as_tensor(D, device=dev)
    Cg = torch.as_tensor(CF, device=dev) if CF is not None else None
    conf_med = float(torch.median(Cg[Cg > 0])) if Cg is not None else 1.0

    def frame_terms(i):
        """Frustum of frame i on the grid: slice, valid-observation mask, sdf, pixel indices (or None)."""
        Rt = torch.as_tensor(R[i], dtype=torch.float32, device=dev)
        tt = torch.as_tensor(t[i], dtype=torch.float32, device=dev)
        centre = (-Rt.T @ tt).cpu().numpy()
        corners = [centre]
        for (u_, v_) in [(0, 0), (wd - 1, 0), (0, hd - 1), (wd - 1, hd - 1)]:
            ray = np.array([(u_ - cx) / fx, (v_ - cy) / fy, 1.0]); ray /= np.linalg.norm(ray)
            corners.append(centre + R[i].T @ (ray * cfg.max_depth))
        corners = np.stack(corners)
        i0 = np.clip(np.floor((corners.min(0) - lo) / cfg.voxel).astype(int) - 1, 0, dims - 1)
        i1 = np.clip(np.ceil((corners.max(0) - lo) / cfg.voxel).astype(int) + 1, 1, dims)
        if np.any(i1 <= i0):
            return None
        gx = torch.arange(i0[0], i1[0], device=dev, dtype=torch.float32)
        gy = torch.arange(i0[1], i1[1], device=dev, dtype=torch.float32)
        gz = torch.arange(i0[2], i1[2], device=dev, dtype=torch.float32)
        Xg = (gx[:, None, None] + 0.5) * cfg.voxel + lo_t[0]
        Yg = (gy[None, :, None] + 0.5) * cfg.voxel + lo_t[1]
        Zg = (gz[None, None, :] + 0.5) * cfg.voxel + lo_t[2]
        cxx = Rt[0, 0] * Xg + Rt[0, 1] * Yg + Rt[0, 2] * Zg + tt[0]
        cyy = Rt[1, 0] * Xg + Rt[1, 1] * Yg + Rt[1, 2] * Zg + tt[1]
        czz = Rt[2, 0] * Xg + Rt[2, 1] * Yg + Rt[2, 2] * Zg + tt[2]
        ok = (czz > 0.15) & (czz < cfg.max_depth)
        u = fx * cxx / czz.clamp(min=1e-6) + cx
        v = fy * cyy / czz.clamp(min=1e-6) + cy
        ok &= (u >= 0) & (u <= wd - 1) & (v >= 0) & (v <= hd - 1)
        if not bool(ok.any()):
            return None
        ui = u.clamp(0, wd - 1).long(); vi = v.clamp(0, hd - 1).long()
        dd = (Dg[i] * (scale * fs[i]))[vi, ui]
        ok &= dd > 0.05
        if Cg is not None and cfg.conf_min > 0:
            ok &= Cg[i][vi, ui] >= cfg.conf_min      # confidence cut-off: low conf is not an observation (neither surface nor free space)
        sdf = dd - czz
        ok &= sdf > -cfg.trunc
        sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
        return sl, ok, sdf, ui, vi, czz, dd

    # Protected carving (carve_protect). Pass 1: count how many frames saw each voxel as surface (|sdf| <= µ/2).
    # Pass 2: a surface supported by ≥ carve_protect frames is OPAQUE - if a pixel's DA3 depth lies beyond
    # the first protected voxel on its ray by more than µ, the depth is clamped to it: free space in front of the
    # surface, a surface vote on it, nothing behind it. This way mirrors, shower glass and glossy surfaces do not
    # carve through a wall that other frames observed (otherwise a large share of rays can pass through such
    # a wall). A doorway is not protected: it has no surface votes. Suppressing only the vote in the protected
    # voxel itself is not enough - the ray would still carve everything BEHIND the wall.
    protect = int(cfg.carve_protect)
    surf_cnt = torch.zeros(tuple(dims), dtype=torch.int16, device=dev) if protect > 0 else None
    n_protected_votes = 0
    if protect > 0:
        for i in range(len(D)):
            if not keep[i]:
                continue
            terms = frame_terms(i)
            if terms is None:
                continue
            sl, ok, sdf, _, _, _, _ = terms
            surf_cnt[sl] += (ok & (sdf.abs() <= 0.5 * cfg.trunc)).to(torch.int16)

    for i in range(len(D)):
        if not keep[i]:
            continue
        terms = frame_terms(i)
        if terms is None:
            continue
        sl, ok, sdf, ui, vi, czz, dd = terms
        if protect > 0:
            prot = ok & (surf_cnt[sl] >= protect)
            if bool(prot.any()):
                pix = vi * wd + ui                                  # pixel of each frustum voxel
                dprot = torch.full((hd * wd,), float("inf"), device=dev)
                dprot.scatter_reduce_(0, pix[prot].reshape(-1), czz[prot].reshape(-1), reduce="amin")
                dp = dprot[pix]                                     # depth of the first protected surface on the pixel's ray
                through = ok & (dd > dp + cfg.trunc)
                n_protected_votes += int(through.sum())
                dd = torch.where(through, dp, dd)
                sdf = dd - czz
                ok = ok & (sdf > -cfg.trunc)
        val = (sdf / cfg.trunc).clamp(-1, 1)
        if cfg.weights == "conf" and Cg is not None:
            wgt = Cg[i][vi, ui] / conf_med
        else:
            wgt = torch.ones_like(val)
        wgt = torch.where(ok, wgt, torch.zeros_like(wgt))
        W0, T0 = wacc[sl], vol[sl]
        Wn = W0 + wgt
        vol[sl] = torch.where(Wn > 0, (T0 * W0 + val * wgt) / Wn.clamp(min=1e-9), T0)
        img = np.asarray(Image.open(Path(cfg.paths()["frames_sel"]) / names[i]).convert("RGB"))
        ti = torch.as_tensor(img, dtype=torch.float32, device=dev)
        near = wgt * (val.abs() < 1.0)
        racc[sl] += ti[(vi.float() / sy).long().clamp(0, imh - 1),
                       (ui.float() / sx).long().clamp(0, imw - 1)] * near[..., None]
        rwacc[sl] += near
        wacc[sl] = Wn

    def _extract(weight_min):  # surface extraction from the finished TSDF (a function so a second cloud can be extracted)
        surf = (wacc >= weight_min) & (vol.abs() <= cfg.surface_band)
        idx = torch.nonzero(surf)
        P = (idx.float() + 0.5) * cfg.voxel + lo_t
        if cfg.subvoxel:
            # the voxel centre lies within the +/-surface_band*µ band, not on the surface: move the point
            # onto the zero isosurface along the normal n = grad tsdf/|grad tsdf|
            g = torch.zeros((len(idx), 3), device=dev)
            for ax in range(3):
                ip, im_ = idx.clone(), idx.clone()
                ip[:, ax] = (ip[:, ax] + 1).clamp(max=dims[ax] - 1)
                im_[:, ax] = (im_[:, ax] - 1).clamp(min=0)
                g[:, ax] = (vol[ip[:, 0], ip[:, 1], ip[:, 2]] - vol[im_[:, 0], im_[:, 1], im_[:, 2]]) / 2
            n = g / g.norm(dim=1, keepdim=True).clamp(min=1e-9)
            P = P - vol[surf][:, None] * cfg.trunc * n
        rgb = (racc[surf] / rwacc[surf].clamp(min=1e-6)[:, None]).clamp(0, 255)
        Pn, rgbn = P.cpu().numpy(), rgb.cpu().numpy().astype(np.uint8)
        nocol = (rwacc[surf] <= 0).cpu().numpy()
        if nocol.any() and (~nocol).any():
            from scipy.spatial import cKDTree
            _, j = cKDTree(Pn[~nocol]).query(Pn[nocol], k=1, workers=-1)
            rgbn[nocol] = rgbn[~nocol][j]
        return surf, Pn, rgbn, nocol

    surf, Pn, rgbn, nocol = _extract(cfg.weight_min)
    bio.write_ply(cfg.paths()["cloud"], Pn, rgbn)
    if getattr(cfg, "cloud_wm1", False):  # second cloud with weight_min=1 from the same TSDF (denser, for downstream detection)
        _s1, P1, rgb1, _n1 = _extract(1.0)
        bio.write_ply(str(Path(cfg.work) / "cloud_wm1.ply"), P1, rgb1)
    res = dict(method="tsdf", points=int(len(Pn)), voxel=cfg.voxel, trunc=cfg.trunc,
               scale=scale, floor=w["floor"], cos_up=w["cos_agree"],
               grid=[int(x) for x in dims], no_colour=int(nocol.sum()),
               conf_min=cfg.conf_min if Cg is not None else 0.0, frames_scaled=int((fs != 1.0).sum()),
               frames_dropped=int((~keep).sum()), frame_scale_max_dev=cfg.frame_scale_max_dev,
               carve_protect=protect, protected_votes=n_protected_votes,
               protected_voxels=int((surf_cnt >= protect).sum()) if surf_cnt is not None else 0)
    if cfg.free_map:
        # input for a downstream wall model: float16 cloud, empty quads, visibility map
        F = free_map(vol, wacc, lo, cfg.voxel, cfg.free_band)
        np.savez_compressed(cfg.paths()["wallfield"], xyz=Pn.astype(np.float16),
                            quads=np.zeros((0, 4, 3), np.float32), free=F.astype(np.float16))
        res.update(free_cells=int(len(F)), free_band=list(cfg.free_band))
    res["box"] = box_info
    res["wall"] = time.time() - t_start
    return res
