"""Per-frame depth scale: scale of DA3 depths from sparse SfM points.

In posed mode DA3 is tied to the poses by one sim3 per pass, while the scale of individual
frames can still deviate. A scale error of a fraction of a percent already means millimetres to
centimetres of systematic error per frame at room distances, which TSDF averages into a "thick" wall and spurious ridges.

Without a reference the problem is ill-conditioned (a z-buffer consensus and a pairwise
Gauss–Newton over log-scales are both unstable: the signal is of the same order as the
biases of pairwise residuals). With a reference the problem is cheap and well posed, and a reference
is available for every frame: the triangulated points from the SfM stage.

Frame scale s_i = exp(median log(z_sfm / d_da3)) over the frame's points with conf ≥ conf_min
after 3·MAD outlier rejection. The factor is dimensionless: DA3 depths and SfM z are in the
same units (DA3 is aligned to the SfM poses); the global scale is applied on top in fuse.
"""
import json
import time

import numpy as np


def frame_scale_from_obs(d, z, conf=None, conf_min=0.0, min_pts=20):
    """Scale of one frame from pairs (DA3 depth d, SfM depth z) at the point pixels.
    -> (s, n_used, mad_log); s = 1.0 and n_used = 0 if fewer than min_pts valid pairs."""
    d = np.asarray(d, np.float64)
    z = np.asarray(z, np.float64)
    m = np.isfinite(d) & (d > 0.05) & (z > 1e-6)
    if conf is not None:
        m &= np.asarray(conf, np.float64) >= conf_min
    if m.sum() < min_pts:
        return 1.0, 0, 0.0
    r = np.log(z[m] / d[m])
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-4
    rr = r[np.abs(r - med) < 3 * 1.4826 * mad]
    if len(rr) < min_pts:
        return 1.0, 0, 0.0
    m2 = np.median(rr)
    return float(np.exp(m2)), int(len(rr)), float(np.median(np.abs(rr - m2)))


def estimate(D, C, obs, conf_min=0.0, min_pts=20):
    """obs - per frame: (xi, yi, z) - pixel indices in the depth grid and the SfM depth of the point.
    -> scales (N,), n_used (N,), mad_log (N,)"""
    N = len(D)
    s, n, mad = np.ones(N, np.float64), np.zeros(N, int), np.zeros(N)
    for i, (xi, yi, z) in enumerate(obs):
        if len(z) == 0:
            continue
        d = np.asarray(D[i], np.float32)[yi, xi]
        c = np.asarray(C[i], np.float32)[yi, xi] if C is not None else None
        s[i], n[i], mad[i] = frame_scale_from_obs(d, z, c, conf_min, min_pts)
    return s, n, mad


def diagnostics(s, n, mad):
    ok = n > 0
    ls = np.log(s[ok])
    out = dict(frames=int(len(s)), frames_scaled=int(ok.sum()), coverage=float(ok.mean()) if len(s) else 0.0)
    if ok.any():
        dev = np.abs(ls - np.median(ls))
        out.update(pts_per_frame_median=float(np.median(n[ok])), pts_p10=float(np.percentile(n[ok], 10)),
                   log_s_median=float(np.median(ls)), abs_dev_p50=float(np.median(dev)),
                   abs_dev_p90=float(np.percentile(dev, 90)), within_frame_mad_median=float(np.median(mad[ok])))
    return out


def _observations(rec, im, max_err):
    """Frame points: the observation (pixel where the point is actually seen - no distortion model)
    and the point depth in the camera; points with reprojection error above max_err are skipped."""
    pose = im.cam_from_world()
    R, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
    uv, z = [], []
    for p2 in im.points2D:
        if not p2.has_point3D():
            continue
        p3 = rec.points3D[p2.point3D_id]
        if p3.error > max_err:
            continue
        Xc = R @ p3.xyz + t
        if Xc[2] > 1e-6:
            uv.append(p2.xy)
            z.append(Xc[2])
    return np.asarray(uv, np.float64).reshape(-1, 2), np.asarray(z, np.float64)


def run(cfg, scale=1.0):
    """scale - global scale (metres per SfM unit); only needed for the range window in metres."""
    import pycolmap   # lazy import: the estimator above is tested without pycolmap

    t0 = time.time()
    p = cfg.paths()
    z = np.load(p["depth"], allow_pickle=True)
    D = np.asarray(z["depth"], np.float32)
    C = np.asarray(z["conf"], np.float32) if bool(z["has_conf"]) else None
    names = [str(s) for s in z["names"]]
    imw, imh = [int(v) for v in z["img_wh"]]
    hd, wd = D.shape[-2:]
    sx, sy = wd / imw, hd / imh
    rec = pycolmap.Reconstruction(str(p["sfm"]))
    by = {im.name: im for im in rec.images.values()}
    obs = []
    for nm in names:
        im = by.get(nm)
        if im is None:
            obs.append((np.zeros(0, int), np.zeros(0, int), np.zeros(0)))
            continue
        uv, zz = _observations(rec, im, cfg.frame_scale_max_err)
        zm = zz * scale
        keep = (zm >= cfg.frame_scale_z_min) & (zm <= cfg.frame_scale_z_max)
        xi = np.clip((uv[keep, 0] * sx).astype(int), 0, wd - 1)
        yi = np.clip((uv[keep, 1] * sy).astype(int), 0, hd - 1)
        obs.append((xi, yi, zz[keep]))
    conf_min = cfg.conf_min if C is not None else 0.0
    s, n, mad = estimate(D, C, obs, conf_min, cfg.frame_scale_min_pts)
    diag = diagnostics(s, n, mad)
    diag.update(conf_min=conf_min, max_err=cfg.frame_scale_max_err, wall=round(time.time() - t0, 2))
    out = dict(names=names, scales=[float(v) for v in s], n_pts=[int(v) for v in n], **diag)
    with open(p["frame_scale"], "w") as f:
        json.dump(out, f)
    return diag
