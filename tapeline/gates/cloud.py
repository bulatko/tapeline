"""Cloud gate: planarity of the floor and the ceiling.

Floor and ceiling are physically flat - a property of the room, not of the reconstruction,
so the measure depends neither on cloud density nor on SfM errors. The residual is measured
against a FITTED plane, not the horizontal: the vertical is only estimated approximately from
the cameras, and a 1 deg tilt over 8 meters gives 14 cm of spurious error.

Interpret with care: the metric's resolution is limited by the voxel size
(the median residual cannot be much smaller than half a cell).
"""
import numpy as np

from .. import io as bio


def _fit_plane(P, iters=40, thr=0.03, seed=0):
    rng = np.random.default_rng(seed)
    best, bn, bd = -1, None, None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ s[0]
        cnt = int((np.abs(P @ n + d) < thr).sum())
        if cnt > best:
            best, bn, bd = cnt, n, d
    inl = np.abs(P @ bn + bd) < thr
    Q = P[inl] - P[inl].mean(0)
    _, V = np.linalg.eigh(Q.T @ Q)
    n = V[:, 0]
    return np.abs(P @ n + (-n @ P[inl].mean(0)))


def cloud_gate(ply, ceiling_hint=None, max_floor_outliers=0.03, min_points=100_000):
    P, _ = bio.read_ply(ply)
    out = dict(name="cloud", points=int(len(P)))
    floor = P[np.abs(P[:, 2]) < 0.12]
    if len(floor) > 2000:
        r = _fit_plane(floor)
        out.update(floor_median_cm=round(float(np.median(r)) * 100, 2),
                   floor_p90_cm=round(float(np.percentile(r, 90)) * 100, 2),
                   floor_outliers=round(float((r > 0.05).mean()), 4))
    top = ceiling_hint if ceiling_hint else float(np.percentile(P[:, 2], 99))
    ceil = P[(P[:, 2] > top - 0.30) & (P[:, 2] < top + 0.25)]
    if len(ceil) > 2000:
        r = _fit_plane(ceil)
        out.update(ceiling_median_cm=round(float(np.median(r)) * 100, 2),
                   ceiling_p90_cm=round(float(np.percentile(r, 90)) * 100, 2))
    out["ok"] = bool(len(P) >= min_points and
                     out.get("floor_outliers", 1.0) <= max_floor_outliers)
    out["thresholds"] = dict(floor_outliers=max_floor_outliers, min_points=min_points)
    return out
