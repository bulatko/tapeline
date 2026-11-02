"""Convention gate: DA3 depths vs. depths of triangulated points.

Catches swapped axes, w2c/c2w order and a diverged scale BEFORE they turn into a
plausible-looking mush of a cloud. With correct conventions the DA3/SfM ratio is
~ 1 with a narrow spread.

This is the only meaningful gate in refine=False mode, where there is nothing to
measure reprojection against.
"""
import numpy as np


def depth_convention_gate(model_dir, depth_npz, max_offset=0.10, max_spread=1.35):
    import pycolmap   # lazy: the gates package is also imported without pycolmap (tests)
    z = np.load(depth_npz, allow_pickle=True)
    D = np.asarray(z["depth"], np.float32)
    names = [str(s) for s in z["names"]]
    imw, imh = [int(v) for v in z["img_wh"]]
    hd, wd = D.shape[-2:]
    sx, sy = wd / imw, hd / imh

    rec = pycolmap.Reconstruction(str(model_dir))
    by = {im.name: im for im in rec.images.values()}
    ratios = []
    for i, nm in enumerate(names):
        im = by.get(nm)
        if im is None:
            continue
        pose = im.cam_from_world()
        R, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
        uv, zz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                Xc = R @ rec.points3D[p2.point3D_id].xyz + t
                if Xc[2] > 1e-6:
                    uv.append(p2.xy); zz.append(Xc[2])
        if not zz:
            continue
        uv, zz = np.asarray(uv), np.asarray(zz)
        xi = np.clip((uv[:, 0] * sx).astype(int), 0, wd - 1)
        yi = np.clip((uv[:, 1] * sy).astype(int), 0, hd - 1)
        d = D[i][yi, xi]
        m = np.isfinite(d) & (d > 1e-6)
        if m.sum():
            ratios.append(d[m] / zz[m])
    r = np.concatenate(ratios)
    med = float(np.median(r))
    spread = float(np.percentile(r, 75) / np.percentile(r, 25))
    ok = abs(med - 1.0) <= max_offset and spread <= max_spread
    return dict(name="depth_convention", ok=bool(ok), median_ratio=round(med, 4),
                spread_p75_p25=round(spread, 4), observations=int(len(r)),
                thresholds=dict(offset=max_offset, spread=max_spread))
