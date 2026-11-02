"""Canonical scene world: metres, Z up, floor at zero.

Three pitfalls:
  * the sign of the vertical must NOT be taken from mean (C - centre)·up - camera centres
    are symmetric around their mean, so the sign comes out random. We use the averaged
    camera Y axis instead: in the CV convention it points down the image, and a phone is
    held nearly upright.
  * the floor is the lowest SIGNIFICANT peak of the height histogram, not the minimum and
    not the densest peak (beds and tables can be denser than a partly occluded floor).
  * the vertical must NOT be taken as the third (smallest) PCA axis of camera centres: in
    recordings where the operator barely moved (a spot of a metre or two rather than tens
    of metres), that axis is noise of the trajectory shape, not gravity, and can tilt the
    world by many degrees. The normal of a RANSAC floor plane is used instead.
"""
import numpy as np

FLOOR_NEAR_M = 8.0   # the floor is searched among points no farther than this from the cameras (see build_world)


def camera_centres(rec, scale=1.0):
    return np.array([(-im.cam_from_world().rotation.matrix().T @ im.cam_from_world().translation)
                     for im in rec.images.values()]) * scale


def _fit_floor_plane(P, iters=1500, thr=0.03, seed=0):
    """RANSAC plane fit to a point cloud (for the floor): normal + inlier count. Deterministic
    (rng with a fixed seed) -- reproducible for the same input points."""
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        A = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(A[1] - A[0], A[2] - A[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        c = int((np.abs((P - A[0]) @ n) < thr).sum())
        if c > best[0]:
            best = (c, (n, A[0].copy()))
    n, p0 = best[1]
    for _ in range(5):
        m = np.abs((P - p0) @ n) < thr
        Q = P[m]
        p0 = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - p0, full_matrices=False)
        n = vt[2]
    return n / np.linalg.norm(n), int(m.sum())


def build_world(rec, points_xyz, scale=1.0):
    """-> dict(A, ctr, floor, shift): X_m = A @ (scale*X_w - ctr), then z -= floor.
    The vertical (up) is the normal of the RANSAC floor plane (lowest 15% of points by a rough
    height estimate from the averaged camera orientation), NOT the third PCA axis of camera
    centres (see the module docstring, pitfall 3)."""
    C = camera_centres(rec, scale)
    # A BA model can contain isolated garbage cameras (orders of magnitude away from the rest) with equally garbage
    # points. They would drag the mean of the centres kilometres away and stretch the floor histogram range over the
    # whole space. Cameras farther than max(10·MAD, 20 m) from the median centre are dropped; centre, axes and floor
    # are built from the rest, the floor only from points within FLOOR_NEAR_M of the remaining cameras.
    dC = np.linalg.norm(C - np.median(C, 0), axis=1)
    C = C[dC <= max(10.0 * np.median(dC), 20.0)]
    ctr = C.mean(0)
    X = np.asarray(points_xyz) * scale
    from scipy.spatial import cKDTree
    near = cKDTree(C).query(X, distance_upper_bound=FLOOR_NEAR_M)[0] < np.inf
    if near.sum() >= 100:
        X = X[near]
    down = np.mean([np.asarray(im.cam_from_world().rotation.matrix()).T @ np.array([0., 1., 0.])
                    for im in rec.images.values()], axis=0)
    down /= np.linalg.norm(down)
    h = X @ (-down)
    lo = X[h < np.percentile(h, 15)]
    rng = np.random.default_rng(0)
    lo = lo[rng.choice(len(lo), min(40000, len(lo)), replace=False)]
    up, floor_ransac_inliers = _fit_floor_plane(lo)
    if up @ down > 0:
        up = -up
    cos_agree = float(-up @ down)
    _, _, vt = np.linalg.svd(C - ctr, full_matrices=True)
    e1 = vt[0] - (vt[0] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, up)
    A = np.stack([e1, e2, up])
    Y = (X - ctr) @ A.T
    floor, first = floor_height(Y[:, 2])
    return dict(A=A, ctr=ctr, floor=floor, up=up, cos_agree=cos_agree, floor_first_bin=first,
                floor_ransac_inliers=floor_ransac_inliers,
                shift=-A @ ctr - np.array([0, 0, floor]))


def floor_height(z, bins=200, frac=0.15):
    """Lowest significant PEAK of the height histogram (the floor mode), not the first significant bin: the first bin
    is the lower edge of a peak blurred by triangulation noise and sits systematically below the floor by half the
    peak width (which would show up as "floor outliers" in the cloud gate).
    -> (floor, first_significant_bin) - the latter for provenance."""
    z = np.asarray(z, np.float64)
    hist, edges = np.histogram(z, bins=bins, range=(np.percentile(z, 0.5), np.percentile(z, 99.5)))
    first = int(np.flatnonzero(hist > frac * hist.max())[0])
    peak = first
    while peak + 1 < len(hist) and hist[peak + 1] >= hist[peak]:
        peak += 1
    centre = float((edges[peak] + edges[peak + 1]) / 2)
    return centre, float(edges[first])


def poses_to_world(w, extrinsics, scale):
    """w2c poses in the source world -> w2c poses in the canonical one (metres, Z up, floor 0)."""
    R = np.einsum('nij,jk->nik', extrinsics[:, :3, :3], w["A"].T)
    t = scale * extrinsics[:, :3, 3] - np.einsum('nij,j->ni', R, w["shift"])
    return R, t


def to_world(w, xyz, scale):
    return (np.asarray(xyz) * scale - w["ctr"]) @ w["A"].T - np.array([0, 0, w["floor"]])
