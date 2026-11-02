"""Pure math for repairing weak segments: Umeyama sim3 and a VGGT chain between anchors (numpy + scipy,
no pycolmap - tested on synthetic data)."""
import numpy as np


def sim3(X, Y):
    """Umeyama X->Y: (s, R, t) with Y ~ s·R·X + t."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (Xc ** 2).sum()
    return s, R, my - s * R @ mx


def align_segment(model_c2w_anchors, prior_c2w_anchors, prior_c2w_inner, step=None):
    """Segment frame poses obtained by moving the WHOLE prior segment into the model world with a single sim3.

    The sim3 is estimated from anchor frames - well-connected neighbours of the segment on both sides (lists of c2w
    in the model world and in the prior world, ≥ 2 frames). To keep the rotation about the "anchor A – anchor B" axis
    determined even for two nearly collinear anchors, points on each anchor's optical axis (centre + R·e_z·step) are
    added to the centres: the alignment uses both positions and orientations. Unlike chain_poses, the error of the
    prior's relative motions does not accumulate along the segment.
    -> (list of c2w of the inner frames, s, R, t)."""
    A = [np.array(M, float) for M in model_c2w_anchors]
    Pa = [np.array(M, float) for M in prior_c2w_anchors]
    Cm = np.array([M[:3, 3] for M in A]); Cp = np.array([M[:3, 3] for M in Pa])
    Zm = np.array([M[:3, 2] for M in A]); Zp = np.array([M[:3, 2] for M in Pa])
    if step is None:                                   # length of the "whisker" along the optical axis - on the order of the camera step
        d = np.linalg.norm(np.diff(Cm, axis=0), axis=1)
        step = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    # preliminary prior->model scale from the spread of anchor centres: the prior whisker must have the same length in model units
    s0 = float(np.linalg.norm(Cm - Cm.mean(0), axis=1).sum() / max(np.linalg.norm(Cp - Cp.mean(0), axis=1).sum(), 1e-12))
    Xm = np.concatenate([Cm, Cm + Zm * step])
    Xp = np.concatenate([Cp, Cp + Zp * (step / max(s0, 1e-12))])
    s, R, t = sim3(Xp, Xm)
    out = []
    for M in prior_c2w_inner:
        M = np.array(M, float)
        N = np.eye(4)
        N[:3, :3] = R @ M[:3, :3]
        N[:3, 3] = s * (R @ M[:3, 3]) + t
        out.append(N)
    return out, s, R, t


def chain_poses(c2w_a, c2w_b, prior_c2w, scale):
    """Poses of frames between two anchors from the prior's relative motions.
    c2w_a, c2w_b - anchor c2w in the model world (4x4; c2w_b may be None - leading anchor only);
    prior_c2w - list of prior c2w for [anchor A, frames..., anchor B] in the prior's own world; scale - model units per
    prior unit (converts translations). The predictions from A and from B are blended by frame index (translation
    linearly, rotation via slerp).
    -> list of c2w for the inner frames (anchors excluded)."""
    from scipy.spatial.transform import Rotation, Slerp
    P = [np.array(M, float).copy() for M in prior_c2w]
    for M in P:
        M[:3, 3] *= scale
    n = len(P)
    fwd = [np.array(c2w_a, float)]
    for k in range(1, n):
        fwd.append(fwd[-1] @ (np.linalg.inv(P[k - 1]) @ P[k]))
    if c2w_b is None:
        return fwd[1:]
    bwd = [None] * n
    bwd[-1] = np.array(c2w_b, float)
    for k in range(n - 2, -1, -1):
        bwd[k] = bwd[k + 1] @ (np.linalg.inv(P[k + 1]) @ P[k])
    out = []
    for k in range(1, n - 1):
        w = k / (n - 1)
        A, B = fwd[k], bwd[k]
        M = np.eye(4)
        M[:3, 3] = (1 - w) * A[:3, 3] + w * B[:3, 3]
        M[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))(w).as_matrix()
        out.append(M)
    return out
