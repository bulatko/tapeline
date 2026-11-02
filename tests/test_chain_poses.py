"""VGGT chain between anchors: if the prior's relative motions are exact (prior = ground truth up to a global
sim3), the propagated poses match the ground truth; with a noisy prior, blending from two anchors beats one."""
import numpy as np
from scipy.spatial.transform import Rotation

from tapeline.stages.chain import chain_poses, align_segment, sim3 as _sim3


def _traj(n, seed=0):
    rng = np.random.default_rng(seed)
    out, M = [], np.eye(4)
    for k in range(n):
        step = np.eye(4)
        step[:3, :3] = Rotation.from_euler("xyz", rng.normal(0, 0.05, 3)).as_matrix()
        step[:3, 3] = [0.1, 0.02 * rng.normal(), 0.0]
        M = M @ step
        out.append(M.copy())
    return out


def _to_prior(truth, scale, R, t):
    """ground truth -> prior world: X_p = R (X/scale) + t"""
    out = []
    for M in truth:
        P = np.eye(4)
        P[:3, :3] = R @ M[:3, :3]
        P[:3, 3] = R @ (M[:3, 3] / scale) + t
        out.append(P)
    return out


def test_exact_prior_reproduces_truth():
    truth = _traj(12)
    Rg = Rotation.from_euler("z", 0.7).as_matrix()
    prior = _to_prior(truth, 4.0, Rg, np.array([1.0, -2.0, 0.5]))
    est = chain_poses(truth[0], truth[-1], prior, 4.0)
    for M, T in zip(est, truth[1:-1]):
        assert np.allclose(M[:3, 3], T[:3, 3], atol=1e-6)
        assert np.allclose(M[:3, :3], T[:3, :3], atol=1e-6)


def test_single_anchor_forward():
    truth = _traj(8)
    prior = _to_prior(truth, 2.0, np.eye(3), np.zeros(3))
    est = chain_poses(truth[0], None, prior, 2.0)
    assert len(est) == 7
    assert np.allclose(est[-1][:3, 3], truth[-1][:3, 3], atol=1e-6)


def test_blend_beats_single_anchor_under_drift():
    truth = _traj(20, seed=1)
    prior = _to_prior(truth, 1.0, np.eye(3), np.zeros(3))
    for k, P in enumerate(prior):               # linear prior drift: error grows with the frame index
        P[:3, 3] += np.array([0.0, 0.0, 0.01 * k])
    one = chain_poses(truth[0], None, prior, 1.0)
    two = chain_poses(truth[0], truth[-1], prior, 1.0)
    e1 = np.mean([np.linalg.norm(M[:3, 3] - T[:3, 3]) for M, T in zip(one[:-1], truth[1:-1])])
    e2 = np.mean([np.linalg.norm(M[:3, 3] - T[:3, 3]) for M, T in zip(two, truth[1:-1])])
    assert e2 < e1


def test_align_segment_exact_prior():
    """prior = ground truth up to sim3; anchors - 3 frames on each side of segment 6..13 out of 20."""
    truth = _traj(20, seed=2)
    Rg = Rotation.from_euler("xyz", [0.3, -0.2, 0.9]).as_matrix()
    prior = _to_prior(truth, 3.0, Rg, np.array([2.0, 1.0, -1.0]))
    anchors = [3, 4, 5, 14, 15, 16]
    inner = list(range(6, 14))
    est, s, R, t = align_segment([truth[k] for k in anchors], [prior[k] for k in anchors], [prior[k] for k in inner])
    assert abs(s - 3.0) < 1e-6
    for M, k in zip(est, inner):
        assert np.allclose(M[:3, 3], truth[k][:3, 3], atol=1e-6)
        assert np.allclose(M[:3, :3], truth[k][:3, :3], atol=1e-6)


def test_align_segment_no_drift_accumulation():
    """Noise in the prior's relative motions: the chain accumulates it along the segment, the sim3 transfer does not."""
    truth = _traj(40, seed=3)
    rng = np.random.default_rng(5)
    prior = _to_prior(truth, 1.0, np.eye(3), np.zeros(3))
    for P in prior:                                  # independent pose noise per prior frame (not drift)
        P[:3, 3] += rng.normal(0, 0.01, 3)
    a, b = 8, 31
    chain = chain_poses(truth[a], truth[b], prior[a:b + 1], 1.0)
    anchors = list(range(a - 3, a + 1)) + list(range(b, b + 4))
    inner = list(range(a + 1, b))
    al, _, _, _ = align_segment([truth[k] for k in anchors], [prior[k] for k in anchors], [prior[k] for k in inner])
    e_chain = np.mean([np.linalg.norm(M[:3, 3] - truth[k][:3, 3]) for M, k in zip(chain, inner)])
    e_align = np.mean([np.linalg.norm(M[:3, 3] - truth[k][:3, 3]) for M, k in zip(al, inner)])
    assert e_align < e_chain


def test_sim3_recovers_transform():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    R = Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    Y = 3.0 * (R @ X.T).T + np.array([1, 2, 3])
    s, Re, te = _sim3(X, Y)
    assert abs(s - 3.0) < 1e-9 and np.allclose(Re, R) and np.allclose(te, [1, 2, 3])
