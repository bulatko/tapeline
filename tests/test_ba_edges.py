"""Relative-pose edges in GPU BA (run on CPU): a room connected to the rest by a single shared point returns to its
place after a rigid shift when prior edges are present; without edges its position is arbitrary. The error is measured
after sim3 alignment on the outer frames (BA is gauge-free)."""
import numpy as np
import torch

from tapeline.sfm_gpu import gpuba as GB
from tapeline.sfm_gpu.gpuba import so3_exp, edge_residuals, edge_jacobian
from tapeline.stages.chain import sim3


def _rot(w):
    return so3_exp(torch.as_tensor(np.asarray(w, float)[None], dtype=GB.DT))[0].numpy()


ROOM = range(4, 8)          # frames "behind the door"
OUT = [k for k in range(12) if k not in ROOM]


def _scene(seed=0, common_for_room=1):
    rng = np.random.default_rng(seed)
    F = 12
    cam = np.array([500.0, 500.0, 320.0, 240.0, 0, 0, 0, 0])
    R_true = np.stack([_rot(rng.normal(0, 0.02, 3)) for _ in range(F)])
    C = np.stack([[0.3 * k, 0.0, 0.0] for k in range(F)]) + rng.normal(0, 0.01, (F, 3))
    t_true = np.stack([-R_true[k] @ C[k] for k in range(F)])
    X_common = np.column_stack([rng.uniform(-1, 4.5, 80), rng.uniform(-1, 1, 80), rng.uniform(3, 6, 80)])
    X_room = np.column_stack([rng.uniform(0.8, 2.4, 60), rng.uniform(-1, 1, 60), rng.uniform(2, 4, 60)])
    X = np.vstack([X_common, X_room])
    pt, fr, xy = [], [], []
    for k in range(F):
        if k in ROOM:
            vis = list(range(80, 140)) + list(range(common_for_room))   # room points + one shared point
        else:
            vis = list(range(80))
        for p in vis:
            Xc = R_true[k] @ X[p] + t_true[k]
            if Xc[2] <= 0.1:
                continue
            u = cam[0] * Xc[0] / Xc[2] + cam[2]; v = cam[1] * Xc[1] / Xc[2] + cam[3]
            pt.append(p); fr.append(k); xy.append([u + rng.normal(0, 0.3), v + rng.normal(0, 0.3)])
    return F, cam, R_true, t_true, X, np.array(pt), np.array(fr), np.array(xy)


def _edges_from_truth(R_true, t_true, pairs, w_rot=100.0, w_t=100.0):
    Rp, tp = [], []
    for i, j in pairs:
        Rij = R_true[i] @ R_true[j].T
        Rp.append(Rij); tp.append(t_true[i] - Rij @ t_true[j])
    return dict(i=[p[0] for p in pairs], j=[p[1] for p in pairs], Rp=np.stack(Rp), tp=np.stack(tp),
                w_rot=[w_rot] * len(pairs), w_t=[w_t] * len(pairs))


def _room_error(res, R_true, t_true):
    C_est = np.stack([(-res["R"][k].T @ res["t"][k]).numpy() for k in range(len(R_true))])
    C_true = np.stack([-R_true[k].T @ t_true[k] for k in range(len(R_true))])
    s, R, t = sim3(C_est[OUT], C_true[OUT])
    C_al = s * (R @ C_est.T).T + t
    return np.linalg.norm(C_al[list(ROOM)] - C_true[list(ROOM)], axis=1).mean(), np.linalg.norm(C_al[OUT] - C_true[OUT], axis=1).mean()


def test_edge_residual_zero_at_truth_and_jacobian_matches_finite_step():
    F, cam, R_true, t_true, X, pt, fr, xy = _scene()
    pairs = [(k, k + 1) for k in range(F - 1)]
    ed = _edges_from_truth(R_true, t_true, pairs)
    prob = GB.Problem(R_true, t_true, X, cam, pt, fr, xy, device="cpu", edges=ed)
    r0 = edge_residuals(prob.R, prob.t, prob.edges)
    assert float(r0.abs().max()) < 1e-9
    J = edge_jacobian(prob.R, prob.t, prob.edges)
    # step on frame 5 only (it is i in edge (5,6) and j in edge (4,5)): dt = (1e-3, 0, 0), dw = (0, 1e-3, 0)
    t1 = prob.t.clone(); t1[5] += torch.tensor([1e-3, 0, 0], dtype=GB.DT)
    R1 = prob.R.clone(); R1[5] = R1[5] @ so3_exp(torch.tensor([[0, 1e-3, 0]], dtype=GB.DT))[0]
    r1 = edge_residuals(R1, t1, prob.edges)
    pred = torch.zeros_like(r0)
    for e in range(prob.edges["i"].numel()):
        if int(prob.edges["i"][e]) == 5:
            pred[e] += J[e, :, 1] * 1e-3 + J[e, :, 3] * 1e-3
        if int(prob.edges["j"][e]) == 5:
            pred[e] += J[e, :, 7] * 1e-3 + J[e, :, 9] * 1e-3
    assert float((r1 - r0 - pred).abs().max()) < 1e-6 * max(1.0, float(pred.abs().max()) * 1e3)


def test_edges_pull_shifted_room_back():
    F, cam, R_true, t_true, X, pt, fr, xy = _scene()
    R0, t0, X0 = R_true.copy(), t_true.copy(), X.copy()
    shift = np.array([0.5, 0.0, 0.0])
    for k in ROOM:
        C = -R0[k].T @ t0[k] + shift
        t0[k] = -R0[k] @ C
    X0[80:] += shift                                   # room points moved together with the room
    pairs = [(k, k + 1) for k in range(2, 9)]
    ed = _edges_from_truth(R_true, t_true, pairs)
    err_with, out_with = _room_error(GB.solve(GB.Problem(R0, t0, X0, cam, pt, fr, xy, device="cpu", edges=ed), max_iter=100), R_true, t_true)
    err_wo, out_wo = _room_error(GB.solve(GB.Problem(R0, t0, X0, cam, pt, fr, xy, device="cpu"), max_iter=100), R_true, t_true)
    assert out_with < 0.02 and out_wo < 0.02, (out_with, out_wo)        # outer frames stay in place in both cases
    assert err_with < 0.03, err_with                                    # edges brought the room back (the shift was 0.5)
    assert err_wo > 0.2, err_wo                                         # without edges a single shared point does not hold the room
