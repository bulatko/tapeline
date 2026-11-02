"""Bundle adjustment on the GPU: Levenberg-Marquardt with the Schur complement,
analytic Jacobians, Cauchy robust loss as in ceres.

What is reproduced from ceres/COLMAP (so that WHERE it is computed changes, not WHAT):
  * robust loss: Triggs correction. For Cauchy rho''<0, and in that case ceres
    (corrector.cc: rho[2] <= 0) uses sqrt(rho') scaling with alpha=0 -- same here;
  * ceres LM policy (trust_region_strategy LEVENBERG_MARQUARDT):
    damping diag(JtJ)/radius clamped to [1e-6, 1e32], radius0=1e4,
    step accepted if rho > 1e-3, radius /= max(1/3, 1-(2rho-1)^3) on acceptance,
    otherwise radius /= decrease_factor, decrease_factor *= 2;
  * stopping criteria: max_num_iterations, function_tolerance (relative
    cost decrease), gradient_tolerance, parameter_tolerance.

Parameterization: pose rig_from_world (R,t), increment R <- R exp([dw]x) (right
perturbation, as ceres QuaternionManifold), t <- t + dt; points additively;
OPENCV camera additively over the mask of refined parameters.
"""
import time
import numpy as np
import torch

DT = torch.float64


# ---------------- geometry ----------------
def so3_exp(w):
    """w (N,3) -> R (N,3,3)."""
    th = torch.linalg.norm(w, dim=1, keepdim=True)
    small = (th < 1e-8).squeeze(1)
    k = w / torch.clamp(th, min=1e-30)
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    s = torch.sin(th).unsqueeze(2); c = (1 - torch.cos(th)).unsqueeze(2)
    R = I + s * K + c * (K @ K)
    if small.any():   # first order is sufficient for |w|<1e-8
        Kw = torch.zeros_like(K[small])
        Kw[:, 0, 1] = -w[small][:, 2]; Kw[:, 0, 2] = w[small][:, 1]
        Kw[:, 1, 0] = w[small][:, 2];  Kw[:, 1, 2] = -w[small][:, 0]
        Kw[:, 2, 0] = -w[small][:, 1]; Kw[:, 2, 1] = w[small][:, 0]
        R[small] = torch.eye(3, dtype=w.dtype, device=w.device) + Kw
    return R


def _orthonormalize(R):
    U, _, Vh = torch.linalg.svd(R)
    Rn = U @ Vh
    det = torch.linalg.det(Rn)
    Vh = Vh.clone(); Vh[:, 2, :] *= det.unsqueeze(1)
    return U @ Vh


def so3_log(R):
    """R (N,3,3) -> w (N,3): axis·angle; stable for small angles."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    th = torch.acos(c)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    small = th < 1e-6
    k = th / torch.clamp(2.0 * torch.sin(th), min=1e-30)
    k = torch.where(small, torch.full_like(k, 0.5), k)
    return v * k.unsqueeze(1)


# ---------------- relative pose edges (VGGT prior) ----------------
def edge_residuals(R, t, ed):
    """Edge residuals (E,6): [w_rot·log(Rp^T R_i R_j^T), w_t·((t_i - R_ij t_j) - tp)] in the w2c convention.
    ed: dict(i, j, Rp (E,3,3), tp (E,3), w_rot (E,), w_t (E,))."""
    Ri, Rj = R[ed["i"]], R[ed["j"]]
    Rij = torch.bmm(Ri, Rj.transpose(1, 2))
    tij = t[ed["i"]] - torch.einsum('nij,nj->ni', Rij, t[ed["j"]])
    r_rot = so3_log(torch.bmm(ed["Rp"].transpose(1, 2), Rij)) * ed["w_rot"].unsqueeze(1)
    r_t = (tij - ed["tp"]) * ed["w_t"].unsqueeze(1)
    return torch.cat([r_rot, r_t], 1)


def edge_jacobian(R, t, ed, eps=1e-6):
    """Numerical Jacobian of edge residuals w.r.t. [dw_i, dt_i, dw_j, dt_j] (E,6,12) under right perturbation
    R <- R·exp([dw]x), t <- t + dt -- the same parameterization as the rest of the problem."""
    E = ed["i"].numel()
    Ri, ti, Rj, tj = R[ed["i"]], t[ed["i"]], R[ed["j"]], t[ed["j"]]
    sub = dict(ed); sub["i"] = torch.arange(E, device=R.device); sub["j"] = torch.arange(E, 2 * E, device=R.device)

    def res(dw_i, dt_i, dw_j, dt_j):
        Rs = torch.cat([torch.bmm(Ri, so3_exp(dw_i)), torch.bmm(Rj, so3_exp(dw_j))], 0)
        ts = torch.cat([ti + dt_i, tj + dt_j], 0)
        return edge_residuals(Rs, ts, sub)

    J = torch.zeros(E, 6, 12, dtype=R.dtype, device=R.device)
    zero = torch.zeros(E, 3, dtype=R.dtype, device=R.device)
    for p in range(12):
        d = torch.zeros(E, 12, dtype=R.dtype, device=R.device); d[:, p] = eps
        args_p = (d[:, 0:3], d[:, 3:6], d[:, 6:9], d[:, 9:12])
        args_m = tuple(-a for a in args_p)
        J[:, :, p] = (res(*args_p) - res(*args_m)) / (2 * eps)
    return J


# ---------------- OPENCV projection model + Jacobians ----------------
def project_jac(Xc, cam, need_jac=True):
    """Xc (N,3), cam (8,).
    Returns uv (N,2), dXc (N,2,3), dcam (N,2,8)."""
    fx, fy, cx, cy, k1, k2, p1, p2 = [cam[i] for i in range(8)]
    z = Xc[:, 2]
    iz = 1.0 / z
    u = Xc[:, 0] * iz
    v = Xc[:, 1] * iz
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    rad = k1 * r2 + k2 * r2 * r2
    xd = u * (1 + rad) + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2)
    yd = v * (1 + rad) + p1 * (r2 + 2.0 * v2) + 2.0 * p2 * uv
    uvp = torch.stack([fx * xd + cx, fy * yd + cy], 1)
    if not need_jac:
        return uvp, None, None
    g = k1 + 2.0 * k2 * r2               # d(rad)/d(r2)
    dxd_du = 1.0 + rad + 2.0 * u2 * g + 2.0 * p1 * v + 6.0 * p2 * u
    dxd_dv = 2.0 * uv * g + 2.0 * p1 * u + 2.0 * p2 * v
    dyd_du = dxd_dv
    dyd_dv = 1.0 + rad + 2.0 * v2 * g + 6.0 * p1 * v + 2.0 * p2 * u
    N = Xc.shape[0]
    dXc = torch.empty(N, 2, 3, dtype=Xc.dtype, device=Xc.device)
    # du/dXc = (iz, 0, -u*iz);  dv/dXc = (0, iz, -v*iz)
    dXc[:, 0, 0] = fx * dxd_du * iz
    dXc[:, 0, 1] = fx * dxd_dv * iz
    dXc[:, 0, 2] = -fx * (dxd_du * u + dxd_dv * v) * iz
    dXc[:, 1, 0] = fy * dyd_du * iz
    dXc[:, 1, 1] = fy * dyd_dv * iz
    dXc[:, 1, 2] = -fy * (dyd_du * u + dyd_dv * v) * iz
    dcam = torch.zeros(N, 2, 8, dtype=Xc.dtype, device=Xc.device)
    dcam[:, 0, 0] = xd
    dcam[:, 1, 1] = yd
    dcam[:, 0, 2] = 1.0
    dcam[:, 1, 3] = 1.0
    dcam[:, 0, 4] = fx * u * r2
    dcam[:, 1, 4] = fy * v * r2
    dcam[:, 0, 5] = fx * u * r2 * r2
    dcam[:, 1, 5] = fy * v * r2 * r2
    dcam[:, 0, 6] = fx * 2.0 * uv
    dcam[:, 1, 6] = fy * (r2 + 2.0 * v2)
    dcam[:, 0, 7] = fx * (r2 + 2.0 * u2)
    dcam[:, 1, 7] = fy * 2.0 * uv
    return uvp, dXc, dcam


# ---------------- robust loss (as in ceres) ----------------
def cauchy_rho(s, scale):
    """s = ||r||^2. Returns (rho(s), rho'(s))."""
    if scale is None:
        return s, torch.ones_like(s)
    a = scale * scale
    inv = 1.0 / (1.0 + s / a)
    return a * torch.log1p(s / a), inv


class Problem:
    """BA problem on the GPU. All tensors are float64 on device."""

    def __init__(self, R, t, X, cam, pt, fr, xy, device="cuda", build_pairs=True, edges=None):
        d = device
        self.dev = d
        self.R = torch.as_tensor(np.ascontiguousarray(R), dtype=DT, device=d).clone()
        self.t = torch.as_tensor(np.ascontiguousarray(t), dtype=DT, device=d).clone()
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=DT, device=d).clone()
        self.cam = torch.as_tensor(np.ascontiguousarray(cam), dtype=DT, device=d).clone()
        # relative pose edges (VGGT prior) for weakly connected frames: dict(i, j, Rp, tp, w_rot, w_t)
        self.edges = None
        if edges is not None and len(edges["i"]):
            self.edges = dict(i=torch.as_tensor(np.asarray(edges["i"], np.int64), device=d),
                              j=torch.as_tensor(np.asarray(edges["j"], np.int64), device=d),
                              Rp=torch.as_tensor(np.ascontiguousarray(edges["Rp"]), dtype=DT, device=d),
                              tp=torch.as_tensor(np.ascontiguousarray(edges["tp"]), dtype=DT, device=d),
                              w_rot=torch.as_tensor(np.asarray(edges["w_rot"], float), dtype=DT, device=d),
                              w_t=torch.as_tensor(np.asarray(edges["w_t"], float), dtype=DT, device=d))
        # observations are sorted by point -- cheaper to build pairs for the Schur complement
        order = np.argsort(pt, kind="stable")
        self.pt = torch.as_tensor(pt[order], device=d)
        self.fr = torch.as_tensor(fr[order], device=d)
        self.xy = torch.as_tensor(np.ascontiguousarray(xy[order]), dtype=DT, device=d)
        self.F = int(self.R.shape[0]); self.P = int(self.X.shape[0]); self.N = int(self.pt.shape[0])
        if build_pairs:      # pairs are needed only by full BA (Schur complement)
            self._build_pairs()

    def _build_pairs(self):
        """Indices of all ordered observation pairs within a single point."""
        cnt = torch.bincount(self.pt, minlength=self.P)
        off = torch.zeros(self.P + 1, dtype=torch.long, device=self.dev)
        off[1:] = torch.cumsum(cnt, 0)
        self.cnt, self.off = cnt, off
        npair = (cnt * cnt)
        self.n_pairs = int(npair.sum().item())
        tot = torch.cumsum(npair, 0)
        pstart = torch.zeros(self.P + 1, dtype=torch.long, device=self.dev)
        pstart[1:] = tot
        idx = torch.arange(self.n_pairs, device=self.dev)
        p = torch.bucketize(idx, pstart[1:], right=True)     # point of each pair
        loc = idx - pstart[p]
        c = cnt[p]
        a = loc // c
        b = loc % c
        pair_i = off[p] + a
        pair_j = off[p] + b
        # block id (f_i, f_j) in the (F,F,6,6) accumulator; pairs are sorted by block,
        # so the per-block sum is a segmented reduction (CSR-SpMM), without atomics
        blk = self.fr[pair_i] * self.F + self.fr[pair_j]
        order = torch.argsort(blk)
        self.pair_i = pair_i[order]; self.pair_j = pair_j[order]; self.pair_p = p[order]
        blk = blk[order]
        self.blk_id, counts = torch.unique_consecutive(blk, return_counts=True)
        crow = torch.zeros(self.blk_id.numel() + 1, dtype=torch.int64, device=self.dev)
        crow[1:] = torch.cumsum(counts, 0)
        self.seg_csr = torch.sparse_csr_tensor(
            crow, torch.arange(self.n_pairs, dtype=torch.int64, device=self.dev),
            torch.ones(self.n_pairs, dtype=DT, device=self.dev),
            size=(self.blk_id.numel(), self.n_pairs))

    # ---------- cost / Jacobians ----------
    def residuals(self, R=None, t=None, X=None, cam=None, need_jac=False):
        R = self.R if R is None else R
        t = self.t if t is None else t
        X = self.X if X is None else X
        cam = self.cam if cam is None else cam
        Xw = X[self.pt]
        Rk = R[self.fr]
        Xc = torch.einsum('nij,nj->ni', Rk, Xw) + t[self.fr]
        uv, dXc, dcam = project_jac(Xc, cam, need_jac)
        r = uv - self.xy
        return r, Xc, Rk, Xw, dXc, dcam

    def cost(self, R=None, t=None, X=None, cam=None, loss_scale=None):
        r, Xc, *_ = self.residuals(R, t, X, cam, need_jac=False)
        s = (r * r).sum(1)
        bad = ~torch.isfinite(s)
        s = torch.where(bad, torch.full_like(s, 1e12), s)
        rho, _ = cauchy_rho(s, loss_scale)
        c = 0.5 * rho.sum()
        if self.edges is not None:
            re = edge_residuals(self.R if R is None else R, self.t if t is None else t, self.edges)
            c = c + 0.5 * (re * re).sum()
        return c


def _gauge_scale(R, t):
    """Model scale as the median distance of camera centers to their MEDIAN center. C = -R^T t for w2c poses.

    The median, not the mean: a single poorly constrained frame that drifts far away drags the mean
    with it, and normalizing by such a "scale" would shrink the whole model every time the outlier
    moves, eventually collapsing it."""
    C = -torch.einsum('nji,nj->ni', R, t)
    d = torch.linalg.norm(C - C.median(0).values.unsqueeze(0), dim=1)
    return float(d.median())


def solve(prob, loss_scale=None, refine_focal=False, refine_pp=False, refine_extra=False,
          refine_poses=True, refine_points=True, max_iter=100, function_tolerance=1e-7,
          gradient_tolerance=1e-10, parameter_tolerance=1e-8, verbose=False,
          pair_chunk=4_000_000, fix_gauge=True):
    """LM + Schur complement. Returns the iteration history.

    fix_gauge: after each accepted step the model scale is reset to its initial value (t and X are multiplied
    by a common factor -- reprojection and cost do not change). Scale is a gauge freedom of BA,
    and without fixing it drifts and can even collapse to zero while the reprojection error stays low."""
    dev = prob.dev
    F, P, N = prob.F, prob.P, prob.N
    cmask = []
    if refine_focal: cmask += [0, 1]
    if refine_pp:    cmask += [2, 3]
    if refine_extra: cmask += [4, 5, 6, 7]
    cmask = torch.as_tensor(sorted(cmask), dtype=torch.long, device=dev)
    nc = int(cmask.numel())
    npose = 6 if refine_poses else 0
    m = F * npose + nc                       # size of the reduced system
    assert m > 0

    R, t, X, cam = prob.R.clone(), prob.t.clone(), prob.X.clone(), prob.cam.clone()
    radius = 1e4
    decrease_factor = 2.0
    hist = []
    cost = float(prob.cost(R, t, X, cam, loss_scale))
    gauge_ref = _gauge_scale(R, t) if (fix_gauge and refine_poses and F >= 3) else None
    if gauge_ref is not None and not (gauge_ref > 0):
        gauge_ref = None                                   # all centers coincide -- nothing to fix
    n_gauge, gauge_klog = 0, 0.0
    t_jac = t_schur = t_solve = t_eval = 0.0

    for it in range(max_iter):
        # ---------- assemble Jacobians and normal-equation blocks ----------
        torch.cuda.synchronize() if dev != "cpu" else None
        ta = time.time()
        r, Xc, Rk, Xw, dXc, dcam = prob.residuals(R, t, X, cam, need_jac=True)
        s = (r * r).sum(1)
        rho, rho1 = cauchy_rho(s, loss_scale)
        w = torch.sqrt(torch.clamp(rho1, min=0.0)).unsqueeze(1)      # ceres: rho''<0 -> alpha=0
        rw = r * w
        # J w.r.t. point: dXc @ R   (N,2,3)
        Jp = torch.bmm(dXc, Rk)
        # J w.r.t. pose: dXc @ [-R [Xw]x , I]
        if refine_poses:
            RX = torch.einsum('nij,nj->ni', Rk, Xw)                  # = R Xw
            # -R[Xw]x = [RXw]x  (identity: R[a]x R^T = [Ra]x, and -R[a]x = [R a]x R^T·... )
            # precisely: d(R exp(dw) Xw)/d(dw) = -R [Xw]x
            Xx = torch.zeros(N, 3, 3, dtype=DT, device=dev)
            Xx[:, 0, 1] = -Xw[:, 2]; Xx[:, 0, 2] = Xw[:, 1]
            Xx[:, 1, 0] = Xw[:, 2];  Xx[:, 1, 2] = -Xw[:, 0]
            Xx[:, 2, 0] = -Xw[:, 1]; Xx[:, 2, 1] = Xw[:, 0]
            dXc_dw = -torch.bmm(Rk, Xx)                              # (N,3,3)
            Jf = torch.cat([torch.bmm(dXc, dXc_dw), dXc], 2)         # (N,2,6)
            Jf = Jf * w.unsqueeze(2)
        Jp = Jp * w.unsqueeze(2)
        if nc:
            Jc = dcam[:, :, cmask] * w.unsqueeze(2)                  # (N,2,nc)
        del dXc, dcam
        torch.cuda.synchronize() if dev != "cpu" else None
        t_jac += time.time() - ta

        ta = time.time()
        # blocks
        V = torch.einsum('nai,naj->nij', Jp, Jp)                     # (N,3,3) -> summed per point
        Vp = torch.zeros(P, 3, 3, dtype=DT, device=dev).index_add_(0, prob.pt, V)
        gp = torch.zeros(P, 3, dtype=DT, device=dev).index_add_(
            0, prob.pt, torch.einsum('nai,na->ni', Jp, rw))
        # block accumulators: Aff (F,F,6,6), Afc (F,6,nc), Acc (nc,nc).
        # Assembled without building index tensors -- index_add_ along the first axis.
        g = torch.zeros(m, dtype=DT, device=dev)
        Aff = torch.zeros(F, F, 6, 6, dtype=DT, device=dev) if refine_poses else None
        Afc = torch.zeros(F, 6, nc, dtype=DT, device=dev) if (refine_poses and nc) else None
        Acc = torch.zeros(nc, nc, dtype=DT, device=dev) if nc else None
        if refine_poses:
            Uf = torch.zeros(F, 6, 6, dtype=DT, device=dev).index_add_(
                0, prob.fr, torch.einsum('nai,naj->nij', Jf, Jf))
            gf = torch.zeros(F, 6, dtype=DT, device=dev).index_add_(
                0, prob.fr, torch.einsum('nai,na->ni', Jf, rw))
            if prob.edges is not None:
                # prior edges: J_e (E,6,12) w.r.t. [pose i | pose j]; diagonal blocks go into Uf (and the damping),
                # off-diagonal (i,j),(j,i) directly into Aff; gradient into gf. Points are not involved, Schur is unchanged.
                ed = prob.edges
                re = edge_residuals(R, t, ed)
                Je = edge_jacobian(R, t, ed)
                Ji, Jj = Je[:, :, :6], Je[:, :, 6:]
                Uf.index_add_(0, ed["i"], torch.einsum('eai,eaj->eij', Ji, Ji))
                Uf.index_add_(0, ed["j"], torch.einsum('eai,eaj->eij', Jj, Jj))
                Aff.view(F * F, 6, 6).index_add_(0, ed["i"] * F + ed["j"], torch.einsum('eai,eaj->eij', Ji, Jj))
                Aff.view(F * F, 6, 6).index_add_(0, ed["j"] * F + ed["i"], torch.einsum('eai,eaj->eij', Jj, Ji))
                gf.index_add_(0, ed["i"], torch.einsum('eai,ea->ei', Ji, re))
                gf.index_add_(0, ed["j"], torch.einsum('eai,ea->ei', Jj, re))
            Aff.diagonal(dim1=0, dim2=1).permute(2, 0, 1).add_(Uf)
            g[:F * 6] = gf.reshape(-1)
        if nc:
            Acc += torch.einsum('nai,naj->ij', Jc, Jc)
            g[F * npose:] = torch.einsum('nai,na->i', Jc, rw)
            if refine_poses:
                Afc += torch.zeros(F, 6, nc, dtype=DT, device=dev).index_add_(
                    0, prob.fr, torch.einsum('nai,naj->nij', Jf, Jc))
        # damping: diag(JtJ)/radius, clamps as in ceres
        dS = torch.zeros(m, dtype=DT, device=dev)
        if refine_poses:
            dS[:F * 6] = torch.diagonal(Uf, dim1=1, dim2=2).reshape(-1)
        if nc:
            dS[F * npose:] = torch.diagonal(Acc)
        dS = torch.clamp(dS, 1e-6, 1e32)
        dV = torch.clamp(torch.diagonal(Vp, dim1=1, dim2=2).clone(), 1e-6, 1e32)
        torch.cuda.synchronize() if dev != "cpu" else None
        t_schur += time.time() - ta

        gmax = float(torch.max(torch.abs(g)).item()) if m else 0.0
        gmax = max(gmax, float(torch.max(torch.abs(gp)).item()))
        if gmax < gradient_tolerance:
            hist.append(dict(it=it, cost=cost, note="gradient_tolerance"))
            break

        # Y blocks (W_p^T in block form) do not depend on the radius -- computed once per iteration
        Yf = torch.bmm(Jf.transpose(1, 2), Jp) if refine_poses else None   # (N,6,3)
        Yc = torch.bmm(Jc.transpose(1, 2), Jp) if nc else None             # (N,nc,3)

        accepted = False
        for _inner in range(20):
            ta = time.time()
            Vd = Vp.clone()
            Vd.diagonal(dim1=1, dim2=2).add_(dV / radius)
            Vinv = torch.linalg.inv(Vd)
            # ---------- Schur complement ----------
            Bff = Aff.clone() if refine_poses else None
            Bfc = Afc.clone() if (refine_poses and nc) else None
            Bcc = Acc.clone() if nc else None
            gs = g.clone()
            # g_S = g - sum_p W_p Vinv g_p
            Vg = torch.einsum('pij,pj->pi', Vinv, gp)                          # (P,3)
            if refine_poses:
                contrib = torch.einsum('nij,nj->ni', Yf, Vg[prob.pt])
                gs[:F * 6] -= torch.zeros(F, 6, dtype=DT, device=dev).index_add_(
                    0, prob.fr, contrib).reshape(-1)
            if nc:
                gs[F * npose:] -= torch.einsum('nij,nj->i', Yc, Vg[prob.pt])
            # S -= sum_p W_p Vinv W_p^T  (over observation pairs within a point).
            # Accumulated into block tensors via index_add_ along the first axis: index
            # tensors are not built, memory O(F^2*36) instead of O(pairs*36).
            i = prob.pair_i; j = prob.pair_j
            Vi = Vinv[prob.pair_p]
            if refine_poses:
                A = torch.bmm(Yf[i], Vi)                                        # (K,6,3)
                S66 = torch.bmm(A, Yf[j].transpose(1, 2)).reshape(-1, 36)       # (K,36)
                seg = torch.sparse.mm(prob.seg_csr, S66)                        # (nblk,36)
                Bff.view(F * F, 6, 6).index_add_(
                    0, prob.blk_id, seg.view(-1, 6, 6), alpha=-1.0)
                del S66, seg
                if nc:
                    B6c = torch.bmm(A, Yc[j].transpose(1, 2)).reshape(-1, 6 * nc)
                    segc = torch.sparse.mm(prob.seg_csr, B6c)
                    Bfc.index_add_(0, torch.div(prob.blk_id, F, rounding_mode='floor'),
                                   segc.view(-1, 6, nc), alpha=-1.0)
                    del B6c, segc
                del A
            if nc:
                Ac = torch.bmm(Yc[i], Vi)
                Bcc -= torch.bmm(Ac, Yc[j].transpose(1, 2)).sum(0)              # (nc,nc)
                del Ac
            del Vi
            # assemble the dense m x m matrix
            Sd_ = torch.zeros(m, m, dtype=DT, device=dev)
            if refine_poses:
                Sd_[:F * 6, :F * 6] = Bff.permute(0, 2, 1, 3).reshape(F * 6, F * 6)
            if nc:
                Sd_[F * npose:, F * npose:] = Bcc
                if refine_poses:
                    Sd_[:F * 6, F * 6:] = Bfc.reshape(F * 6, nc)
                    Sd_[F * 6:, :F * 6] = Bfc.reshape(F * 6, nc).T
            Sd_.diagonal().add_(dS / radius)
            torch.cuda.synchronize() if dev != "cpu" else None
            t_schur += time.time() - ta

            ta = time.time()
            try:
                L = torch.linalg.cholesky(Sd_)
                dxc = torch.cholesky_solve(-gs.unsqueeze(1), L).squeeze(1)
            except Exception:
                dxc = torch.linalg.lstsq(Sd_, -gs.unsqueeze(1)).solution.squeeze(1)
            if not torch.isfinite(dxc).all():
                radius /= decrease_factor; decrease_factor *= 2.0
                continue
            # back-substitution for points
            rhs = -gp.clone()
            if refine_poses:
                dxf = dxc[:F * 6].view(F, 6)
                rhs.index_add_(0, prob.pt, -torch.einsum('nij,ni->nj', Yf, dxf[prob.fr]))
            if nc:
                dxcam = dxc[F * npose:]
                rhs.index_add_(0, prob.pt, -torch.einsum('nij,i->nj', Yc, dxcam))
            dX = torch.einsum('pij,pj->pi', Vinv, rhs)
            torch.cuda.synchronize() if dev != "cpu" else None
            t_solve += time.time() - ta

            # ---------- predicted cost decrease ----------
            ta = time.time()
            # ceres (trust_region_minimizer): model_cost_change = 0.5*dx^T(D*dx - g)
            dxp = dX
            pred = 0.5 * (float((dxc * (dS / radius * dxc - g)).sum())
                          + float((dxp * ((dV / radius) * dxp - gp)).sum()))
            Rn = R.clone(); tn = t.clone(); Xn = X.clone(); camn = cam.clone()
            if refine_poses:
                dw = dxf[:, :3]; dt = dxf[:, 3:]
                Rn = torch.bmm(R, so3_exp(dw))
                tn = t + dt
            if refine_points:
                Xn = X + dX
            if nc:
                camn = cam.clone(); camn[cmask] = cam[cmask] + dxcam
            new_cost = float(prob.cost(Rn, tn, Xn, camn, loss_scale))
            torch.cuda.synchronize() if dev != "cpu" else None
            t_eval += time.time() - ta

            actual = cost - new_cost
            rel = actual / pred if pred > 0 else -1.0
            if rel > 1e-3 and np.isfinite(new_cost):
                R, t, X, cam = Rn, tn, Xn, camn
                if gauge_ref is not None:
                    k = gauge_ref / _gauge_scale(R, t)      # common factor on the world: reprojection unchanged
                    if np.isfinite(k) and k > 0 and abs(k - 1.0) > 1e-12:
                        t = t * k
                        X = X * k
                        n_gauge += 1
                        gauge_klog += abs(float(np.log(k)))
                dcost = cost - new_cost
                cost = new_cost
                radius = radius / max(1.0 / 3.0, 1.0 - (2.0 * rel - 1.0) ** 3)
                radius = min(radius, 1e16)
                decrease_factor = 2.0
                accepted = True
                hist.append(dict(it=it, cost=cost, rel=rel, radius=radius,
                                 step=float(torch.linalg.norm(dxc).item())))
                if verbose:
                    print("  it %3d cost=%.6e rel=%.3f radius=%.3e" % (it, cost, rel, radius), flush=True)
                break
            radius /= decrease_factor
            decrease_factor *= 2.0
            if radius < 1e-32:
                break
        if not accepted:
            hist.append(dict(it=it, cost=cost, note="no step accepted"))
            break
        if dcost < function_tolerance * max(cost, 1e-30):
            hist.append(dict(it=it, cost=cost, note="function_tolerance"))
            break

    R = _orthonormalize(R)
    times = dict(jac=t_jac, schur=t_schur, solve=t_solve, eval=t_eval)
    gauge = dict(ref=gauge_ref, final=_gauge_scale(R, t) if refine_poses else None,
                 n_fix=n_gauge, sum_abs_log_k=gauge_klog)
    return dict(R=R, t=t, X=X, cam=cam, hist=hist, times=times, cost=cost,
                iters=len(hist), n_pairs=prob.n_pairs, gauge=gauge)


def solve_points(prob, loss_scale=None, max_iter=50, function_tolerance=1e-6,
                 verbose=False):
    """Points-only BA with fixed poses and intrinsics -- exactly what COLMAP runs
    inside triangulate_points (IterativeGlobalRefinement -> AdjustGlobalBundle
    with refine_rig_from_world=False, refine_intrinsics=False, trivial loss).

    The problem is fully separable: each point is an independent 3-parameter problem.
    Hence LM here uses a PER-POINT trust radius: the optimum is the same as with
    the global ceres trust region (for a separable problem the minimum is reached
    independently per block), but the steps do not block each other.
    """
    dev = prob.dev
    P = prob.P
    R, t, cam = prob.R, prob.t, prob.cam
    X = prob.X.clone()
    radius = torch.full((P,), 1e4, dtype=DT, device=dev)
    decrease = torch.full((P,), 2.0, dtype=DT, device=dev)

    def point_costs(Xv):
        Xc = torch.einsum('nij,nj->ni', R[prob.fr], Xv[prob.pt]) + t[prob.fr]
        uv, _, _ = project_jac(Xc, cam, need_jac=False)
        r = uv - prob.xy
        s = (r * r).sum(1)
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, 1e12))
        rho, _ = cauchy_rho(s, loss_scale)
        return torch.zeros(P, dtype=DT, device=dev).index_add_(0, prob.pt, 0.5 * rho)

    cost = point_costs(X)
    hist = []
    for it in range(max_iter):
        Xw = X[prob.pt]
        Rk = R[prob.fr]
        Xc = torch.einsum('nij,nj->ni', Rk, Xw) + t[prob.fr]
        uv, dXc, _ = project_jac(Xc, cam)
        r = uv - prob.xy
        s = (r * r).sum(1)
        rho, rho1 = cauchy_rho(s, loss_scale)
        w = torch.sqrt(torch.clamp(rho1, min=0.0)).unsqueeze(1)
        Jp = torch.bmm(dXc, Rk) * w.unsqueeze(2)
        rw = r * w
        V = torch.zeros(P, 3, 3, dtype=DT, device=dev).index_add_(
            0, prob.pt, torch.einsum('nai,naj->nij', Jp, Jp))
        g = torch.zeros(P, 3, dtype=DT, device=dev).index_add_(
            0, prob.pt, torch.einsum('nai,na->ni', Jp, rw))
        dV = torch.clamp(torch.diagonal(V, dim1=1, dim2=2), 1e-6, 1e32)
        gmax = float(g.abs().max().item())
        if gmax < 1e-10:
            hist.append(dict(it=it, note="gradient_tolerance")); break
        Vd = V.clone()
        Vd.diagonal(dim1=1, dim2=2).add_(dV / radius.unsqueeze(1))
        dX = torch.linalg.solve(Vd, -g.unsqueeze(2)).squeeze(2)
        bad = ~torch.isfinite(dX).all(1)
        dX = torch.where(bad.unsqueeze(1), torch.zeros_like(dX), dX)
        pred = 0.5 * ((dX * ((dV / radius.unsqueeze(1)) * dX - g)).sum(1))
        Xn = X + dX
        new_cost = point_costs(Xn)
        actual = cost - new_cost
        rel = torch.where(pred > 0, actual / torch.clamp(pred, min=1e-300),
                          torch.full_like(pred, -1.0))
        acc = (rel > 1e-3) & torch.isfinite(new_cost)
        X = torch.where(acc.unsqueeze(1), Xn, X)
        dcost = torch.where(acc, actual, torch.zeros_like(actual))
        cost = torch.where(acc, new_cost, cost)
        fac = torch.clamp(1.0 - (2.0 * rel - 1.0) ** 3, min=1.0 / 3.0)
        radius = torch.where(acc, torch.clamp(radius / fac, max=1e16), radius / decrease)
        decrease = torch.where(acc, torch.full_like(decrease, 2.0), decrease * 2.0)
        tot_cost = float(cost.sum().item()); tot_d = float(dcost.sum().item())
        hist.append(dict(it=it, cost=tot_cost, accepted=int(acc.sum().item())))
        if verbose:
            print("   pts it %2d cost=%.6e acc=%d" % (it, tot_cost, int(acc.sum().item())), flush=True)
        if tot_d < function_tolerance * max(tot_cost, 1e-30):
            hist.append(dict(it=it, note="function_tolerance")); break
    return dict(X=X, hist=hist, iters=len(hist), cost=float(cost.sum().item()))
