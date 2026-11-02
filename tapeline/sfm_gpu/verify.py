"""Geometric verification of image pairs on the GPU (replacement for pycolmap.verify_matches).

The task is perfectly batchable: pairs are independent, and within a pair RANSAC is hundreds
of independent hypotheses over hundreds of points. On GPU the computation is batched
over pairs AND hypotheses simultaneously.

Model: 8-point fundamental matrix (linear solver = batched SVD), Sampson error,
local refinement on inliers. The threshold and minimum number of inliers follow
COLMAP defaults (max_error 4 px, min_num_inliers 15).
"""
import numpy as np
import torch


def _normalize(x, mask):
    """Hartley normalization: centered at the mean, mean distance sqrt(2). (B,N,2)->(B,N,3),(B,3,3)"""
    w = mask.float()
    n = w.sum(1, keepdim=True).clamp(min=1)
    mu = (x * w[..., None]).sum(1, keepdim=True) / n[..., None]
    d = ((x - mu).norm(dim=-1) * w).sum(1, keepdim=True) / n
    s = (np.sqrt(2.0) / d.clamp(min=1e-9))[..., None]
    xn = (x - mu) * s
    B = x.shape[0]
    T = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)
    T[:, 0, 0] = s[:, 0, 0]; T[:, 1, 1] = s[:, 0, 0]; T[:, 2, 2] = 1
    T[:, 0, 2] = -s[:, 0, 0] * mu[:, 0, 0]; T[:, 1, 2] = -s[:, 0, 0] * mu[:, 0, 1]
    return torch.cat([xn, torch.ones_like(xn[..., :1])], -1), T


def _fundamental_8pt(p1, p2):
    """p1,p2: (M,S,3) homogeneous. -> F (M,3,3), rank-2."""
    A = torch.stack([p2[..., 0] * p1[..., 0], p2[..., 0] * p1[..., 1], p2[..., 0],
                     p2[..., 1] * p1[..., 0], p2[..., 1] * p1[..., 1], p2[..., 1],
                     p1[..., 0], p1[..., 1], torch.ones_like(p1[..., 0])], -1)
    _, _, Vh = torch.linalg.svd(A)
    F = Vh[:, -1].reshape(-1, 3, 3)
    U, S, Vh2 = torch.linalg.svd(F)                 # zero the smallest singular value: rank(F)=2
    S = torch.cat([S[:, :2], torch.zeros_like(S[:, :1])], 1)
    return U @ torch.diag_embed(S) @ Vh2


def _sampson(F, p1, p2):
    """F:(B,H,3,3) p:(B,N,3) -> error (B,H,N) in normalized coordinates."""
    Fx1 = torch.einsum('bhij,bnj->bhni', F, p1)
    Ftx2 = torch.einsum('bhji,bnj->bhni', F, p2)
    num = torch.einsum('bhni,bni->bhn', Fx1, p2) ** 2
    den = Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2
    return num / den.clamp(min=1e-12)


def verify_batch(x1, x2, mask, max_error=4.0, hypotheses=512, min_inliers=15,
                 lo_iters=2, seed=0):
    """x1,x2: (B,N,2) in pixels, mask: (B,N) validity. -> inlier_mask (B,N), F (B,3,3)."""
    dev = x1.device
    B, N, _ = x1.shape
    g = torch.Generator(device=dev); g.manual_seed(seed)
    p1, T1 = _normalize(x1, mask)
    p2, T2 = _normalize(x2, mask)
    # Solve in normalized coordinates (SVD is more stable), but count inliers
    # in PIXELS using the denormalized F: rescaling the threshold by the product of scales
    # underestimates it by orders of magnitude and rejects most valid pairs.
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], -1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], -1)
    thr = max_error ** 2

    cnt = mask.sum(1).clamp(min=1)
    idx = (torch.rand(B, hypotheses, 8, device=dev, generator=g)
           * cnt.view(B, 1, 1).float()).long().clamp(max=N - 1)
    valid_idx = torch.where(mask, torch.arange(N, device=dev).expand(B, N), torch.zeros(1, dtype=torch.long, device=dev))
    valid_sorted = torch.sort(valid_idx, dim=1, descending=True).values      # valid ones first
    take = torch.gather(valid_sorted, 1, idx.reshape(B, -1)).reshape(B, hypotheses, 8)

    s1 = torch.gather(p1, 1, take.reshape(B, -1, 1).expand(-1, -1, 3)).reshape(B, hypotheses, 8, 3)
    s2 = torch.gather(p2, 1, take.reshape(B, -1, 1).expand(-1, -1, 3)).reshape(B, hypotheses, 8, 3)
    F = _fundamental_8pt(s1.reshape(-1, 8, 3), s2.reshape(-1, 8, 3)).reshape(B, hypotheses, 3, 3)
    Fpx = T2.transpose(1, 2)[:, None] @ F @ T1[:, None]

    err = _sampson(Fpx, h1, h2)
    inl = (err < thr) & mask[:, None, :]
    best = inl.sum(-1).argmax(1)
    Fb = Fpx[torch.arange(B, device=dev), best]
    inl_b = inl[torch.arange(B, device=dev), best]

    for _ in range(lo_iters):                       # refine on all inliers of the best hypothesis
        w = inl_b.float()
        A = torch.stack([p2[..., 0] * p1[..., 0], p2[..., 0] * p1[..., 1], p2[..., 0],
                         p2[..., 1] * p1[..., 0], p2[..., 1] * p1[..., 1], p2[..., 1],
                         p1[..., 0], p1[..., 1], torch.ones_like(p1[..., 0])], -1)
        A = A * w[..., None]
        _, _, Vh = torch.linalg.svd(A.transpose(1, 2) @ A)
        Fn = Vh[:, -1].reshape(B, 3, 3)
        U, S, Vh2 = torch.linalg.svd(Fn)
        S = torch.cat([S[:, :2], torch.zeros_like(S[:, :1])], 1)
        Fn = U @ torch.diag_embed(S) @ Vh2
        Fn_px = T2.transpose(1, 2) @ Fn @ T1
        e = _sampson(Fn_px[:, None], h1, h2)[:, 0]
        new = (e < thr) & mask
        better = new.sum(1) >= inl_b.sum(1)
        inl_b = torch.where(better[:, None], new, inl_b)
        Fb = torch.where(better[:, None, None], Fn_px, Fb)

    ok = inl_b.sum(1) >= min_inliers
    return inl_b & ok[:, None], Fb, ok


def write_two_view_geometries(db_path, pairs, inliers, F, config=3):
    """Write verification results into a COLMAP database.

    Table format: pair_id, rows, cols, data (uint32 index pairs), config, F/E/H (3x3 float64).
    For an uncalibrated camera COLMAP writes F and H and leaves E/qvec/tvec empty -- we do the same.
    config=3 (UNCALIBRATED) is what COLMAP chooses for the vast majority of pairs; pairs without
    parallax get 6 (PLANAR_OR_PANORAMIC). We do not single them out --
    points without a triangulation angle are filtered out by min_angle during triangulation anyway.
    """
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM two_view_geometries")
    rows = []
    for (i1, i2), inl, f in zip(pairs, inliers, F):
        if len(inl) < 15:
            continue
        pid = i1 * 2147483647 + i2 if i1 < i2 else i2 * 2147483647 + i1
        m = np.ascontiguousarray(inl, np.uint32)
        rows.append((int(pid), int(m.shape[0]), 2, m.tobytes(), int(config),
                     np.ascontiguousarray(f, np.float64).tobytes()))
    con.executemany("INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, F) "
                    "VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    return len(rows)


def verify_database(db_path, batch=256, max_error=4.0, hypotheses=512, min_inliers=15, device="cuda"):
    """Full replacement for pycolmap.verify_matches: reads matches, writes two_view_geometries."""
    import sqlite3
    import time
    t0 = time.time()
    con = sqlite3.connect(db_path)
    kp = {iid: np.frombuffer(d, np.float32).reshape(r, c)[:, :2].copy()
          for iid, r, c, d in con.execute("SELECT image_id, rows, cols, data FROM keypoints")}
    raw = {pid: np.frombuffer(d, np.uint32).reshape(r, c).copy()
           for pid, r, c, d in con.execute("SELECT pair_id, rows, cols, data FROM matches WHERE rows>0")}
    con.close()

    pids = sorted(raw)
    out_pairs, out_inl, out_F = [], [], []
    for s in range(0, len(pids), batch):
        chunk = pids[s:s + batch]
        n = max(len(raw[p]) for p in chunk)
        x1 = torch.zeros(len(chunk), n, 2, device=device)
        x2 = torch.zeros(len(chunk), n, 2, device=device)
        msk = torch.zeros(len(chunk), n, dtype=torch.bool, device=device)
        for b, p in enumerate(chunk):
            i1, i2 = p // 2147483647, p % 2147483647
            m = raw[p]
            x1[b, :len(m)] = torch.as_tensor(kp[i1][m[:, 0]], device=device)
            x2[b, :len(m)] = torch.as_tensor(kp[i2][m[:, 1]], device=device)
            msk[b, :len(m)] = True
        inl, F, ok = verify_batch(x1, x2, msk, max_error, hypotheses, min_inliers)
        inl = inl.cpu().numpy(); Fc = F.cpu().numpy()
        for b, p in enumerate(chunk):
            i1, i2 = p // 2147483647, p % 2147483647
            m = raw[p]
            keep = inl[b, :len(m)]
            if keep.sum() >= min_inliers:
                out_pairs.append((i1, i2)); out_inl.append(m[keep]); out_F.append(Fc[b])
    n_written = write_two_view_geometries(db_path, out_pairs, out_inl, out_F)
    return dict(pairs_in=len(pids), pairs_verified=n_written,
                inliers=int(sum(len(x) for x in out_inl)), wall=time.time() - t0)
