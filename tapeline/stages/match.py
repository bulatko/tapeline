"""Matching: SuperPoint + LightGlue, pairs = sequential | retrieval.

For retrieval a frame is described either by DINOv2 (a separate model, extra inference cost)
or by VGGT features captured by the prior stage during its own pass (no extra cost) - see
cfg.retrieval. The VGGT descriptors give comparable verified pairs and trajectories.
Retrieval itself is required: without it the reconstruction drifts.

COLMAP's incremental mapper is not run here or later: it is the most expensive step
and also less accurate than the prior+BA combination. Only the COLMAP database and
geometric pair verification are used here.
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import pycolmap


def _hub_load_dino():
    """DINOv2 ViT-L/14 through the standard torch.hub cache: a previously downloaded copy of the
    repository is used offline; otherwise it is fetched from GitHub (facebookresearch/dinov2)."""
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if local.is_dir():
        return torch.hub.load(str(local), "dinov2_vitl14", source="local", pretrained=True)
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True, trust_repo=True)


def load_dino(fast_init=False):
    if fast_init:  # build the model directly on the GPU
        with torch.device("cuda"):
            m = _hub_load_dino()
        return m.eval().cuda()
    return _hub_load_dino().eval().cuda()


def _dino_descriptors(paths, batch=32, cfg=None):
    fast = bool(getattr(cfg, "fast_init", False))
    if cfg is not None and (getattr(cfg, "model_cache", False) or getattr(cfg, "prefetch", False)):
        from . import _models
        m = _models.get("dino", lambda: load_dino(fast))
        _models.drop("dino")
    else:
        m = load_dino(fast)
    nth = int(getattr(cfg, "match_io_threads", 0) or 0)
    if nth:
        from .. import cpu
        nth = cpu.limit(cfg, nth)
    if nth:  # read + resize frames in threads
        from concurrent.futures import ThreadPoolExecutor
        _ex = ThreadPoolExecutor(nth)
        rd = lambda q: cv2.resize(cv2.imread(str(q))[:, :, ::-1], (392, 224), interpolation=cv2.INTER_AREA)  # noqa: E731
        allims = list(_ex.map(rd, paths))
        _ex.shutdown()
    else:
        allims = None
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    D = np.zeros((len(paths), 1024), np.float32)
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            ims = allims[s:s + batch] if allims is not None else \
                [cv2.resize(cv2.imread(str(p))[:, :, ::-1], (392, 224),
                            interpolation=cv2.INTER_AREA) for p in paths[s:s + batch]]
            t = torch.from_numpy(np.ascontiguousarray(np.stack(ims))).cuda()
            t = ((t.permute(0, 3, 1, 2).float() / 255.) - mean) / std
            D[s:s + batch] = torch.nn.functional.normalize(m(t), dim=1).cpu().numpy()
    del m
    torch.cuda.empty_cache()
    return D


def _vggt_descriptors(cfg, n_frames):
    """Precomputed descriptors from prior.npz: the prior stage captures them during its own pass."""
    p = Path(cfg.paths()["prior"])
    if not p.exists():
        raise RuntimeError('retrieval="vggt" requires the prior stage to have been run')
    z = np.load(p, allow_pickle=True)
    D = np.asarray(z["descriptors"], np.float32) if "descriptors" in z.files else np.zeros(0)
    if D.ndim != 2 or len(D) != n_frames:
        raise RuntimeError(
            f'file {p} has no descriptors for {n_frames} frames (found {D.shape}). '
            'Re-run prior with force=True or set retrieval="dino"')
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)


def _descriptors(cfg, paths):
    if cfg.retrieval == "vggt":
        return _vggt_descriptors(cfg, len(paths))
    return _dino_descriptors(paths, cfg=cfg)


def _pad(feats, idx, n):
    """A batch needs the same number of keypoints: pad with zeros and remember how many
    are real in each frame - matches to padding are discarded afterwards."""
    dim = feats[idx[0]]["descriptors"].shape[-1]
    kp = torch.zeros(len(idx), n, 2, device="cuda")
    ds = torch.zeros(len(idx), n, dim, device="cuda")
    sz = torch.zeros(len(idx), 2, device="cuda")
    real = []
    for b, i in enumerate(idx):
        f = feats[i]
        m = f["keypoints"].shape[1]
        kp[b, :m] = f["keypoints"][0]
        ds[b, :m] = f["descriptors"][0]
        sz[b] = f["image_size"][0]
        real.append(m)
    return {"keypoints": kp, "descriptors": ds, "image_size": sz}, real


def _match_batched(lg, feats, pairs, min_score, budget):
    """Pairs are sorted by keypoint count to minimise padding, and the batch size is
    chosen from a B*N^2 budget: attention is quadratic in the number of keypoints, so a
    fixed batch size runs out of GPU memory on frames with many keypoints."""
    nk = [f["keypoints"].shape[1] for f in feats]
    order = sorted(range(len(pairs)), key=lambda k: max(nk[pairs[k][0]], nk[pairs[k][1]]))
    out, s = [], 0
    while s < len(order):
        n = max(nk[pairs[order[s]][0]], nk[pairs[order[s]][1]])
        B = max(1, min(32, int(budget / max(n * n, 1))))
        idx = order[s:s + B]
        n = max(max(nk[pairs[k][0]] for k in idx), max(nk[pairs[k][1]] for k in idx))
        d0, r0 = _pad(feats, [pairs[k][0] for k in idx], n)
        d1, r1 = _pad(feats, [pairs[k][1] for k in idx], n)
        res = lg({"image0": d0, "image1": d1})
        m_all = res["matches0"].cpu().numpy()
        sc_all = res["matching_scores0"].cpu().numpy()
        for b, k in enumerate(idx):
            i, j = pairs[k]
            m0 = m_all[b][:r0[b]]
            sc = sc_all[b][:r0[b]]
            keep = (m0 >= 0) & (sc >= min_score) & (m0 < r1[b])
            if keep.sum() >= 15:
                m = np.stack([np.flatnonzero(keep), m0[keep]], 1).astype(np.uint32)
                out.append((i, j, m))
        s += B
    return out


def run(cfg):
    p = cfg.paths()
    paths = sorted(x for x in Path(p["frames_sel"]).iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))   # Jupyter may put .ipynb_checkpoints next to the frames
    names = [x.name for x in paths]
    N = len(paths)
    H, W = cv2.imread(str(paths[0])).shape[:2]
    work = Path(cfg.work) / "sfm"; work.mkdir(parents=True, exist_ok=True)
    db_path = str(p["db"])
    db_final = db_path
    if getattr(cfg, "db_local", False):  # build the match database on a local (RAM) disk, copy it to work at the end
        import os
        import tempfile
        local_dir = getattr(cfg, "db_local_dir", None) or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
        db_path = os.path.join(local_dir, f"tapeline_{os.getpid()}_database.db")
    pairs_file = str(work / "pairs.txt")

    T = {}
    t0 = time.time()
    if getattr(cfg, "pairs_exhaustive", False):
        # unordered image set (e.g. ETH3D): all pairs, no retrieval or sequential window needed
        T["descriptors"] = 0.0
        pairs = {(i, j) for i in range(N) for j in range(i + 1, N)}
        n_seq = 0
    else:
        D = _descriptors(cfg, paths)
        T["descriptors"] = time.time() - t0
        t0 = time.time()
        S = D @ D.T
        pairs = {(i, i + d) for i in range(N) for d in range(1, cfg.seq_window + 1) if i + d < N}
        n_seq = len(pairs)
        for i in range(N):
            picked = []
            for j in np.argsort(-S[i]):
                j = int(j)
                if abs(j - i) < cfg.retrieval_knn or any(abs(j - q) < 5 for q in picked):
                    continue
                picked.append(j)
                pairs.add((min(i, j), max(i, j)))
                if len(picked) >= cfg.retrieval_knn:
                    break
    n_extra = 0
    if cfg.extra_pairs:
        # user-supplied bridges for weakly connected segments (e.g. a small room linked only through a doorway,
        # where retrieval proposes look-alike pairs from another room and the true doorway pairs do not survive)
        by = {n: i for i, n in enumerate(names)}
        for line in Path(cfg.extra_pairs).read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) >= 2 and f[0] in by and f[1] in by and f[0] != f[1]:
                i, j = by[f[0]], by[f[1]]
                if (min(i, j), max(i, j)) not in pairs:
                    pairs.add((min(i, j), max(i, j))); n_extra += 1
    pairs = sorted(pairs)
    T["pairs"] = time.time() - t0

    t0 = time.time()
    from lightglue import LightGlue, SuperPoint
    sp = SuperPoint(max_num_keypoints=cfg.max_keypoints).eval().cuda()
    # depth/width_confidence=-1 disable adaptive pruning: batched inference fails without it
    # (ind0[k, ...] assumes a single element). Match counts differ slightly from the
    # pruned, unbatched path; batching makes matching considerably faster.
    # mp=True enables autocast for the whole transformer: attention already runs in half
    # precision, this also covers the projections and MLP - faster and less memory, with
    # practically identical matches.
    lg = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1,
                   mp=cfg.match_mp).eval().cuda()

    T["load_models"] = time.time() - t0
    feats, kpts = [None] * N, [None] * N
    long_side = max(H, W)
    with torch.no_grad():
        t0 = time.time()
        if getattr(cfg, "match_io_threads", 0):  # read jpgs in threads, overlapping with SuperPoint
            from concurrent.futures import ThreadPoolExecutor
            from .. import cpu
            _ex = ThreadPoolExecutor(cpu.limit(cfg, cfg.match_io_threads))
            ims = _ex.map(lambda q: cv2.imread(str(q)), paths)
        else:
            ims = (cv2.imread(str(q)) for q in paths)
        for i, (path, im) in enumerate(zip(paths, ims)):
            im = im[:, :, ::-1]
            t = torch.from_numpy(np.ascontiguousarray(im)).float().permute(2, 0, 1)[None] / 255.
            f = sp.extract(t.cuda())
            feats[i] = f
            kpts[i] = f["keypoints"][0].cpu().numpy().copy().astype(np.float32)
        # guard against keypoints in the wrong scale: that would silently break all triangulation
        allkp = np.concatenate(kpts)
        assert allkp[:, 0].max() <= W + 1 and allkp[:, 1].max() <= H + 1, "keypoint scale"
        T["superpoint"] = time.time() - t0

        t0 = time.time()
        matched = _match_batched(lg, feats, pairs, cfg.min_match_score, cfg.match_budget)
        T["lightglue"] = time.time() - t0
    del feats
    torch.cuda.empty_cache()

    t0 = time.time()
    for suffix in ("", "-wal", "-shm"):
        f = Path(db_path + suffix)
        if f.exists():
            f.unlink()
    db = pycolmap.Database.open(db_path)
    # focal length for the initial camera: from the prior if it has been computed, otherwise a rough guess.
    # 0.95*long side can be far off for phone videos (almost 2x) - BA fixes it later, but pair
    # verification would run with a wrong camera.
    focal = cfg.focal_init
    if focal is None and Path(p["prior"]).exists():
        zp = np.load(p["prior"], allow_pickle=True)
        Kp = zp["intrinsics"] if "intrinsics" in zp else None
        if Kp is not None and np.size(Kp) and np.asarray(Kp).ndim == 3:
            Kp = np.asarray(Kp)
            focal = float(np.median(Kp[:, 0, 0])) * W / (2 * float(np.median(Kp[:, 0, 2])))
    focal = focal or 0.95 * long_side
    cam = pycolmap.Camera(model=cfg.camera_model, width=W, height=H,
                          params=[focal, focal, W / 2, H / 2, 0, 0, 0, 0])
    cam.has_prior_focal_length = False
    txn = pycolmap.DatabaseTransaction(db) if getattr(cfg, "db_txn", False) else None  # single commit
    cid = db.write_camera(cam)
    iids = [db.write_image(pycolmap.Image(name=n, camera_id=cid)) for n in names]
    for i in range(N):
        db.write_keypoints(iids[i], kpts[i])
    with open(pairs_file, "w") as pf:
        for i, j, m in matched:
            db.write_matches(iids[i], iids[j], m)
            pf.write(f"{names[i]} {names[j]}\n")
    if txn is not None:
        del txn  # the DatabaseTransaction destructor performs COMMIT
    db.close()
    T["write_db"] = time.time() - t0

    t0 = time.time()
    if cfg.verify_backend == "gpu":
        from ..sfm_gpu.verify import verify_database
        vinfo = verify_database(db_path, hypotheses=cfg.verify_hypotheses)
    else:
        pycolmap.verify_matches(db_path, pairs_file)
        vinfo = None
    T["verify"] = time.time() - t0
    with sqlite3.connect(db_path) as c:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        nv = c.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows>0").fetchone()[0]
        ninl = c.execute("SELECT SUM(rows) FROM two_view_geometries").fetchone()[0]
    if db_path != db_final:
        import shutil
        t0 = time.time()
        Path(db_final).parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("-wal", "-shm"):
            if Path(db_final + suffix).exists():
                Path(db_final + suffix).unlink()
        shutil.copyfile(db_path, db_final)
        for suffix in ("", "-wal", "-shm"):
            if Path(db_path + suffix).exists():
                Path(db_path + suffix).unlink()
        T["db_copy"] = time.time() - t0
        db_path = db_final

    info = dict(frames=N, pairs_total=len(pairs), pairs_seq=n_seq, pairs_extra=n_extra, pairs_kept=len(matched),
                verified_pairs=int(nv), inliers=int(ninl or 0),
                keypoints_median=int(np.median([len(k) for k in kpts])), focal_init=float(focal),
                retrieval=cfg.retrieval, verify=cfg.verify_backend, verify_info=vinfo,
                times={k: round(v, 2) for k, v in T.items()})
    (work / "match.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info
