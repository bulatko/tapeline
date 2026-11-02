"""Pose refinement: programmatic template from the database + GPU triangulation + GPU bundle adjustment.

The COLMAP mapper is not run: the reconstruction template is built from database.db
in milliseconds and initial poses come from the prior. The round schedule follows
the same logic as the CPU version: thresholds 16->8->4->2->1.5 px, Cauchy
8->4->2->1->trivial, degrees of freedom are released gradually.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import pycolmap

from ..sfm_gpu import bacore as B, gpuba as GB, trigpu as TG
from ..sfm_gpu.template import build_template
from .chain import chain_poses, sim3 as _sim3

SCHEDULE = [
    dict(thr=16.0, min_ang=0.5,  loss=8.0,  focal=False, extra=False, pp=False),
    dict(thr=8.0,  min_ang=0.5,  loss=4.0,  focal=True,  extra=False, pp=False),
    dict(thr=4.0,  min_ang=0.75, loss=2.0,  focal=True,  extra=True,  pp=False),
    dict(thr=2.0,  min_ang=1.0,  loss=1.0,  focal=True,  extra=True,  pp=True),
    dict(thr=1.5,  min_ang=1.0,  loss=None, focal=True,  extra=True,  pp=True),
]


def components(rec):
    """Number of connected components over frames that share a 3D point. Only frames with observations
    are counted: in the programmatic template every frame has a pose, so a pose-based count checks nothing."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    reg = [im.image_id for im in rec.images.values()
           if any(p2.has_point3D() for p2 in im.points2D)]
    for i in reg:
        find(i)
    for p3 in rec.points3D.values():
        els = [el.image_id for el in p3.track.elements]
        for k in range(1, len(els)):
            ra, rb = find(els[0]), find(els[k])
            if ra != rb:
                parent[ra] = rb
    return len({find(i) for i in reg}) if reg else 0


def prune_stray(rec, max_frac=0.01, max_abs=3):
    """Stray frames: tiny components (<= max(max_abs, max_frac·N) frames) outside the main map - a burst of motion
    blur or a turn towards a blank wall, with no verified pair to the rest of the set. Their observations are
    removed (the frame stays in the model without 3D points and is excluded from depth/fusion); names are returned
    for the manifest. A component larger than the threshold is NOT touched - it is a genuine second map, caught by
    the poses gate."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    reg = [im.image_id for im in rec.images.values() if any(p2.has_point3D() for p2 in im.points2D)]
    for i in reg:
        find(i)
    for p3 in rec.points3D.values():
        els = [el.image_id for el in p3.track.elements]
        for k in range(1, len(els)):
            ra, rb = find(els[0]), find(els[k])
            if ra != rb:
                parent[ra] = rb
    comps = {}
    for i in reg:
        comps.setdefault(find(i), []).append(i)
    if len(comps) <= 1:
        return []
    big = max(comps.values(), key=len)
    lim = max(max_abs, int(max_frac * len(reg)))
    pruned = []
    for c in comps.values():
        if c is big or len(c) > lim:
            continue
        for i in c:
            im = rec.images[i]
            for idx, p2 in enumerate(im.points2D):
                if p2.has_point3D():
                    rec.delete_observation(i, idx)
            pruned.append(im.name)
    return sorted(pruned)


def prune_sparse(rec, min_obs=30, zero=False):
    """Frames with fewer than min_obs 3D observations in the main map are not constrained: BA can place them
    anywhere while keeping a normal reprojection error. Their observations are removed (as for prune_stray),
    and such frames are excluded from depth and fusion. -> names."""
    pruned = []
    for im in rec.images.values():
        idx = [i for i, p2 in enumerate(im.points2D) if p2.has_point3D()]
        # zero=True (flag sparse_zero_obs): also frames with exactly 0 observations - otherwise a frame that lost all
        # points during retriangulation gets no prior-chain pose and BA lets its centre drift far away. Default: 0 < n.
        if (zero or len(idx) > 0) and len(idx) < min_obs:
            for i in idx:
                if im.points2D[i].has_point3D():      # removing a neighbouring observation may have deleted the point (2-view track) - otherwise Point3D -1
                    rec.delete_observation(im.image_id, i)
            pruned.append(im.name)
    return sorted(pruned)


def pose_outliers(rec, k=10.0, skip=()):
    """Frames whose centre after BA is farther than k·R from the median trajectory centre, where R is the median
    of these distances (a robust scene radius in the scene's own units; the threshold is relative, not metric).
    Normal scenes have max/R of a few units; an exploded pose is orders of magnitude beyond. Outlier observations
    are removed as for sparse frames, and the pose is set by refill_sparse (VGGT prior chain from neighbours).
    -> [(name, d/R)]"""
    ims = list(rec.images.values())
    if len(ims) < 5:
        return []
    C = np.array([im.projection_center() for im in ims])
    d = np.linalg.norm(C - np.median(C, 0), axis=1)
    R = float(np.median(d))
    if not R > 0:
        return []
    out = []
    for im, di in zip(ims, d):
        if di > k * R and im.name not in skip:  # frames already pruned as sparse/stray get refilled anyway
            for i, p2 in enumerate(im.points2D):
                if p2.has_point3D():
                    rec.delete_observation(im.image_id, i)
            out.append((im.name, round(float(di / R), 1)))
    return sorted(out)


def refill_sparse(rec, cfg, sparse_names, min_obs):
    """Frames dropped from the map (prune_sparse) get a VGGT prior-chain pose from the nearest frames (in sequence
    order) with ≥ min_obs observations: consecutive dropped frames form one chain between two anchors (chain_poses
    blends the predictions from A and B); at the start/end of the recording a single anchor is used. The prior->model
    scale is a robust sim3 over the centres of good frames. Rationale: for dropped frames the BA pose is far less
    accurate than the prior. -> report."""
    if not sparse_names:
        return []
    prior_path = Path(cfg.paths()["prior"])
    if not prior_path.exists():
        return []
    z = np.load(prior_path, allow_pickle=True)
    if "extrinsics" not in z.files:
        return []
    E = np.asarray(z["extrinsics"], np.float64)
    pi = {str(n): i for i, n in enumerate(z["image_names"])}
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    names = [im.name for im in imgs]
    if any(n not in pi for n in names):
        return []
    obs = np.array([sum(1 for p2 in im.points2D if p2.has_point3D()) for im in imgs])
    good = obs >= min_obs
    if good.sum() < 10:
        return []
    # target: ALL frames without reliable observations - dropped (sparse), stray, and frames that never got any points
    # (obs == 0 from the start; prune_sparse does not see them, its condition is 0 < n); none of them has a constrained BA pose
    sparse_names = set(sparse_names) | {names[k] for k in range(len(imgs)) if not good[k]}

    def c2w_of(im):
        p = im.cam_from_world()
        R, tt = np.asarray(p.rotation.matrix()), np.asarray(p.translation)
        M = np.eye(4); M[:3, :3] = R.T; M[:3, 3] = -R.T @ tt
        return M

    def prior_c2w(n):
        Ex = E[pi[n]]
        M = np.eye(4); M[:3, :3] = Ex[:3, :3].T; M[:3, 3] = -Ex[:3, :3].T @ Ex[:3, 3]
        return M

    C_ba = np.array([c2w_of(im)[:3, 3] for im in imgs])
    C_pr = np.array([prior_c2w(n)[:3, 3] for n in names])
    s, R, tt = _sim3(C_pr[good], C_ba[good])
    res = np.linalg.norm((s * (R @ C_pr.T).T + tt) - C_ba, axis=1)
    for _ in range(3):
        inl = good & (res < np.percentile(res[good], 70))
        if inl.sum() < 10:
            break
        s, R, tt = _sim3(C_pr[inl], C_ba[inl])
        res = np.linalg.norm((s * (R @ C_pr.T).T + tt) - C_ba, axis=1)
    frame_of = {}
    for im in rec.images.values():
        fid = im.frame_id() if callable(im.frame_id) else im.frame_id
        frame_of[im.name] = rec.frames[fid]
    target = set(sparse_names)
    report = []
    k = 0
    while k < len(imgs):
        if names[k] not in target:
            k += 1; continue
        k1 = k
        while k1 + 1 < len(imgs) and names[k1 + 1] in target:
            k1 += 1
        a = next((j for j in range(k - 1, -1, -1) if good[j]), None)
        b = next((j for j in range(k1 + 1, len(imgs)) if good[j]), None)
        if a is None and b is None:
            k = k1 + 1; continue
        if a is None:                      # start of the recording: only the trailing anchor, walk backwards from it
            chain = list(range(b, k - 1, -1)); new = chain_poses(c2w_of(imgs[b]), None, [prior_c2w(names[j]) for j in chain], s)
            inner = chain[1:]
        else:
            chain = list(range(a, (b if b is not None else k1) + 1))
            new = chain_poses(c2w_of(imgs[a]), c2w_of(imgs[b]) if b is not None else None, [prior_c2w(names[j]) for j in chain], s)
            inner = chain[1:-1] if b is not None else chain[1:]
        for j, M in zip(inner, new):
            Rw2c = M[:3, :3].T
            frame_of[names[j]].rig_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(np.ascontiguousarray(Rw2c)), np.ascontiguousarray(-Rw2c @ M[:3, 3]))
            report.append(dict(frame=names[j], obs=int(obs[j]), anchors=([names[a]] if a is not None else []) + ([names[b]] if b is not None else []),
                               shift=float(np.linalg.norm(M[:3, 3] - C_ba[j]))))
        k = k1 + 1
    return report


def _far_links(rec, dt=15.0, fps=30.0):
    """Per frame (in name order): time (NNNNNN.jpg at fps frames/s - Config.frame_fps), number of 3D observations and number
    of "far" points - shared with a frame more than dt seconds away. Returns (imgs, t, obs, far), or None if names do not match the pattern."""
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    try:
        t = np.array([int(im.name.split(".")[0]) / float(fps) for im in imgs])
    except ValueError:
        return None
    idx = {im.image_id: k for k, im in enumerate(imgs)}
    far = np.zeros(len(imgs), int)
    obs = np.zeros(len(imgs), int)
    for p3 in rec.points3D.values():
        ks = [idx[el.image_id] for el in p3.track.elements]
        for k in ks:
            obs[k] += 1
            if any(abs(t[k2] - t[k]) > dt for k2 in ks):
                far[k] += 1
    return imgs, t, obs, far


def effective_dt(rec, dt=15.0, frac=0.15, fps=30.0):
    """A "far" link is relative to the recording length: min(dt, frac x duration). On short recordings an absolute 15 s
    would mark most frames as weak; on long recordings (≥ 100 s) the threshold stays at dt."""
    try:
        t = np.array([int(im.name.split(".")[0]) / float(fps) for im in rec.images.values()])
    except ValueError:
        return dt
    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    return float(min(dt, max(frac * span, 1.0)))


def weak_segments(rec, dt=15.0, min_len=6, max_frac=0.4, fps=30.0):
    """Segments without far bridges: a run of ≥ min_len consecutive frames, none of which shares a 3D point with a
    frame more than dt seconds away (dt is the effective one, see effective_dt), hangs on a chain of sequential
    matches - its poses are verified by nothing but their neighbours. Typical case: a small side room entered
    through a door, where BA keeps a low reprojection error but drifts the segment relative to the self-consistent
    VGGT chain between the door anchors. A segment longer than max_frac of all frames is not considered weak: that
    indicates a detector failure rather than the data."""
    dt = effective_dt(rec, dt, fps=fps)
    fl = _far_links(rec, dt, fps=fps)
    if fl is None:
        return []
    imgs, t, obs, far = fl
    out, start = [], None
    for k in range(len(imgs) + 1):
        weak = k < len(imgs) and obs[k] > 0 and far[k] == 0
        if weak and start is None:
            start = k
        if not weak and start is not None:
            if min_len <= k - start <= max_frac * len(imgs):
                out.append(dict(k0=start, k1=k - 1, t0=float(t[start]), t1=float(t[k - 1]), frames=int(k - start),
                                obs_median=float(np.median(obs[start:k])), dt_s=float(dt)))
            start = None
    return out


def rot_scene_stats(rec, cfg, w=3, abs_deg=1.5, frac_thr=0.3):
    """SCENE-level provenance (always computed, cheap): disagreement of relative rotations BA<->prior over the whole recording -
    median of the feature and fraction of frames with feature > abs_deg. Healthy scenes have a sub-degree median and a small
    fraction above abs_deg; metre-scale BA failures show a large fraction. flagged = fraction ≥ frac_thr: "do not trust BA"
    (the action is up to the caller; only the flag is set here)."""
    fl = _far_links(rec, fps=getattr(cfg, "frame_fps", 30.0))
    prior_path = Path(cfg.paths()["prior"])
    if fl is None or not prior_path.exists():
        return None
    imgs, t, obs, far = fl
    z = np.load(prior_path, allow_pickle=True)
    if "extrinsics" not in z.files:
        return None
    E = np.asarray(z["extrinsics"], np.float64)
    pi = {str(n): i for i, n in enumerate(z["image_names"])}
    names = [im.name for im in imgs]
    if any(n not in pi for n in names) or len(imgs) < 2 * w + 2:
        return None
    Rb = np.stack([np.asarray(im.cam_from_world().rotation.matrix()) for im in imgs])
    Rp = np.stack([E[pi[n]][:3, :3] for n in names])
    Rr = np.einsum("nij,nik->njk", np.einsum("nij,nik->njk", Rb[:-1], Rb[1:]), np.einsum("nij,nik->njk", Rp[:-1], Rp[1:]))
    step = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    feat = np.array([np.median(step[max(0, k - w):min(len(step), k + w)]) for k in range(len(imgs))])
    m = obs > 0
    if not m.any():
        return None
    frac = float(np.mean(feat[m] > abs_deg))
    return dict(median_deg=float(np.median(feat[m])), p90_deg=float(np.percentile(feat[m], 90)), frac_above=frac, abs_deg=abs_deg,
                frac_thr=frac_thr, flagged=bool(frac >= frac_thr), frames=int(m.sum()))


def centre_spread(rec):
    """Model scale: median distance of camera centres (with observations) to their centroid - gauge provenance per round."""
    C = [(-im.cam_from_world().rotation.matrix().T @ im.cam_from_world().translation)
         for im in rec.images.values() if any(p.has_point3D() for p in im.points2D)]
    if len(C) < 2:
        return 0.0
    C = np.asarray(C)
    return float(np.median(np.linalg.norm(C - np.median(C, axis=0), axis=1)))   # median centre: a single runaway frame has no influence


def stats(rec, fps=30.0):
    prob = B.export_problem(rec)
    st = B.stats(B.reproj_errors(prob))
    return dict(frames_with_obs=int(len(np.unique(prob["fr"]))) if len(prob["fr"]) else 0, centre_spread=centre_spread(rec),
                frames=rec.num_images(), points=len(rec.points3D), components=components(rec),
                weak_segments=weak_segments(rec, fps=fps),
                camera=[float(x) for x in list(rec.cameras.values())[0].params], **st)


def _hstats(rec, cfg):
    """Statistics of intermediate rounds (only in sfm.json history) - full or light (sfm_hist_stats=False):
    full stats() = export_problem + components + weak_segments, which is a noticeable share of the stage runtime."""
    if getattr(cfg, "sfm_hist_stats", True):
        return stats(rec, cfg.frame_fps)
    d = dict(frames=rec.num_images(), points=len(rec.points3D))
    print("SFM_ROUND", json.dumps(d), flush=True)  # round trace (to catch a model that became empty)
    return d


def _seed(cfg, db_path):
    z = np.load(cfg.paths()["prior"], allow_pickle=True)
    E = np.asarray(z["extrinsics"], np.float64)
    names = [str(s) for s in z["image_names"]]
    K = z["intrinsics"] if "intrinsics" in z and np.size(z["intrinsics"]) else None
    import cv2
    first = Path(cfg.paths()["frames_sel"]) / names[0]
    H, W = cv2.imread(str(first)).shape[:2]
    if cfg.focal_init:
        f = cfg.focal_init
    elif K is not None and K.ndim == 3:
        # prior intrinsics are usually at its working resolution - convert to frame pixels
        f = float(np.median(K[:, 0, 0])) * W / (2 * float(np.median(K[:, 0, 2])))
    else:
        f = 0.95 * max(H, W)
    rec = build_template(str(db_path), np.array([f, f, W / 2, H / 2, 0, 0, 0, 0]))
    frame_of = {}
    for im in rec.images.values():
        fid = im.frame_id() if callable(im.frame_id) else im.frame_id
        frame_of[im.name] = rec.frames[fid]
    # prior frames without a single verified pair are not in the template (template.images_with_pairs):
    # there is nowhere to set their pose, and they are excluded downstream (depth, fusion) - like prune_stray frames
    no_pairs = [nm for nm in names if nm not in frame_of]
    for k, nm in enumerate(names):
        if nm not in frame_of:
            continue
        frame_of[nm].rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.ascontiguousarray(E[k][:3, :3])),
            np.ascontiguousarray(E[k][:3, 3]))
    if not frame_of:
        raise RuntimeError("no frames with verified pairs - matching is empty")
    return rec, float(f), sorted(no_pairs)


def run(cfg):
    p = cfg.paths()
    out = Path(p["sfm"]); out.mkdir(parents=True, exist_ok=True)
    db_path, img = str(p["db"]), str(p["frames_sel"])
    retri = [int(x) for x in cfg.retri_rounds.split(",")]
    t0 = time.time()

    rec, focal, no_pairs = _seed(cfg, db_path)
    if not cfg.refine:
        rec = pycolmap.triangulate_points(rec, db_path, img, str(out / "tri"), True,
                                          TG.pipe_opts(4.0, 0.75), False)
        rec.write(str(out)); st = stats(rec, cfg.frame_fps)
        return dict(mode="as_is", focal_init=focal, no_pair_frames=no_pairs, wall=time.time() - t0, **st)

    tg = TG.TriGPU(db_path, img, str(out / "tri"))
    hist, times = [], []
    rounds_on = [int(x) for x in str(getattr(cfg, "sfm_rounds", "0,1,2,3,4")).split(",")]  # subset of rounds to run
    for i, s in enumerate(SCHEDULE):
        if i not in rounds_on:
            continue
        pre = out / f"pre{i}"; pre.mkdir(exist_ok=True); rec.write(str(pre))
        t_tri = 0.0
        if i in retri:
            n_before = len(rec.points3D)
            rec = pycolmap.Reconstruction(str(pre))
            t = time.time()
            rec, phases = tg.run(rec, s["thr"], s["min_ang"], verbose=False)
            # sfm_collapse_guard: non-deterministic BA drift can make retriangulation lose almost all points -
            # roll back to the model after BA round 0 and repeat this round's retriangulation
            if (getattr(cfg, "sfm_collapse_guard", False) and i > 0 and n_before > 0
                    and len(rec.points3D) < 0.3 * n_before and (out / "stage0").exists()):
                print("SFM_COLLAPSE", json.dumps(dict(round=i, before=n_before, after=len(rec.points3D))), flush=True)
                hist.append(dict(tag=f"collapse{i}", before=n_before, after=len(rec.points3D)))
                rec = pycolmap.Reconstruction(str(out / "stage0"))
                rec, phases = tg.run(rec, s["thr"], s["min_ang"], verbose=False)
            t_tri = time.time() - t
            hist.append(dict(tag=f"tri{i}", phases=phases, **_hstats(rec, cfg)))
        t = time.time()
        bx = B.make_exporter(rec)
        prob = bx.export(rec)
        gp = GB.Problem(prob["R"], prob["t"], prob["X"], prob["cam"],
                        prob["pt"], prob["fr"], prob["xy"], device="cuda")
        torch.cuda.synchronize(); t_setup = time.time() - t
        t = time.time()
        res = GB.solve(gp, loss_scale=s["loss"], refine_focal=s["focal"],
                       refine_pp=s["pp"], refine_extra=s["extra"], pair_chunk=int(5e6),
                       max_iter=getattr(cfg, "ba_max_iter", 100))  # cap on LM iterations per BA round
        torch.cuda.synchronize(); t_ba = time.time() - t
        bx.import_problem(rec, prob, res["R"].cpu().numpy(), res["t"].cpu().numpy(),
                          res["X"].cpu().numpy(), res["cam"].cpu().numpy())
        gauge = res.get("gauge")
        del gp, prob, res
        hist.append(dict(tag=f"ba{i}", gauge=gauge, **_hstats(rec, cfg)))
        times.append(dict(round=i, tri=t_tri, ba=t_ba, setup=t_setup))
        st = out / f"stage{i}"; st.mkdir(exist_ok=True); rec.write(str(st))

    if not getattr(cfg, "sfm_hist_stats", True):
        print("SFM_PRE_PRUNE", json.dumps(dict(frames=rec.num_images(), points=len(rec.points3D))), flush=True)
    stray = prune_stray(rec)
    zero = bool(getattr(cfg, "sparse_zero_obs", False))
    if zero:
        zf = sorted(im.name for im in rec.images.values() if not any(p2.has_point3D() for p2 in im.points2D))
        if zf:
            print("SFM_ZERO_OBS", json.dumps(zf), flush=True)
    sparse = prune_sparse(rec, cfg.min_frame_obs, zero=zero) if cfg.min_frame_obs > 0 else []
    outl = (pose_outliers(rec, getattr(cfg, "pose_outlier_k", 10.0), skip=set(sparse) | set(stray))
            if getattr(cfg, "pose_outlier_guard", False) else [])
    if outl:  # camera-centre outlier after BA - handled like sparse: observations removed, pose from the prior chain (refill_sparse)
        print("SFM_POSE_OUTLIER", json.dumps(outl), flush=True)
        sparse = sorted(set(sparse) | {n for n, _ in outl})
    # both dropped (sparse) and stray frames get a prior-chain pose: neither has a constrained BA pose
    refilled = refill_sparse(rec, cfg, sorted(set(sparse) | set(stray)), cfg.min_frame_obs) if ((sparse or stray) and getattr(cfg, "sparse_refill", True)) else []
    try:
        rot_scene = rot_scene_stats(rec, cfg)
    except Exception as ex:                                # provenance only: must not fail the stage
        rot_scene = dict(error=repr(ex))
    rec.write(str(out))
    if not getattr(cfg, "sfm_hist_stats", True):
        print("SFM_FINAL", json.dumps(dict(frames=rec.num_images(), points=len(rec.points3D), stray=len(stray),
                                           sparse=len(sparse))), flush=True)
    if getattr(cfg, "pose_outlier_guard", False):
        hist.append(dict(tag="pose_outliers", frames=[n for n, _ in outl], ratio=[r for _, r in outl]))
    info = dict(mode="refine", focal_init=focal, wall=time.time() - t0, stray_frames=stray, sparse_frames=sparse, sparse_refilled=refilled,
                no_pair_frames=no_pairs, rot_scene=rot_scene, rounds=times, history=hist,
                **stats(rec, cfg.frame_fps))
    (Path(cfg.work) / "sfm.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info
