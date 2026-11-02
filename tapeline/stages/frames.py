"""Frames: video decoding and sharpness-based selection.

The window is chosen automatically to hit a target frame count: the config specifies
"how many frames", not "which window". ffmpeg always runs with -vsync 0: without it a VFR
video is padded to CFR with duplicated frames, producing more frames than actually exist.
"""
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import cv2


def _lapvar(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return -1.0 if im is None else float(cv2.Laplacian(im, cv2.CV_64F).var())


def extract(video, out_dir, quality=2):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if not any(out.glob("*.jpg")):
        # VFR videos can contain frames with identical pts; with -vsync 0 they reach the mjpeg encoder (mpegvideo_enc),
        # which with frame threading tracks the "last pts" per thread - two duplicates in one thread
        # -> "Invalid pts (N) <= last (N)" and ffmpeg fails (intermittently, depends on how frames land on threads).
        # setpts=N/(FRAME_RATE*TB) makes the timestamp equal to the frame index in the encoder time base (1/FRAME_RATE;
        # a bare setpts=N in a 1/15360 base collapses again when converted to 1/30): strictly increasing, same set of
        # frames (-vsync 0), same JPEG bytes. Frame times are taken below from ffprobe on the source video; JPEG
        # timestamps are not used anywhere.
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vsync", "0", "-vf", "setpts=N/(FRAME_RATE*TB)",
                        "-q:v", str(quality), str(out / "%06d.jpg")], check=True)
    times = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pkt_pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout
    t = [float(x) for x in times.split() if x.strip()]
    paths = sorted(out.glob("*.jpg"))
    assert len(paths) == len(t), f"frames {len(paths)} != timestamps {len(t)}"
    return paths, np.array(t)


def _fill_gaps(sel, t, score, ok, max_gap):
    """Fill every gap > max_gap with the sharpest valid frame inside it, repeating until no gaps remain.
    On long videos the window is ~ length/target, and the sharpest frames of adjacent windows can sit at the far edges,
    giving a gap of up to two windows. The sharp frame is searched only where it closes the gap with the minimum number of
    insertions k = ceil(G/max_gap) - 1: t in [t_b - k·max_gap, t_a + max_gap]; if that is empty (frame discreteness),
    in (t_a, t_a + max_gap]. Without this narrowing the sharpest frame of a gap is often right next to an edge, and
    insertions creep forward one frame at a time, inflating the frame count (and memory use of the pose prior).
    A gap without any valid frame (a real hole in the video) is left alone - the max_gap check below catches it."""
    sel = list(sel)
    while True:
        add = []
        for a, b in zip(sel[:-1], sel[1:]):
            gap = t[b] - t[a]
            if gap > max_gap:
                cand = np.arange(a + 1, b)
                cand = cand[ok[cand] & (t[cand] - t[a] <= max_gap)]  # same difference as np.diff in the gap check
                k = int(np.ceil(gap / max_gap)) - 1
                near = cand[t[cand] >= t[b] - k * max_gap]
                cand = near if len(near) else cand
                if len(cand):
                    add.append(int(cand[np.argmax(score[cand])]))
        if not add:
            return sel
        sel = sorted(set(sel) | set(add))


def select(paths, t, sel_dir, target=600, lo=500, hi=700, max_gap=1.5, workers=32):
    cv2.setNumThreads(1)
    with Pool(workers) as p:
        lap = np.array(p.map(_lapvar, [str(x) for x in paths], chunksize=8))
    ok = lap > 0

    def pick(win):
        score = np.where(ok, lap, -1e9)
        out = []
        for a in np.arange(t[0], t[-1] + win, win):
            m = (t >= a) & (t < a + win)
            if m.any():
                idx = np.flatnonzero(m)
                out.append(int(idx[np.argmax(score[idx])]))
        return sorted(set(out))

    best = None
    for win in np.arange(0.005, 2.0, 0.005):  # fine 0.005 s step so that closely spaced raw VFR frames can fall into separate windows
        s = pick(round(float(win), 4))
        if lo <= len(s) <= hi and (best is None or abs(len(s) - target) < abs(len(best[1]) - target)):
            best = (round(float(win), 4), s)
    if best is None:  # small targets on long videos need windows > 2 s
        for win in np.arange(2.0, 30.0, 0.05):
            s = pick(round(float(win), 4))
            if lo <= len(s) <= hi and (best is None or abs(len(s) - target) < abs(len(best[1]) - target)):
                best = (round(float(win), 4), s)
    if best is None:
        raise RuntimeError("could not reach the target frame count range; widen frames_lo/hi")
    win, sel = best
    n_win = len(sel)
    sel = _fill_gaps(sel, t, lap, ok, max_gap)

    gaps = np.diff(t[sel])
    if gaps.max() > max_gap:
        raise RuntimeError(f"time gap {gaps.max():.2f} s exceeds max_gap {max_gap}")

    d = Path(sel_dir); d.mkdir(parents=True, exist_ok=True)
    for i in sel:
        dst = d / Path(paths[i]).name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(paths[i]).resolve())

    thr = float(np.percentile(lap[ok], 5))
    risky = [int(i) for i in sel if lap[i] < thr]
    info = dict(window_s=win, count=len(sel), max_gap_s=float(gaps.max()),
                lap_median=float(np.median(lap[sel])), lap_median_all=float(np.median(lap[ok])),
                risky_frames=len(risky),
                frames=[dict(idx=int(i), file=Path(paths[i]).name, t=float(t[i]),
                             lap=float(lap[i])) for i in sel])
    if len(sel) > n_win:  # only when gaps were filled - without gaps the output is byte-identical
        info["gap_fill"] = dict(count_window=n_win, added=len(sel) - n_win, over_hi=max(0, len(sel) - hi))
    return info


def _probe(video):
    """Frames are the DECODED frames in decoder output order - same as extract() (ffmpeg -vsync 0, setpts=N:
    frame index = its index at the decoder output). Packets are not counted at all: edit-list pre-roll (packets with
    pts < 0 that the decoder legitimately drops), repeated pts and packets without a frame do not affect the count.
    One sequential decode pass without pixel conversion -> frame pts (int, stream time base; a frame without pts gets
    the previous one's), keyframe flags, time base."""
    import av
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        pts, key, last = [], [], 0
        for fr in c.decode(s):
            p = fr.pts if fr.pts is not None else last
            pts.append(int(p))
            key.append(bool(fr.key_frame))
            last = p
        return np.array(pts, dtype=np.int64), np.array(key, dtype=bool), float(s.time_base)


def _decode_range(args):
    """Worker: decode from keyframe start_idx (decoder index from _probe) and map each decoded frame to its decoder index
    using the pts sequence of the probe pass (cursor: nearest i ≥ cursor with the same pts - repeated pts are told apart
    by order; a frame not found ahead precedes our keyframe and is skipped).
    mode='win': frames with indices [lo, hi) - sharpness of each and the best per window (window by selection
    position, wpos); mode='want': frames at positions in want. jpg named by selection position: f"{pos + 1:06d}.jpg"."""
    import av
    video, lo, hi, all_pts, start_idx, pos_of, wpos, out_dir, mode, want, opt = args
    size = (opt or {}).get("size")
    cv2.setNumThreads(1)
    n = len(all_pts)
    lap, best = {}, {}
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        if start_idx > 0:
            c.seek(int(all_pts[start_idx]), stream=s, backward=True, any_frame=False)
        cur = start_idx
        for fr in c.decode(s):
            p = fr.pts
            i = cur
            while i < n and all_pts[i] != p and i < cur + 512:
                i += 1
            if i >= n or all_pts[i] != p:
                continue                      # frame before our keyframe (seek landed earlier / open-GOP leading frames)
            cur = i + 1
            if i < lo:
                continue
            if i >= hi:
                break
            pos = i if pos_of is None else pos_of.get(i)
            if pos is None:
                continue
            if mode == "want":
                if pos in want:
                    best[pos] = (0.0, pos, fr.to_ndarray(format="bgr24"))
                continue
            y = fr.planes[0]
            Y = np.frombuffer(y, np.uint8).reshape(fr.height, y.line_size)[:, :fr.width]
            if size is not None:
                Y = cv2.resize(Y, size, interpolation=cv2.INTER_AREA)
            v = float(cv2.Laplacian(Y, cv2.CV_32F).var())
            lap[pos] = v
            w = int(wpos[pos])
            if w not in best or v > best[w][0]:
                best[w] = (v, pos, fr.to_ndarray(format="bgr24"))
    sel = []
    for _, pos, img in best.values():
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(Path(out_dir) / f"{pos + 1:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sel.append(pos)
    return lap, sel


def _src_opts(cfg, video, pts, tb):
    """Select directly from the source video instead of a re-encoded normalized copy (fps <= frames_src_fps, long side
    <= frames_long_side). Candidates are the frames ffmpeg -vf fps=F would keep (nearest to tick k/F, rounding near);
    frames are resized with INTER_AREA to the size an ffmpeg scale filter would produce (even short side)."""
    F = float(getattr(cfg, "frames_src_fps", 0) or 0)
    L = int(getattr(cfg, "frames_long_side", 0) or 0)
    if F <= 0 and L <= 0:
        return {}
    opt = {"n_src": int(len(pts))}
    t = pts * tb
    if F > 0 and len(t) > 1 and np.median(np.diff(t)) < 1.0 / (F + 0.5):  # source fps is above the target
        ticks = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * F)) + 1) / F
        j = np.clip(np.searchsorted(t, ticks), 1, len(t) - 1)
        j = np.where(np.abs(t[j - 1] - ticks) <= np.abs(t[j] - ticks), j - 1, j)
        opt["cand"] = set(int(x) for x in pts[np.unique(j)])
    if L > 0:
        import av
        with av.open(str(video)) as c:
            st = c.streams.video[0]
            w, h = st.codec_context.width, st.codec_context.height
        if max(w, h) > L:
            if w >= h:
                W, H = L, int(round(h * L / w / 2)) * 2
            else:
                W, H = int(round(w * L / h / 2)) * 2, L
            opt["size"] = (W, H)
    return opt


def _key_before(key, i):
    """Decoder index of the nearest keyframe <= i (0 if none)."""
    k = np.flatnonzero(key[:i + 1])
    return int(k[-1]) if len(k) else 0


def select_fast(cfg, video, frames_dir, sel_dir):
    """Fast selection: frames are decoded frames (_probe), window fixed = duration/target, parallel pass over frame INDEX
    RANGES (not pts), gaps > max_gap are filled by targeted decoding, only selected frames are written as jpg.
    On videos where frames == packets with unique pts, the result is byte-identical to the pts-based selection.
    The pool is created BEFORE any decoding in the main process, and the probe pass (_probe) runs in a pool worker: a fork
    after decoding with libav threads (thread_type AUTO) in the main process deadlocked the workers (futex)."""
    from .. import cpu
    kmax = max(1, cpu.limit(cfg, cfg.frames_workers))
    pool = Pool(kmax)  # one pool per stage: fork only here, before decoding and before background model loading starts
    try:
        return _select_fast_pool(cfg, video, frames_dir, sel_dir, pool, kmax)
    finally:
        pool.close()
        pool.join()


def _select_fast_pool(cfg, video, frames_dir, sel_dir, pool, kmax):
    all_pts, key, tb = pool.apply(_probe, (str(video),))
    opt = _src_opts(cfg, video, all_pts, tb)
    if opt.get("cand") is not None:      # fps-grid frames (as after fps resampling): selection positions are a subset of decoder frames
        kept = np.array([i for i, x in enumerate(all_pts) if int(x) in opt["cand"]], dtype=np.int64)
        pos_of = {int(i): k for k, i in enumerate(kept)}
    else:
        kept = np.arange(len(all_pts), dtype=np.int64)
        pos_of = None
    pts = all_pts[kept]
    t = pts * tb
    out = Path(frames_dir); out.mkdir(parents=True, exist_ok=True)
    dur = float(t[-1] - t[0])
    target = cfg.target_frames
    fps_sel = float(getattr(cfg, "frames_per_s", 0) or 0)
    if fps_sel > 0:  # frame count from duration, clamped to [frames_min, frames_max]
        target = int(min(max(round(dur * fps_sel), getattr(cfg, "frames_min", 100)), getattr(cfg, "frames_max", 180)))
    win = max(dur / max(target, 1), 1e-3)
    win_pts = win / tb
    nwin = int(np.floor((pts[-1] - pts[0]) / win_pts)) + 1
    wpos = ((pts - pts[0]) // win_pts).astype(np.int64)          # window of each position: int((pts - t0) // win_pts)
    k = max(1, min(kmax, nwin))
    wins = [int(round(j * nwin / k)) for j in range(k)] + [10 ** 12]  # windows [wins[j], wins[j+1]) go to worker j
    wmono = np.maximum.accumulate(wpos)
    pb = [int(np.searchsorted(wmono, wins[j], side="left")) for j in range(k)] + [len(pts)]
    n_all = len(all_pts)
    jobs = []
    for j in range(k):
        if pb[j] >= pb[j + 1]:
            continue
        lo = int(kept[pb[j]]); hi = int(kept[pb[j + 1]]) if pb[j + 1] < len(kept) else n_all
        jobs.append((str(video), lo, hi, all_pts, _key_before(key, lo), pos_of, wpos, str(out), "win", None, opt))
    from . import _models
    _models.run_pending()
    res = pool.map(_decode_range, jobs)
    info = _select_fast_rest(cfg, video, pool, res, pts, t, out, win, win_pts, sel_dir, opt, all_pts, key, kept, pos_of)
    if opt:
        info["source_mode"] = dict(src_fps=getattr(cfg, "frames_src_fps", 0), size=opt.get("size"),
                                   n_source_frames=opt.get("n_src"))
    return info


def _select_fast_rest(cfg, video, pool, res, pts, t, out, win, win_pts, sel_dir, opt, all_pts, key, kept, pos_of):
    lapd = {}
    sel_idx = []   # selection positions (frame indices), not pts
    for l, s in res:
        lapd.update(l)
        sel_idx += s
    assert len(lapd) == len(pts), f"decoded {len(lapd)} frames != {len(pts)} frames in the probe pass"
    lap = np.array([lapd[i] for i in range(len(pts))])
    ok = lap > 0
    sel = sorted(int(i) for i in sel_idx)
    n_win = len(sel)
    sel = _fill_gaps(sel, t, lap, ok, cfg.max_gap_s)
    add = sorted(set(sel) - set(int(i) for i in sel_idx))
    if add:  # gap fill: separate decode of the needed frames (one segment per frame from its keyframe); want = positions
        want = [(str(video), int(kept[i]), int(kept[i]) + 1, all_pts, _key_before(key, int(kept[i])), pos_of, None, str(out),
                 "want", {int(i)}, opt or {}) for i in add]
        pool.map(_decode_range, want)
    gaps = np.diff(t[sel]) if len(sel) > 1 else np.zeros(1)
    if gaps.max() > cfg.max_gap_s:
        raise RuntimeError(f"time gap {gaps.max():.2f} s exceeds max_gap {cfg.max_gap_s}")
    paths = [out / f"{i + 1:06d}.jpg" for i in range(len(pts))]
    d = Path(sel_dir); d.mkdir(parents=True, exist_ok=True)
    for i in sel:
        dst = d / paths[i].name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(paths[i].resolve())
    thr = float(np.percentile(lap[ok], 5))
    risky = [int(i) for i in sel if lap[i] < thr]
    info = dict(window_s=round(win, 4), count=len(sel), max_gap_s=float(gaps.max()),
                lap_median=float(np.median(lap[sel])), lap_median_all=float(np.median(lap[ok])),
                risky_frames=len(risky), n_video_frames=int(len(pts)), frames_fast=True,
                frames=[dict(idx=int(i), file=paths[i].name, t=float(t[i]), lap=float(lap[i])) for i in sel])
    if len(sel) > n_win:
        info["gap_fill"] = dict(count_window=n_win, added=len(sel) - n_win)
    return info


def run(cfg):
    p = cfg.paths()
    if cfg.video and getattr(cfg, "frames_fast", False):
        info = select_fast(cfg, cfg.video, p["frames_all"], p["frames_sel"])
        (Path(cfg.work) / "frames_sel.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return info
    if cfg.video:
        paths, t = extract(cfg.video, p["frames_all"])
    else:
        paths = sorted(Path(cfg.frames).glob("*.jpg")) + sorted(Path(cfg.frames).glob("*.png"))
        t = np.arange(len(paths), dtype=float) / 30.0
    from .. import cpu
    info = select(paths, t, p["frames_sel"], cfg.target_frames, cfg.frames_lo,
                  cfg.frames_hi, cfg.max_gap_s, workers=cpu.limit(cfg, 32))
    (Path(cfg.work) / "frames_sel.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info
