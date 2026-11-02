"""Command-line entry point.

    tapeline run --video room.mp4 --work runs/room --vggt-ckpt vggt_omega_1b_512.pt
    tapeline run --frames frames/ --poses poses.txt --work runs/room     # your own poses
    tapeline plan --video room.mp4 --work runs/room                       # print stages and artefact paths
    tapeline gates --video room.mp4 --work runs/room                      # print recorded stage gates

`run` executes the stages one by one; a red stage gate is
recorded (manifest.json, run_summary.json) and the run continues; `--strict` stops at the first red
gate instead. Any Config field can be overridden with `--set key=value` (value parsed as JSON when
possible). The runtime helpers are on by default: the CPU quota of the container is applied to OpenMP/MKL
before numpy/torch are imported, and model weights are prefetched in the background while CPU
stages run. Prefetched models stay on the GPU, so the depth stage has slightly less memory and may
split its frames into more chunks, which changes the result within run-to-run noise; use
`--no-prefetch --no-cpu-quota` for runs that must match a plain library call.
"""
import argparse
import json
import os
import sys
import time

DEFAULT_STAGES = ["frames", "prior", "match", "sfm", "depth", "scale", "frame_scale", "fuse"]
GATE_OF = {"sfm": "poses", "depth": "depth_convention", "frame_scale": "frame_scale", "fuse": "cloud"}


def _parse_overrides(items):
    over = {}
    for kv in items or []:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    return over


def _parser():
    ap = argparse.ArgumentParser(prog="tapeline", description="Camera poses and room reconstruction from video.")
    ap.add_argument("command", choices=["run", "plan", "gates"])
    ap.add_argument("--video", help="input video file")
    ap.add_argument("--frames", help="directory with frames (instead of --video)")
    ap.add_argument("--poses", help="known camera poses for --frames (switches the prior to 'external')")
    ap.add_argument("--work", default="runs/scene", help="working directory for all artefacts")
    ap.add_argument("--vggt-ckpt", default=os.environ.get("TAPELINE_VGGT_CKPT"),
                    help="VGGT-Omega checkpoint (default: $TAPELINE_VGGT_CKPT)")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="subset of stages to run (default: %s)" % " ".join(DEFAULT_STAGES))
    ap.add_argument("--export", action="store_true", help="also run the export stage (meshes / USD)")
    ap.add_argument("--pose-convention", default=None, choices=["w2c", "c2w"])
    ap.add_argument("--pose-axes", default=None, choices=["cv", "gl"])
    ap.add_argument("--poses-metric", action="store_true", help="external poses are already metric")
    ap.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                    help="override any Config field (repeatable), e.g. --set max_keypoints=2048")
    ap.add_argument("--strict", action="store_true", help="stop at the first red stage gate")
    ap.add_argument("--force", action="store_true", help="recompute stages whose artefacts already exist")
    ap.add_argument("--no-prefetch", action="store_true", help="load models on demand instead of in the background")
    ap.add_argument("--no-cpu-quota", action="store_true",
                    help="do not apply the container CPU quota to process-wide thread pools (OpenMP/MKL, torch, "
                         "OpenCV, COLMAP); stage worker pools still follow the cpu_quota config field")
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    over = _parse_overrides(a.overrides)
    if a.no_prefetch:
        over["prefetch"] = False
    for k, v in (("pose_convention", a.pose_convention), ("pose_axes", a.pose_axes)):
        if v is not None:
            over[k] = v
    if a.poses_metric:
        over["poses_metric"] = True

    # The CPU quota must reach OpenMP/MKL before numpy/torch are imported; cpu.py only imports os.
    from . import cpu as _cpu
    cpu_n = None
    proc_quota = not a.no_cpu_quota and over.get("cpu_quota", True)
    if proc_quota:
        cpu_n = _cpu.quota()
        _cpu.set_env(cpu_n)

    from .config import Config
    from .runner import Pipeline, RedGate

    kw = dict(video=a.video, frames=a.frames, work=a.work, **over)
    if a.poses:
        kw.update(poses=a.poses, prior="external")
    if a.vggt_ckpt:
        kw["vggt_ckpt"] = a.vggt_ckpt
    cfg = Config(**kw)
    stages = a.stages or (DEFAULT_STAGES + (["export"] if a.export else []))

    if a.command == "plan":
        cfg.validate()
        print(json.dumps({"stages": stages, "paths": {k: str(v) for k, v in cfg.paths().items()}}, indent=1))
        return 0
    if a.command == "gates":
        print(json.dumps(Pipeline(cfg).manifest.get("gates", {}), ensure_ascii=False, indent=1))
        return 0

    if cfg.prior != "external" and not cfg.vggt_ckpt:
        raise SystemExit("a VGGT-Omega checkpoint is required: --vggt-ckpt or $TAPELINE_VGGT_CKPT")
    print("OVERRIDES:", json.dumps(over, ensure_ascii=False), flush=True)
    if proc_quota and getattr(cfg, "cpu_quota", False):
        from .sfm_gpu import trigpu as _tg
        n = cpu_n or _cpu.quota()
        _tg.NUM_THREADS = min(16, n)
        print("CPU_QUOTA", json.dumps({**_cpu.apply(n), "colmap": _tg.NUM_THREADS,
                                       "omp_env": os.environ.get("OMP_NUM_THREADS")}), flush=True)
    p = Pipeline(cfg)
    t0 = time.time()
    from .stages import _models  # background model loading, deferred until the fork pools of 'frames' are done
    _models.start_prefetch(cfg)
    summary = dict(stages_s={}, gates={}, red_gates=[], overrides=over)
    for st in stages:
        t1 = time.time()
        try:
            r = p.stage(st, force=a.force)
            print(f"STAGE {st} {time.time() - t0:.0f}s", json.dumps(r, ensure_ascii=False, default=str)[:600], flush=True)
            if st == "frames" and isinstance(r, dict):
                summary["frames"] = r.get("count")
        except RedGate as e:
            summary["red_gates"].append(st)
            if a.strict:
                print(f"STAGE {st} {time.time() - t0:.0f}s RED GATE (stopping): {e}"[:800], flush=True)
                break
            print(f"STAGE {st} {time.time() - t0:.0f}s RED GATE (continuing): {e}"[:800], flush=True)
        _models.after_stage(cfg, st)
        summary["stages_s"][st] = round(time.time() - t1, 1)
        g = p.manifest["gates"].get(GATE_OF.get(st, ""), None)
        if g is not None:
            print(f"GATE {st}", json.dumps(g, ensure_ascii=False), flush=True)
            summary["gates"][st] = g
    summary["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(cfg.work, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)
    return 1 if (a.strict and summary["red_gates"]) else 0


if __name__ == "__main__":
    sys.exit(main())
