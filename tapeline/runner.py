"""Stage orchestration: order, skipping finished stages, gates, provenance manifest.

The manifest is always written: without it there is no way to answer later "which code
and which weights produced this" - /tmp gets cleaned, checkpoints move, and a timing
profile is misleading unless it records whether the cache was cold or warm.
"""
import json
import platform
import time
from pathlib import Path

from .config import Config

ORDER = ["frames", "prior", "match", "sfm", "depth", "scale", "frame_scale", "fuse", "export"]


class RedGate(RuntimeError):
    """Red oracle. Must not be swallowed by the generic except in _gates_for, otherwise the gate is mere reporting."""


def _versions():
    from . import __version__
    out = {"python": platform.python_version(), "tapeline": __version__}
    for mod in ("torch", "numpy", "pycolmap", "cv2"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg.validate()
        Path(cfg.work).mkdir(parents=True, exist_ok=True)
        self.manifest = {"config": cfg.__dict__, "versions": _versions(), "stages": {},
                         "gates": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.scale = cfg.scale if cfg.scale else 1.0
        # the manifest is cumulative: a partial run (--stages fuse) continues the previous one instead of wiping it -
        # otherwise the scale stage is lost and fusion silently runs in SfM units
        mp = Path(cfg.paths()["manifest"])
        if mp.exists():
            try:
                old = json.loads(mp.read_text(encoding="utf-8"))
                self.manifest["stages"] = old.get("stages", {})
                self.manifest["gates"] = old.get("gates", {})
                self.manifest["resumed_from"] = old.get("updated") or old.get("started")
                sc = old.get("stages", {}).get("scale", {})
                if cfg.scale is None and "scale" in sc:
                    self.scale = float(sc["scale"])
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------ stages
    def _skip(self, name):
        p = self.cfg.paths()
        done = {"frames": p["frames_sel"], "prior": p["prior"], "match": p["db"],
                "sfm": Path(p["sfm"]) / "cameras.bin", "depth": p["depth"],
                "frame_scale": p["frame_scale"], "fuse": p["cloud"]}
        t = done.get(name)
        return t is not None and Path(t).exists()

    def stage(self, name, force=False):
        if self._skip(name) and not force:
            self.manifest["stages"][name] = {"skipped": True}
            return self.manifest["stages"][name]
        cfg = self.cfg
        t0 = time.time()
        if name == "frames":
            from .stages import frames as m; res = m.run(cfg)
        elif name == "prior":
            from .stages import prior as m; res = m.run(cfg)
        elif name == "match":
            if cfg.prior == "external" and not (cfg.refine or cfg.verify_poses):
                res = {"skipped": "external poses without refinement - matches are not needed"}
            else:
                from .stages import match as m; res = m.run(cfg)
        elif name == "sfm":
            from .stages import sfm as m; res = m.run(cfg)
        elif name == "depth":
            from .stages import depth as m; res = m.run(cfg)
        elif name == "scale":
            from .stages import scale as m
            res = m.run(cfg); self.scale = res["scale"]
        elif name == "frame_scale":
            if not cfg.frame_scale:
                res = {"skipped": "frame_scale=False"}
            else:
                from .stages import frame_scale as m; res = m.run(cfg, self.scale)
        elif name == "fuse":
            from .stages import fuse as m; res = m.run(cfg, self.scale)
        elif name == "export":
            from .stages import export as m; res = m.run(cfg)
        else:
            raise ValueError(f"unknown stage {name}")
        res = dict(res or {}); res["wall_s"] = round(time.time() - t0, 2)
        self.manifest["stages"][name] = res
        self._gates_for(name)
        self._save()
        return res

    # ------------------------------------------------------------------ gates
    def _gates_for(self, name):
        from . import gates as G      # lazy: the package is also imported where pycolmap is absent (evaluator tests)
        p = self.cfg.paths()
        try:
            if name == "sfm" and self.cfg.refine:
                self._gate(G.poses_gate(p["sfm"]))
            elif name == "depth":
                self._gate(G.depth_convention_gate(p["sfm"], p["depth"]))
            elif name == "frame_scale" and self.cfg.frame_scale:
                self._gate(G.frame_scale_gate(self.manifest["stages"][name]))
            elif name == "fuse":
                self._gate(G.cloud_gate(p["cloud"]))
        except RedGate:
            raise                                    # a red gate stops the run - that is its whole purpose
        except Exception as e:                       # but an error inside the gate itself must not silently kill the run
            self.manifest["gates"][name] = {"error": str(e)}

    def _gate(self, res):
        self.manifest["gates"][res["name"]] = res
        if not res["ok"]:
            self._save()
            raise RedGate(f"RED GATE {res['name']}: {json.dumps(res, ensure_ascii=False)}")

    # ------------------------------------------------------------------ run
    def run(self, *only, force=False):
        for name in (only or ORDER):
            self.stage(name, force=force)
        self._save()
        return self.manifest

    def _save(self):
        self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # explicit encoding: under nohup/cron the locale may be ascii, and non-ASCII gate text would break the manifest write
        Path(self.cfg.paths()["manifest"]).write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
