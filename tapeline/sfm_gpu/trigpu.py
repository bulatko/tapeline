"""Triangulation with a GPU part.

Most of pycolmap.triangulate_points time is spent not on per-track DLT but on the
inner IterativeGlobalRefinement loop -- a global BA of POINTS only
(poses and intrinsics fixed) plus complete/merge/filter, all on CPU via ceres.
The ba_use_gpu flag has no effect there: ceres is built without CUDA.

This module reproduces that loop one-to-one, but the point BA runs on the GPU:
   pycolmap.triangulate_points(..., ba_global_max_refinements=0)   # tracks only, no loop
   repeat ba_global_max_refinements times:
       point BA on GPU (same setup: fixed poses, fixed intrinsics, trivial loss)
       complete_all_tracks + merge_all_tracks         (C++, sequential -- left as is)
       filter_all_points3D(thr, min_angle)
       stop if the fraction of changed observations < ba_global_max_refinement_change
What changes is WHERE the point BA is computed, not WHAT is computed.
"""
import time
import numpy as np
import torch
import pycolmap
from . import bacore as B
from . import gpuba as GB


NUM_THREADS = 16


def pipe_opts(thr, min_ang, **kw):
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = NUM_THREADS  # default 16; with a CPU quota use min(16, quota)
    o.triangulation.min_angle = min_ang
    o.triangulation.create_max_angle_error = max(2.0, thr / 2)
    o.triangulation.continue_max_angle_error = max(2.0, thr / 2)
    o.triangulation.merge_max_reproj_error = thr
    o.triangulation.complete_max_reproj_error = thr
    o.triangulation.ignore_two_view_tracks = False
    o.mapper.filter_max_reproj_error = thr
    o.mapper.filter_min_tri_angle = min_ang
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def tri_opts(o):
    """IncrementalTriangulatorOptions taken from the pipeline options (same thresholds)."""
    return o.triangulation


class TriGPU:
    def __init__(self, db_path, img_path, work_dir, device="cuda"):
        self.db_path, self.img_path, self.work = db_path, img_path, work_dir
        self.device = device
        t0 = time.time()
        self._db = pycolmap.Database.open(db_path)
        self._cache = pycolmap.DatabaseCache.create(self._db, pycolmap.DatabaseCacheOptions())
        self.graph = self._cache.correspondence_graph
        self.t_cache = time.time() - t0

    def run(self, rec, thr, min_ang, max_refinements=None, verbose=True):
        ref_o = pycolmap.IncrementalPipelineOptions()
        n_ref = ref_o.ba_global_max_refinements if max_refinements is None else max_refinements
        eps = ref_o.ba_global_max_refinement_change
        T = dict(colmap=0.0, export=0.0, togpu=0.0, gpu_ba=0.0, imprt=0.0,
                 bookkeep=0.0, iters=0)

        t0 = time.time()
        rec = pycolmap.triangulate_points(rec, self.db_path, self.img_path, self.work,
                                          True, pipe_opts(thr, min_ang,
                                                          ba_global_max_refinements=0), False)
        T["colmap"] = time.time() - t0

        topt = tri_opts(pipe_opts(thr, min_ang))
        # PITFALL: two ObservationManagers on the same reconstruction crash in C++
        # ("num_tri_corrs <= num_total_corrs"). The triangulator and filtering share one instance.
        obsm = pycolmap.ObservationManager(rec, self.graph)
        tri = pycolmap.IncrementalTriangulator(self.graph, rec, obsm)
        fx = B.make_exporter(rec)         # memory layout is validated right here
        for i in range(int(n_ref)):
            n_obs = sum(1 for im in rec.images.values() for p in im.points2D if p.has_point3D()) \
                if i == 0 else n_obs_prev
            t0 = time.time()
            prob_np = fx.export(rec)
            prob = GB.Problem(prob_np["R"], prob_np["t"], prob_np["X"], prob_np["cam"],
                              prob_np["pt"], prob_np["fr"], prob_np["xy"],
                              device=self.device, build_pairs=False)
            torch.cuda.synchronize(); T["export"] += time.time() - t0
            t0 = time.time()
            res = GB.solve_points(prob, loss_scale=None, max_iter=50)
            torch.cuda.synchronize(); T["gpu_ba"] += time.time() - t0
            t0 = time.time()
            Xn = res["X"].cpu().numpy()
            # Fast write through pycolmap memory instead of a per-point pybind loop,
            # which is slow for large point counts.
            fx.write_points(rec, prob_np["pids"], Xn)
            T["imprt"] += time.time() - t0
            t0 = time.time()
            changed = tri.complete_all_tracks(topt) + tri.merge_all_tracks(topt)
            changed += obsm.filter_all_points3D(thr, min_ang)
            T["bookkeep"] += time.time() - t0
            T["iters"] += 1
            n_obs_prev = sum(1 for im in rec.images.values() for p in im.points2D if p.has_point3D())
            frac = changed / max(n_obs, 1)
            if verbose:
                print("   refine %d: gpu_ba_iters=%d changed=%d (%.5f) pts=%d"
                      % (i, res["iters"], changed, frac, len(rec.points3D)), flush=True)
            if frac < eps:
                break
        return rec, T
