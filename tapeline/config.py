"""Run configuration.

Rule: anything that can be derived from the data is derived automatically and defaults to None.
Only the input, the work directory and the compute budget are set by hand - otherwise people
end up reading printed numbers by eye and copying them into a profile.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json


@dataclass
class Config:
    # --- input: either a video or ready-made frames (+ optionally known poses) ---
    video: Optional[str] = None
    frames: Optional[str] = None            # frame directory, if the video is already extracted
    poses: Optional[str] = None             # COLMAP sparse/ | .npz | .jsonl
    pose_format: Literal["auto", "colmap", "npz", "jsonl"] = "auto"
    pose_convention: Literal["w2c", "c2w"] = "w2c"
    pose_axes: Literal["cv", "gl"] = "cv"   # gl = ARKit-like (Y up, Z backward)
    pose_matrix_order: Literal["C", "F"] = "C"
    poses_metric: bool = False              # poses are already in meters (ARKit and similar)

    work: str = "runs/scene"

    # --- frame selection ---
    target_frames: int = 150  # frame target of the default frame selector
    frames_lo: int = 100  # lower bound of the frame count (default frame selector)
    frames_hi: int = 200  # upper bound of the frame count (default frame selector)
    max_gap_s: float = 1.5  # max time gap between consecutive selected frames, s
    # Fast frame path: PyAV decodes the video in segments across frames_workers processes, sharpness is
    # computed on the Y plane, and only the selected frames are written as jpg (window = duration/target_frames,
    # no window search); frames_all contains only those frames
    frames_fast: bool = True
    frames_workers: int = 32  # decoder processes of the fast frame path
    frames_src_fps: float = 0.0             # (frames_fast) select from the source video - candidates on a grid of this fps
    frames_long_side: int = 0               # (frames_fast) frames are downscaled to this long side (like normalize)
    frames_per_s: float = 0.6  # (frames_fast only) >0: target = clamp(duration·frames_per_s, frames_min, frames_max)
    frames_min: int = 600  # lower bound of the frame count (fast frame path)
    frames_max: int = 600  # upper bound of the frame count (fast frame path)
    # Load DA3 once per process (depth and scale take the model from the cache in stages/_models.py)
    model_cache: bool = True
    # Background loading of VGGT (at process start) and DA3 (after prior) while CPU stages run
    prefetch: bool = True
    # Write the match database in a single transaction (on a network filesystem every implicit transaction is an fsync)
    db_txn: bool = True
    # Build the VGGT/DA3 models directly on the GPU (no random initialization on the CPU)
    fast_init: bool = True
    tf32: bool = True  # TF32 during the VGGT/DA3 forward passes (fp32 parts outside autocast)
    cloud_wm1: bool = True  # fuse also writes a second cloud cloud_wm1.ply (weight_min=1, same TSDF)
    cpu_quota: bool = True  # size CPU-stage threads/processes by the cgroup quota (tapeline/cpu.py)
    db_local: bool = True  # the match database is written/verified in /dev/shm, then copied into work
    db_local_dir: Optional[str] = None  # where db_local builds the database (None: /dev/shm if present, else the system temp dir)
    prior_io_threads: int = 8  # >1: frames for VGGT are read in chunks by a thread pool
    match_io_threads: int = 8  # >0: jpgs for SuperPoint are read by a thread pool

    # --- initial poses ---
    prior: Literal["vggt", "external"] = "vggt"
    vggt_ckpt: Optional[str] = None
    vggt_res: int = 512

    # --- matching ---
    max_keypoints: int = 4096  # at 720p SuperPoint does not find more than this anyway
    seq_window: int = 8
    retrieval_knn: int = 10
    # Frame descriptor for retrieval. vggt is taken from the prior stage's forward pass and
    # is therefore free; pair quality is comparable, but dino also works with external poses.
    retrieval: Literal["dino", "vggt"] = "dino"
    pairs_exhaustive: bool = False    # all pairs i<j instead of "sequential | retrieval" - for UNORDERED image sets
                                      # (name order is not time, seq_window is meaningless); cheap for up to ~150 images
    min_match_score: float = 0.2
    match_budget: int = 120_000_000   # B*N^2 budget for the LightGlue batch size
    match_mp: bool = True             # fp16 in LightGlue: about 2x faster, same matches
    verify_backend: Literal["gpu", "colmap"] = "gpu"   # gpu verification is much faster than COLMAP
    verify_hypotheses: int = 512

    # --- pose refinement ---
    refine: bool = True                     # run GPU triangulation + BA on top of the prior
    verify_poses: bool = False              # only measure reprojection of external poses, do not change them
    camera_model: str = "OPENCV"            # the sfm_gpu core supports only this model
    focal_init: Optional[float] = None      # None -> from the prior intrinsics
    retri_rounds: str = "0,1,2,3"  # BA rounds after which points are re-triangulated
    ba_max_iter: int = 100  # cap on LM iterations in each BA round
    sfm_rounds: str = "0,1,2,3,4"           # which SCHEDULE rounds to run (default: all 5)
    sparse_zero_obs: bool = True  # prune_sparse also takes frames with 0 observations (-> refill_sparse); False: only 0 < n
    pose_outlier_guard: bool = True  # frame center after BA farther than k·R (R = median distance to the median center) -> sparse+refill
    pose_outlier_k: float = 10.0
    sfm_collapse_guard: bool = True  # re-triangulation lost >70% of points -> roll back to stage0 and retry
    sfm_hist_stats: bool = False  # False: intermediate rounds skip the full stats() (history only)

    # --- depth ---
    depth_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    depth_res: int = 504
    depth_chunk: Optional[int] = None       # None = single pass; a number = fallback chunking
    scale_model: Optional[str] = None       # DA3 model of the scale stage (None = depth_model)

    # --- scale ---
    scale: Optional[float] = None           # None -> estimate from DA3 (or 1.0 for metric poses)

    # --- fusion ---
    fuse: Literal["tsdf", "concat"] = "tsdf"
    voxel: float = 0.02
    trunc: float = 0.08                     # µ; must exceed the depth noise
    max_depth: float = 6.0
    weight_min: float = 2.0  # minimum accumulated TSDF weight for a voxel to be kept
    surface_band: float = 0.4               # |tsdf| band for surface extraction
    subvoxel: bool = True
    weights: Literal["none", "conf"] = "none"   # with many frames conf weighting makes no difference
    conf_percentile: float = 0.0            # >0 only makes sense for fuse="concat"
    extra_pairs: Optional[str] = None       # file of "name name [...]" lines - extra pairs for matching (bridges weak segments)
    min_frame_obs: int = 50                 # a frame with fewer 3D observations after BA drops out of the main map
                                            # (lower thresholds let poorly observed frames through with large pose errors;
                                            # a frame with very few observations can drift far while its reprojection looks fine);
                                            # 0 = no filtering. Frames without a single verified pair never enter
                                            # the template (template.images_with_pairs)
    sparse_refill: bool = True              # such frames get a VGGT-chain pose from their nearest well-observed neighbours
                                            # (full coverage at prior-level error instead of a BA outlier)
    frame_fps: float = 30.0                 # rate at which frame name NNNNNN.jpg = video frame index: detector time
                                            # (long-range links dt, effective_dt) is computed as N / frame_fps. Set it to the
                                            # frame rate the names refer to (e.g. 60 when names index a 60-fps video),
                                            # otherwise time-based thresholds are off by that factor
    conf_mode: Literal["frame", "global"] = "frame"
    box_pad: float = 0.6
    z_max: float = 3.6

    # --- fusion: confidence cut-off, per-frame depth scale, protected carving, visibility map ---
    conf_min: float = 3.8                   # confidence cut-off: DA3 pixels with lower confidence are not observations (0 = off)
    frame_scale: bool = True                # per-frame depth scale from SfM points
    frame_scale_min_pts: int = 20           # fewer points per frame - the frame keeps scale 1
    frame_scale_max_err: float = 2.0        # px: max reprojection error of a point in the model
    frame_scale_z_min: float = 0.3          # m: point range window (via the global scale)
    frame_scale_z_max: float = 6.0
    frame_scale_max_dev: float = 0.2        # |log s - median| above this - the frame is excluded from fusion (DA3 and SfM
                                            # disagree: mirror/glass/blur, which scaling cannot fix); 0 = never exclude
    carve_protect: int = 0                  # protected carving: a surface supported by ≥ k frames (|sdf| <= µ/2) is opaque - ray depth
                                            # is clipped at it (mirrors, glass, gloss); two TSDF passes. OFF by default:
                                            # it doubles the TSDF cost; useful when many rays pass through a mirror wall
    free_map: bool = True                   # visibility map + cloud -> wallfield.npz (xyz, quads, free)
    free_band: tuple = (0.5, 2.0)           # m above the floor: band in which empty voxels are counted

    # --- output ---
    export: tuple = ("ply",)                # ply | mesh | obj | usdz
    gpu: int = 0

    def paths(self) -> dict:
        w = Path(self.work)
        return {k: w / v for k, v in dict(
            frames_all="frames_all", frames_sel="scan/images", prior="prior.npz",
            db="sfm/database.db", sfm="sfm/model", depth="depth.npz",
            frame_scale="frame_scale.json", wallfield="wallfield.npz",
            cloud="cloud.ply", mesh="mesh.obj", manifest="manifest.json",
            gates="gates.json", viz="viz").items()}

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8")

    def validate(self):
        if not self.video and not self.frames:
            raise ValueError("video= or frames= is required")
        if self.prior == "external" and not self.poses:
            raise ValueError('prior="external" requires poses=')
        if self.poses and self.prior != "external":
            self.prior = "external"
        if self.trunc <= self.voxel:
            raise ValueError("trunc (µ) must be noticeably larger than the voxel")
        if self.conf_min < 0:
            raise ValueError("conf_min must be ≥ 0 (0 = no cut)")
        if self.free_band[0] >= self.free_band[1]:
            raise ValueError("free_band = (low, high) in meters above the floor")
        return self
