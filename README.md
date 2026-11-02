# Tapeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23897699.svg)](https://doi.org/10.5281/zenodo.23897699)

Camera poses and room reconstruction from a video.

Tapeline turns a handheld indoor video (phone or tablet) into metric camera poses, a coloured point
cloud of the room in a gravity-aligned frame with the floor at `z = 0`, and a 2D visibility map
for downstream wall/layout models. It can also start from frames with known poses (for example
ARKit) and only densify them.

```
frames ─► VGGT-Omega pose prior ─► SuperPoint + LightGlue (+ DINOv2 retrieval)
       ─► GPU triangulation + bundle adjustment ─► Depth Anything 3 posed depth
       ─► metric scale ─► per-frame scale ─► world (floor at 0)
       ─► TSDF fusion + visibility map ─► export
```

Every stage writes its artefacts into a working directory and is checked by a stage gate;
every run writes a provenance manifest.

## Installation

Requirements: Linux, Python ≥ 3.10, an NVIDIA GPU with a CUDA build of PyTorch, `ffmpeg` on `PATH`.
Bundle adjustment on ~600 frames needs a large GPU (an 80 GB card is the tested configuration,
see [Limitations](#limitations)).

```bash
# 1. PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
# 2. model code (installed separately so that it matches your torch/CUDA build)
pip install git+https://github.com/cvg/LightGlue
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3
git clone https://github.com/facebookresearch/vggt-omega && pip install -e vggt-omega
# 3. Tapeline
git clone https://github.com/bulatko/tapeline && pip install -e "tapeline[mesh,test]"
```

`pycolmap` is pinned to `4.0.4`: the GPU bundle-adjustment bridge reads the memory layout of
pycolmap's C++ structures directly for fast export/import; on a different build it silently falls
back to a much slower per-element path.

Weights:

* **VGGT-Omega** - request access at <https://huggingface.co/facebook/VGGT-Omega> and download
  `vggt_omega_1b_512.pt`; pass it with `--vggt-ckpt` or `TAPELINE_VGGT_CKPT`.
* **Depth Anything 3** (`depth-anything/DA3NESTED-GIANT-LARGE-1.1`), **SuperPoint/LightGlue** and
  **DINOv2** (`dinov2_vitl14` via `torch.hub`) are downloaded automatically on first use.

## Quick start

Video -> point cloud:

```bash
tapeline run --video room.mp4 --work runs/room --vggt-ckpt /path/to/vggt_omega_1b_512.pt
```

Frames with your own poses (no pose prior, poses are refined by bundle adjustment unless `--set refine=false`):

```bash
tapeline run --frames frames/ --poses poses.jsonl --pose-convention c2w --pose-axes gl \
             --poses-metric --work runs/room
```

`--poses` accepts a COLMAP model directory, an `.npz` (`extrinsics`, `image_names`, optional
`intrinsics`) or JSON lines (`name`, `transform`/`pose`, optional `intrinsics`). The convention
(`w2c`/`c2w`) and axes (`cv`/`gl`) are never guessed and must be given explicitly.

Other commands and options:

```bash
tapeline plan  --video room.mp4 --work runs/room      # stages and artefact paths
tapeline gates --video room.mp4 --work runs/room      # recorded stage gates
tapeline run ... --set max_keypoints=2048 --set frames_max=300   # override any Config field
tapeline run ... --strict          # stop at the first red gate (default: record it and continue)
tapeline run ... --export          # also write a mesh
```

From Python:

```python
from tapeline import Pipeline, Config
Pipeline(Config(video="room.mp4", work="runs/room", vggt_ckpt="vggt_omega_1b_512.pt")).run()
```

## Stages

| stage | what it does | main artefacts |
|---|---|---|
| `frames` | decodes the video (parallel decode; frame index and time come from decoded frames, so variable frame rate, duplicated timestamps and pre-roll packets are handled) and selects sharp frames evenly in time, filling temporal gaps | `scan/images/`, `frames_sel.json` |
| `prior` | VGGT-Omega pose and intrinsics prior for the selected frames | `prior.npz` |
| `match` | SuperPoint keypoints, LightGlue matches on sequential pairs plus DINOv2 retrieval pairs, geometric verification into a COLMAP database | `sfm/database.db`, `match.json` |
| `sfm` | GPU triangulation and multi-round bundle adjustment on top of the prior; frames with too few observations or outlying camera centres get their pose from the prior chain between well-observed neighbours | `sfm/model/`, `sfm.json` |
| `depth` | Depth Anything 3 in posed mode on the refined poses | `depth.npz` |
| `scale` | metric scale of the reconstruction from DA3 metric depth against triangulated depth (with a focal-length correction) | manifest |
| `frame_scale` | per-frame depth scale against SfM points; frames whose scale deviates too much are excluded from fusion | `frame_scale.json` |
| `fuse` | gravity-aligned world (floor at `z = 0`), TSDF fusion of confident depth pixels, point cloud extraction, visibility map | `cloud.ply`, `cloud_wm1.ply`, `wallfield.npz` |
| `export` | optional mesh export | `mesh.obj` |

Fusion options: the confidence cut-off `conf_min` discards depth pixels whose DA3 confidence is below the threshold (they
are not treated as observations, neither of surface nor of free space); the per-frame depth scale `frame_scale` rescales each
frame's depth by its own scale before fusion; protected carving `carve_protect` (off by default) treats surfaces seen
by at least k frames as opaque when carving free space (mirrors, glass). `cloud_wm1.ply` is a second extraction of the same
TSDF volume with a lower weight threshold (denser, noisier).

### Visibility map (`wallfield.npz`)

Written by `fuse` when `free_map=True` (default). Coordinates are metres in the world frame
(`z` up, floor at `z = 0`).

| key | dtype, shape | meaning |
|---|---|---|
| `xyz` | float16, (N, 3) | the fused point cloud (same points as `cloud.ply`) |
| `quads` | float32, (0, 4, 3) | reserved for wall quads, empty |
| `free` | float16, (M, 3) | `x, y, f`: centre of a `2·voxel` XY cell and the mean number of observed *free* voxels per column of that cell inside the height band `free_band` (default 0.5–2.0 m above the floor) |

A voxel counts as observed free space when it has non-zero TSDF weight and a TSDF value above 0.5.
The map distinguishes "nobody looked here" from "somebody looked here and saw empty space", which
wall/layout models need to avoid closing contours around unscanned areas.

### Stage gates and the manifest

Each stage is followed by a gate that checks a necessary condition on its output; a red gate raises
`RedGate` in the library (the CLI records it and continues unless `--strict`):

* `poses` (after `sfm`): registered fraction ≥ 0.70, median reprojection error <= 2.5 px, one connected component;
* `depth_convention` (after `depth`): DA3 depth versus triangulated depth - median ratio within 1 +/- 0.10 and bounded spread;
* `frame_scale`: enough frames with a per-frame scale and median log-scale near zero;
* `cloud` (after `fuse`): at least 100 000 points and few points below the floor plane.

`manifest.json` accumulates, per stage, its result and wall time, all gate results, the full
configuration and the versions of Python, Tapeline, torch, numpy, pycolmap and OpenCV.
`run_summary.json` (CLI) lists stage times, gates and red gates.

## Configuration

All parameters live in `tapeline/config.py` (`Config` dataclass) and can be set with `--set key=value`.
The defaults are the production configuration: ~600 frames per video (`frames_min = frames_max = 600`),
maximum temporal gap 1.5 s, 4096 SuperPoint keypoints, retriangulation in rounds 0–3, up to
100 Levenberg–Marquardt iterations per bundle-adjustment round, TSDF voxel 2 cm.

Runtime helpers (on by default): the CPU quota of the container (cgroup) sizes thread pools and
OpenMP/MKL (`--no-cpu-quota` to disable), models are prefetched in the background while CPU stages
run (`--no-prefetch`), the match database is built on a RAM disk (`db_local`, `db_local_dir`).
Prefetched models stay on the GPU, so the depth stage has slightly less free memory and may split
its frames into more chunks; results then differ within run-to-run noise. `--no-prefetch --no-cpu-quota`
sets the process up like a plain library call (`Pipeline(cfg).run()`).

## Limitations

* **Frame count and GPU memory.** By default every video is sampled to ~600 frames regardless of its
  duration. For short videos of small rooms this means very dense, highly overlapping frames, and
  bundle adjustment can run out of GPU memory even on an 80 GB card. Lower `frames_min`/`frames_max`
  (e.g. `--set frames_min=200 --set frames_max=200`) for short captures or smaller GPUs.
* **Metric scale** comes from monocular depth (Depth Anything 3); expect a scale error of about
  2–3 % unless poses are already metric (`--poses-metric`).
* Designed for indoor scenes captured as a continuous video; unordered photo collections need
  `--set pairs_exhaustive=true`.
* Linux only (RAM-disk default, `fork`-based frame decoding); one GPU per run.
* **Results are not bit-for-bit reproducible between runs.** Frame selection, the pose prior and the
  match database are deterministic, but GPU bundle adjustment is not (atomic accumulation on the GPU):
  on the same input, camera centres can differ by centimetres between runs, and everything downstream
  (depth, scale, fusion) inherits that.
* **DA3 depth chunking depends on free GPU memory.** The depth stage runs all frames in one chunk and
  halves the chunk on out-of-memory; how many chunks are used therefore depends on what else occupies
  the GPU, and the resulting depth (and point cloud) differs slightly between chunkings.
* **The default runtime mode is numerically different from a plain single-process call.** Background
  model prefetch and process-wide thread settings (`--no-prefetch --no-cpu-quota` disable them) change
  floating-point evaluation order and GPU memory headroom; the differences are of the same kind as the
  run-to-run variation above.

## Benchmarks

Coming soon.

## Models, weights and licenses

Tapeline's code is released under the [Apache License 2.0](LICENSE). It does not vendor third-party
code; models and libraries are installed separately and keep their own licenses. **The default
pipeline uses weights released for non-commercial use only** (VGGT-Omega, DA3-Giant/Nested,
SuperPoint); commercial use requires replacing them.

| component | used for | code license | weights license |
|---|---|---|---|
| [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) (`vggt_omega_1b_512.pt`, [HF](https://huggingface.co/facebook/VGGT-Omega), gated) | pose prior | FAIR Noncommercial Research License | FAIR Noncommercial Research License |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) ([DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)) | depth, scale | Apache-2.0 | CC BY-NC 4.0 (Giant/Nested/Large; Base, Small and Metric models are Apache-2.0) |
| [LightGlue](https://github.com/cvg/LightGlue) | matching | Apache-2.0 | Apache-2.0 |
| [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) (via LightGlue) | keypoints | [non-commercial research license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) | non-commercial research license |
| [DINOv2](https://github.com/facebookresearch/dinov2) (`dinov2_vitl14`) | image retrieval | Apache-2.0 | Apache-2.0 |
| [pycolmap / COLMAP](https://github.com/colmap/colmap) | database, triangulation | BSD-3-Clause | - |
| [PyAV](https://github.com/PyAV-Org/PyAV) | video decoding | BSD-3-Clause (links FFmpeg, LGPL/GPL depending on the build) | - |
| [PyTorch](https://pytorch.org), NumPy, SciPy, OpenCV, threadpoolctl, [Open3D](https://www.open3d.org) (optional) | runtime | BSD / Apache-2.0 / MIT | - |

## Citation

If you use Tapeline, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{gabdullin_tapeline,
  author  = {Gabdullin, Bulat},
  title   = {Tapeline: camera poses and room reconstruction from video},
  version = {0.1.0},
  year    = {2026},
  doi     = {10.5281/zenodo.23897699},
  url     = {https://github.com/bulatko/tapeline},
  license = {Apache-2.0}
}
```

Please also cite the models Tapeline builds on: VGGT-Omega (arXiv:2605.15195), Depth Anything 3,
SuperPoint, LightGlue, DINOv2 and COLMAP.

## License

Apache License 2.0, see [LICENSE](LICENSE).
