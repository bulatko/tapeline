"""Initial poses: VGGT-Omega in one batch, or external poses from a file.

VGGT takes the whole set in a single pass - chunking breaks the anchoring to the first frame,
while memory grows only linearly with the frame count. Chunking is enabled only as a
fallback response to OOM.
"""
from pathlib import Path
import numpy as np

from .. import io as bio


def load_vggt(cfg):
    """Load VGGT-Omega (same as in run) - for the cache / background preloading."""
    import torch
    from vggt_omega.models.vggt_omega import VGGTOmega
    if getattr(cfg, "fast_init", False):  # random init directly on the GPU (much faster); the loaded weights are the same
        with torch.device("cuda"):
            model = VGGTOmega()
        model = model.to("cuda").eval()  # buffers created explicitly on the CPU (_resnet_mean/std) are moved to the GPU, as in the regular path
    else:
        model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(cfg.vggt_ckpt, map_location="cpu"))
    return model


def run(cfg):
    p = cfg.paths()
    out = p["prior"]
    names = sorted(x.name for x in Path(p["frames_sel"]).iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))

    if cfg.prior == "external":
        ext = bio.read_poses(cfg.poses, cfg.pose_format, cfg.pose_convention,
                             cfg.pose_axes, cfg.pose_matrix_order)
        keep = [i for i, n in enumerate(ext["names"]) if n in set(names)]
        if len(keep) < len(names):
            missing = len(names) - len(keep)
            raise RuntimeError(f"no external poses for {missing} selected frames")
        E = ext["extrinsics"][keep]
        K = ext["intrinsics"][keep] if ext["intrinsics"] is not None else None
        np.savez(out, extrinsics=E.astype(np.float32), image_names=np.array([ext["names"][i] for i in keep]),
                 intrinsics=(K.astype(np.float32) if K is not None else np.zeros(0)),
                 source="external", is_metric=int(cfg.poses_metric))
        return dict(source="external", n=len(keep), metric=cfg.poses_metric)

    import torch
    from vggt_omega.models.vggt_omega import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    paths = [str(Path(p["frames_sel"]) / n) for n in names]
    if getattr(cfg, "model_cache", False) or getattr(cfg, "prefetch", False):
        from . import _models
        model = _models.get("vggt", lambda: load_vggt(cfg))
        _models.drop("vggt")  # VGGT is not needed afterwards: we hold the only reference, it is freed when the stage exits
    elif getattr(cfg, "fast_init", False):
        model = load_vggt(cfg)
    else:
        model = VGGTOmega().to("cuda").eval()
        model.load_state_dict(torch.load(cfg.vggt_ckpt, map_location="cpu"))
    torch.cuda.empty_cache()
    nth = int(getattr(cfg, "prior_io_threads", 0) or 0)
    if nth > 1:
        from .. import cpu
        nth = cpu.limit(cfg, nth)
    if nth > 1 and len(paths) > nth:  # read+resize frames in chunks on threads (frames of one video share one size)
        from concurrent.futures import ThreadPoolExecutor
        ch = [paths[k * len(paths) // nth:(k + 1) * len(paths) // nth] for k in range(nth)]
        with ThreadPoolExecutor(nth) as ex:
            parts = list(ex.map(lambda c: load_and_preprocess_images(c, mode="balanced", image_resolution=cfg.vggt_res), ch))
        assert len({tuple(x.shape[1:]) for x in parts}) == 1, "frames have different sizes - threaded loading is not applicable"
        images = torch.cat(parts, 0)
    else:
        images = load_and_preprocess_images(paths, mode="balanced", image_resolution=cfg.vggt_res)

    # Retrieval descriptors are taken from this same pass and are therefore free.
    # The patch_embed output is used because it is per-frame: cross-frame attention starts
    # at block 0 of the aggregator and by layer 4 makes frames nearly indistinguishable
    # (similarities collapse to ~1), so its tokens are useless for retrieval.
    # max over patches rather than mean: it yields more verified pairs.
    desc = {}

    def _grab(mod, inp, out):
        x = out["x_norm_patchtokens"] if isinstance(out, dict) else out
        desc["d"] = x.amax(1).float().cpu().numpy()   # reduce before the cast: otherwise the peak memory grows noticeably

    hook = model.aggregator.patch_embed.register_forward_hook(_grab)
    try:
        from ._models import tf32
        with torch.inference_mode(), tf32(cfg):
            pred = model(images.to("cuda", non_blocking=True))
    finally:
        hook.remove()
    ext34, K = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    E = torch.eye(4, device=ext34.device).expand(*ext34.shape[:-2], 4, 4).clone()
    E[..., :3, :] = ext34
    E = E[0].float().cpu().numpy()
    K = K[0].float().cpu().numpy()
    np.savez(out, extrinsics=E.astype(np.float32), intrinsics=K.astype(np.float32),
             image_names=np.array(names), source="vggt", is_metric=0,
             descriptors=desc.get("d", np.zeros(0, np.float32)),
             peak_gb=float(torch.cuda.max_memory_allocated() / 1e9))
    return dict(source="vggt", n=len(names), descriptors=int(np.size(desc.get("d", 0)) > 0),
                peak_gb=float(torch.cuda.max_memory_allocated() / 1e9))
