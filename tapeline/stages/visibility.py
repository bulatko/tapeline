"""Visibility map from the TSDF volume: a "free space" channel for bird's-eye-view wall/layout models.

A wall that nobody observed and a wall behind which empty space was observed are
different things for such a model: without this channel it tends to close contours around
unscanned areas (porches, wardrobes, "rooms inside rooms").

For each XY column we count observed EMPTY voxels (wacc > 0, tsdf > 0.5) in the height
band above the floor, sum them over a 2·voxel cell (4 cm for a 2 cm voxel) and divide
by 4 - the mean number of empty voxels per 2 cm column. Models that consume `free` must
be trained on maps built the same way.
"""
import numpy as np
import torch


def free_map(vol, wacc, lo, voxel, band=(0.5, 2.0)):
    """vol, wacc - TSDF volume and weights (X, Y, Z) on any device; lo - grid corner (m);
    -> (M, 3) float32: x, y, mean number of empty voxels per column of a 4 cm cell."""
    k0 = max(int((band[0] - lo[2]) / voxel), 0)
    k1 = min(int((band[1] - lo[2]) / voxel), vol.shape[2])
    if k1 <= k0:
        return np.zeros((0, 3), np.float32)
    free = ((wacc[:, :, k0:k1] > 0) & (vol[:, :, k0:k1] > 0.5)).sum(2).float()
    nx, ny = (free.shape[0] // 2) * 2, (free.shape[1] // 2) * 2       # 2x2-voxel cell; the tail of an odd-sized grid is dropped
    f2 = free[:nx, :ny].reshape(nx // 2, 2, ny // 2, 2).sum((1, 3))
    nz = torch.nonzero(f2 > 0)
    cell = 2 * voxel
    fx = (nz[:, 0].float() + 0.5) * cell + float(lo[0])
    fy = (nz[:, 1].float() + 0.5) * cell + float(lo[1])
    return torch.stack([fx, fy, f2[nz[:, 0], nz[:, 1]] / 4.0], 1).cpu().numpy().astype(np.float32)
