"""Visibility map from a TSDF volume, on a synthetic volume (CPU)."""
import numpy as np
import torch

from tapeline.stages.visibility import free_map


def test_free_map_counts_empty_observed_voxels_per_cell():
    voxel = 0.02
    lo = np.array([1.0, -2.0, -0.25])
    dims = (9, 6, 200)                                  # odd X: the last column is dropped
    vol = torch.ones(dims)                              # tsdf = 1 - empty
    wacc = torch.zeros(dims)
    wacc[:4, :2, :] = 1.0                               # only 4x2 columns were observed
    vol[:4, :2, 100:] = -1.0                            # above 1.75 m - "occupied", not counted
    F = free_map(vol, wacc, lo, voxel, band=(0.5, 2.0))
    k0, k1 = int((0.5 + 0.25) / voxel), int((2.0 + 0.25) / voxel)   # 37 ... 112
    per_column = min(k1, 100) - k0                      # empty voxels per column: 100 - 37 = 63
    assert F.shape == (2, 3)                            # 4x2 columns -> 2 cells of 2x2
    assert np.allclose(F[:, 2], per_column)             # mean over the 4 columns of a cell = 63
    assert np.allclose(sorted(F[:, 0]), [lo[0] + 0.02, lo[0] + 0.06])
    assert np.allclose(F[:, 1], lo[1] + 0.02)


def test_free_map_empty_when_band_outside_volume():
    vol, wacc = torch.ones((4, 4, 10)), torch.ones((4, 4, 10))
    assert free_map(vol, wacc, np.zeros(3), 0.02, band=(0.5, 2.0)).shape == (0, 3)
