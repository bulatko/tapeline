"""CPU unit tests for stage helpers: floor plane fit, temporal gap filling, DA3 focal-length scale correction,
sparse/outlier frame pruning, CPU quota helpers."""
import types

import numpy as np
import pytest

from tapeline import cpu
from tapeline.stages.frames import _fill_gaps, _key_before
from tapeline.stages.scale import _focal_correction, _focal_fix_enabled
from tapeline.stages.sfm import pose_outliers, prune_sparse
from tapeline.stages.world import _fit_floor_plane


# ---------------------------------------------------------------- world: RANSAC floor plane
def test_floor_plane_recovers_tilted_plane_among_clutter():
    rng = np.random.default_rng(1)
    n_true = np.array([0.05, -0.03, 1.0]); n_true /= np.linalg.norm(n_true)
    u = np.cross(n_true, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n_true, u)
    floor = (rng.uniform(-3, 3, (4000, 1)) * u + rng.uniform(-3, 3, (4000, 1)) * v
             + rng.normal(0, 0.005, (4000, 1)) * n_true)
    clutter = rng.uniform([-3, -3, 0.2], [3, 3, 2.5], (2500, 3))      # furniture/walls above the floor
    n, inl = _fit_floor_plane(np.vstack([floor, clutter]))
    assert abs(abs(n @ n_true) - 1) < 1e-4
    assert 3900 <= inl <= 4100


def test_floor_plane_is_deterministic():
    P = np.random.default_rng(2).normal(size=(500, 3)) * [3, 3, 0.01]
    a, b = _fit_floor_plane(P), _fit_floor_plane(P)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


# ---------------------------------------------------------------- frames: gap filling, keyframe lookup
def test_fill_gaps_closes_gap_with_sharpest_frame():
    t = np.arange(0, 10.0, 0.1)
    score = np.ones_like(t); score[42] = 5.0                       # sharpest frame inside the gap
    ok = np.ones_like(t, bool)
    sel = _fill_gaps([0, 30, 60, 99], t, score, ok, max_gap=2.0)
    assert 42 in sel                                               # gap 30..60: one insertion within [t_b - 2, t_a + 2]
    assert np.all(np.diff(t[sel]) <= 2.0 + 1e-9)


def test_fill_gaps_leaves_real_hole():
    t = np.arange(0, 10.0, 0.1)
    ok = np.ones_like(t, bool); ok[31:60] = False                  # no valid frame in the gap
    sel = _fill_gaps([0, 30, 60, 99], t, np.ones_like(t), ok, max_gap=2.0)
    assert 30 in sel and 60 in sel and not any(31 <= i <= 59 for i in sel)


def test_key_before():
    key = np.array([1, 0, 0, 1, 0, 0, 0, 1], bool)
    assert [_key_before(key, i) for i in range(8)] == [0, 0, 0, 3, 3, 3, 3, 7]
    assert _key_before(np.zeros(4, bool), 2) == 0


# ---------------------------------------------------------------- scale: focal-length correction
def test_focal_correction_applies_median_ratio():
    frec = dict(error=None, f_pred=[1050.0, 1100.0, 1000.0], f_known=[1000.0, 1000.0, 1000.0])
    out = _focal_correction(2.0, frec, 3, enabled=True)
    assert out["focal_fix"] is True
    assert out["scale"] == pytest.approx(2.0 * np.median([1000 / 1050, 1000 / 1100, 1.0]))
    assert out["f_ratio_median"] == pytest.approx(1.05)


def test_focal_correction_disabled_or_missing_keeps_raw_scale():
    frec = dict(error=None, f_pred=[1050.0, 1100.0], f_known=[1000.0, 1000.0])
    assert _focal_correction(2.0, frec, 2, enabled=False)["scale"] == 2.0
    out = _focal_correction(2.0, frec, 3, enabled=True)               # view count mismatch
    assert out["scale"] == 2.0 and out["focal_fix"] is False and out["focal_note"]


def test_focal_fix_env_switch(monkeypatch):
    cfg = types.SimpleNamespace(scale_focal_fix=True)
    monkeypatch.delenv("TAPELINE_SCALE_FOCAL_FIX", raising=False)
    assert _focal_fix_enabled(cfg) is True
    monkeypatch.setenv("TAPELINE_SCALE_FOCAL_FIX", "0")
    assert _focal_fix_enabled(cfg) is False


# ---------------------------------------------------------------- sfm: sparse frames and pose outliers
class _P2:
    def __init__(self, has):
        self.has = has

    def has_point3D(self):
        return self.has


class _Im:
    def __init__(self, image_id, n_obs, centre=(0.0, 0.0, 0.0)):
        self.image_id, self.name = image_id, f"{image_id:06d}.jpg"
        self.points2D = [_P2(True) for _ in range(n_obs)] + [_P2(False)]
        self.c = np.asarray(centre, float)

    def projection_center(self):
        return self.c


class _Rec:
    def __init__(self, ims):
        self.images = {im.image_id: im for im in ims}

    def delete_observation(self, image_id, i):
        self.images[image_id].points2D[i].has = False


def test_prune_sparse_zero_obs_flag():
    def rec():
        return _Rec([_Im(1, 100), _Im(2, 10), _Im(3, 0)])
    r = rec()
    assert prune_sparse(r, min_obs=50) == ["000002.jpg"]               # default: only 0 < n < min_obs
    assert not any(p.has for p in r.images[2].points2D)
    assert prune_sparse(rec(), min_obs=50, zero=True) == ["000002.jpg", "000003.jpg"]


def test_pose_outliers_flags_exploded_centre_and_respects_skip():
    ims = [_Im(i, 20, (i * 0.1, 0.0, 0.0)) for i in range(1, 10)] + [_Im(10, 20, (1e6, 0, 0))]
    r = _Rec(ims)
    out = pose_outliers(r, k=10.0)
    assert [n for n, _ in out] == ["000010.jpg"]
    assert not any(p.has for p in r.images[10].points2D)
    assert pose_outliers(_Rec(ims[:-1] + [_Im(10, 20, (1e6, 0, 0))]), k=10.0, skip={"000010.jpg"}) == []


# ---------------------------------------------------------------- cpu quota helpers
def test_cpu_quota_and_env(monkeypatch):
    assert cpu.quota() >= 1
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    cpu.set_env(7)
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3" and os.environ["MKL_NUM_THREADS"] == "7"
