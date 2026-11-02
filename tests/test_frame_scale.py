"""Per-frame depth scale estimator on synthetic data: no pycolmap, no GPU."""
import numpy as np

from tapeline.stages.frame_scale import estimate, frame_scale_from_obs, diagnostics
from tapeline.gates.frame_scale import frame_scale_gate


def _synth(seed=0, N=12, hd=40, wd=60, n_obs=300, noise=0.01):
    rng = np.random.default_rng(seed)
    s_true = np.exp(rng.normal(0, 0.02, N))                       # per-frame scale +/-2 %
    Z = rng.uniform(0.5, 5.0, (N, hd, wd))                        # "true" depth (SfM)
    D = Z / s_true[:, None, None] * np.exp(rng.normal(0, noise, (N, hd, wd)))   # DA3 = z/s with noise
    C = rng.uniform(2.0, 8.0, (N, hd, wd))
    D[C < 3.8] *= 3.0                                             # low conf - garbage (heavy tail)
    obs = []
    for i in range(N):
        xi = rng.integers(0, wd, n_obs)
        yi = rng.integers(0, hd, n_obs)
        obs.append((xi, yi, Z[i, yi, xi]))
    return s_true, D.astype(np.float32), C.astype(np.float32), obs


def test_recovers_per_frame_scale_with_conf_cut():
    s_true, D, C, obs = _synth()
    s, n, mad = estimate(D, C, obs, conf_min=3.8, min_pts=20)
    assert (n > 0).all()
    err = np.abs(np.log(s / s_true))
    assert np.percentile(err, 90) < 0.003, err


def test_without_conf_the_mad_clip_still_holds():
    s_true, D, C, obs = _synth()
    s, n, mad = estimate(D, None, obs, conf_min=0.0, min_pts=20)
    err = np.abs(np.log(s / s_true))
    assert np.percentile(err, 90) < 0.01, err          # 30 % garbage is removed by the MAD cut, accuracy is lower


def test_too_few_points_gives_unit_scale():
    s, n, mad = frame_scale_from_obs(np.ones(5), np.ones(5) * 1.02, None, 0.0, min_pts=20)
    assert s == 1.0 and n == 0


def test_diagnostics_and_gate():
    s_true, D, C, obs = _synth()
    obs[0] = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))    # frame without points
    s, n, mad = estimate(D, C, obs, conf_min=3.8)
    d = diagnostics(s, n, mad)
    assert d["frames_scaled"] == len(s) - 1 and abs(d["coverage"] - (len(s) - 1) / len(s)) < 1e-9
    assert frame_scale_gate(d)["ok"]
    assert not frame_scale_gate(dict(coverage=0.3, log_s_median=0.0))["ok"]
    assert not frame_scale_gate(dict(coverage=1.0, log_s_median=0.3))["ok"]
