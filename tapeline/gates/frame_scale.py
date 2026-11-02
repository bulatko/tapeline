"""Per-frame depth scale gate: per-frame scale from SfM points.

Red if the scale was estimated on fewer than half of the frames (too few points - no
reference, and the stage does nothing useful) or if the median log-scale is far from zero:
DA3 and SfM diverged globally - the same convention problem depth_convention catches, only
after alignment. The per-frame spread itself (abs_dev) is a measurement, not a criterion.
"""


def frame_scale_gate(res, min_coverage=0.5, max_offset=0.10):
    cov = float(res.get("coverage", 0.0))
    off = float(res.get("log_s_median", 0.0))
    ok = cov >= min_coverage and abs(off) <= max_offset
    return dict(name="frame_scale", ok=bool(ok), coverage=round(cov, 3), log_s_median=round(off, 4),
                abs_dev_p50=res.get("abs_dev_p50"), abs_dev_p90=res.get("abs_dev_p90"),
                pts_per_frame_median=res.get("pts_per_frame_median"),
                thresholds=dict(coverage=min_coverage, offset=max_offset))
