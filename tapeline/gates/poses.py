"""Pose gate (offline pose protocol).

Coverage ≥ 0.70, median reprojection <= 2.5 px, exactly one connected component.
Components are counted over frames that CARRY observations: in the programmatic template
every frame has a pose, so a "by poses" metric would always be green.
"""


def poses_gate(model_dir, min_registered=0.70, max_reproj=2.5):
    import pycolmap   # lazy: the gates package is also imported without pycolmap (tests)
    from ..stages.sfm import stats
    rec = pycolmap.Reconstruction(str(model_dir))
    st = stats(rec)
    frac = st["frames_with_obs"] / max(st["frames"], 1)
    ok = (frac >= min_registered and st["median"] <= max_reproj and st["components"] == 1)
    return dict(name="poses", ok=bool(ok), registered_frac=round(frac, 4),
                median_px=round(st["median"], 4), p90_px=round(st["p90"], 4),
                components=st["components"], points=st["points"], observations=st["n_obs"],
                thresholds=dict(registered=min_registered, reproj=max_reproj, components=1))
