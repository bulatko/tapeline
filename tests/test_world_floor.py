"""The floor is the mode of the lowest significant peak, not its lower edge."""
import numpy as np

from tapeline.stages.world import floor_height


def test_floor_is_peak_not_first_bin():
    rng = np.random.default_rng(0)
    floor = rng.normal(0.0, 0.03, 20000)                 # floor with 3 cm triangulation noise
    walls = rng.uniform(0.05, 2.6, 60000)                # walls/furniture, uniform
    ceiling = rng.normal(2.7, 0.03, 15000)
    z = np.concatenate([floor, walls, ceiling])
    f, first = floor_height(z)
    assert abs(f) < 0.015, f                             # mode within one bin
    assert first < f - 0.03                              # first significant bin is clearly below the floor


def test_floor_ignores_denser_furniture_peak():
    # the table is denser than the floor (higher peak), but the floor is the lowest significant peak; the 0.15·max
    # significance threshold is inherent to the rule: a floor sparser than 15 % of the densest surface is not detected
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.normal(0.0, 0.03, 20000), rng.normal(0.75, 0.02, 30000), rng.normal(2.6, 0.02, 8000)])
    f, _ = floor_height(z)
    assert abs(f) < 0.015, f
