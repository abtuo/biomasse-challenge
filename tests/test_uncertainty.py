import numpy as np

from biomass_mapping.uncertainty import (
    conformal_radius,
    empirical_coverage,
    interval_from_radius,
    split_conformal_evaluate,
)


def test_conformal_interval():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    p = np.array([1, 2, 2, 5, 4], dtype=float)
    r = conformal_radius(y, p, alpha=0.2)
    lo, hi = interval_from_radius(p, r)
    cov = empirical_coverage(y, lo, hi)
    assert r >= 0
    assert 0 <= cov <= 1

def test_split_conformal_uses_separate_arrays():
    result, lo, hi = split_conformal_evaluate([1, 2, 3], [1, 2, 2], [4, 5], [4, 4], alpha=.1)
    assert result["radius"] >= 0
    assert len(lo) == len(hi) == 2
