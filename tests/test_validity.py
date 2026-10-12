import numpy as np

from biomass_mapping.validity import KNNDomainValidity


def test_domain_validity_scores_and_flags_extrapolation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    obj = KNNDomainValidity(n_neighbors=5, max_reference_samples=100).fit(X)
    scores = obj.score_samples(np.vstack([np.zeros(3), np.full(3, 20.)]))
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    assert obj.in_domain(np.full((1, 3), 20.)).item() is False
