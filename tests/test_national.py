import numpy as np
import pandas as pd

from biomass_mapping.national import (
    S2_TOPO_LC,
    _balanced_fold_map,
    _block_ids,
    _national_model,
    _region,
)


def test_balanced_fold_map_is_deterministic_and_complete():
    counts = {"a": 100, "b": 90, "c": 80, "d": 70, "e": 60, "f": 50}
    first = _balanced_fold_map(counts, 3)
    second = _balanced_fold_map(counts, 3)
    assert first == second
    assert set(first) == set(counts)
    assert set(first.values()) == {0, 1, 2}


def test_metric_blocks_and_regions_are_stable():
    frame = pd.DataFrame({"longitude": [-5.0, -5.01], "latitude": [6.0, 6.01]})
    blocks = _block_ids(frame, 50_000)
    assert len(blocks) == 2
    assert blocks[0] == blocks[1]
    assert list(_region(np.array([6.49, 6.5, 8.49, 8.5])).astype(str)) == [
        "South (<6.5N)", "Centre (6.5-8.5N)", "Centre (6.5-8.5N)", "North (>=8.5N)",
    ]


def test_landcover_is_fit_as_categorical():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(80, len(S2_TOPO_LC))), columns=S2_TOPO_LC)
    frame["landcover"] = np.tile([10, 20, 30, 40], 20)
    target = frame.B2 * 2 + (frame.landcover == 20) * 3
    model = _national_model(S2_TOPO_LC, {"max_iter": 5, "min_samples_leaf": 5})
    prediction = model.fit(frame, target).predict(frame)
    assert np.isfinite(prediction).all()
    assert model.named_steps["model"].categorical_features is not None
