import numpy as np
import pandas as pd

from biomass_mapping.field_validation import (
    interval_coverage_summary,
    nearest_gedi_matches,
    regression_summary,
    threshold_match_counts,
)


def _field():
    return pd.DataFrame({
        "plot_id": ["p1", "p2"],
        "measurement_date": ["2022-01-01", "2022-01-02"],
        "longitude": [0.0001, 0.01],
        "latitude": [0.0, 0.0],
        "field_agbd": [100.0, 200.0],
    })


def _gedi():
    return pd.DataFrame({
        "longitude": [0.0, 0.02], "latitude": [0.0, 0.0],
        "agbd": [90.0, 180.0], "agbd_se": [10.0, 12.0],
        "relative_se": [0.1, 0.07], "delta_time": [126230400.0, 126230500.0],
        "shot_number": [1, 2],
    })


def test_nearest_gedi_and_time_reconstruction():
    matched = nearest_gedi_matches(_field(), _gedi())
    assert matched.iloc[0].nearest_distance_m < 25
    assert matched.iloc[0].gedi_agbd == 90.0
    assert str(matched.iloc[0].gedi_acquisition_datetime).startswith("2022-01-01")


def test_threshold_counts_include_no_match_case():
    matched = nearest_gedi_matches(_field(), _gedi())
    counts = threshold_match_counts(matched, thresholds=(5, 25, 500))
    assert counts.set_index("threshold_m").loc[5, "n_matches"] == 0
    assert counts.set_index("threshold_m").loc[25, "n_matches"] == 1


def test_interval_coverage_and_empty_metrics():
    result = interval_coverage_summary([100, 200], [90, 210], [110, 220])
    assert result["n_covered"] == 1
    assert result["empirical_coverage"] == 0.5
    empty = regression_summary([], [])
    assert empty["n"] == 0
    assert np.isnan(empty["rmse"])
