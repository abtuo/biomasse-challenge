import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from biomass_mapping.temporal import (
    add_gedi_acquisition_fields,
    build_temporal_common_sample,
    delta_time_to_datetime,
    folds_for_shots,
    sample_yearly_sensor,
    temporal_window_bounds,
    yearly_raster_lookup,
)
from biomass_mapping.temporal_2022 import biomass_metric_rows, paired_error_summary


def _point(x, y):
    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(x, y)
    return lon, lat


def _write_s1(path, vv=1.0):
    values = np.array([[[vv]], [[vv + 1]], [[-1.0]]], dtype="float32")
    profile = {"driver": "GTiff", "width": 1, "height": 1, "count": 3, "dtype": "float32",
               "crs": "EPSG:32631", "transform": from_origin(499995, 5, 10, 10)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values)
        for i, name in enumerate(["VV", "VH", "VV_minus_VH"], 1):
            dst.set_band_description(i, name)


def test_delta_time_to_datetime_and_calendar_fields():
    raw = pd.DataFrame({"delta_time": [0, 86400, -1, np.nan]})
    dated = add_gedi_acquisition_fields(raw)
    assert str(dated.loc[0, "gedi_datetime"]) == "2018-01-01 00:00:00+00:00"
    assert str(dated.loc[1, "gedi_datetime"]) == "2018-01-02 00:00:00+00:00"
    assert dated.loc[1, "gedi_year"] == 2018
    assert dated.loc[1, "gedi_doy"] == 2
    assert dated.gedi_datetime.isna().sum() == 2
    assert delta_time_to_datetime(pd.Series(["bad"])).isna().all()


def test_window_assignment_and_empty_datetimes():
    dt = pd.Series(pd.to_datetime(["2022-07-01", None], utc=True))
    bounds = temporal_window_bounds(dt, 183)
    assert (bounds.loc[0, "window_end"] - bounds.loc[0, "window_start"]).days == 366
    assert pd.isna(bounds.loc[1, "window_start"])


def test_yearly_lookup_missing_and_same_row_preservation(tmp_path):
    _write_s1(tmp_path / "s1_2020.tif", 4)
    lon, lat = _point(500000, 0)
    points = pd.DataFrame({"shot_number": [8, 9], "longitude": [lon, lon], "latitude": [lat, lat], "gedi_year": [2020, 2021]}, index=[10, 20])
    values, availability = sample_yearly_sensor(points, tmp_path, "S1")
    assert values.index.tolist() == [10, 20]
    assert values.loc[10, "VV"] == 4
    assert values.loc[20].isna().all()
    assert not bool(availability.set_index("year").loc[2021, "available"])
    assert yearly_raster_lookup(tmp_path, "S1", 2021) is None
    with pytest.raises(FileNotFoundError):
        sample_yearly_sensor(points, tmp_path, "S1", missing_year="raise")


def test_common_sample_matches_shots_and_drops_incomplete(tmp_path):
    from biomass_mapping.temporal import ALL_FEATURES
    base = pd.DataFrame({"shot_number": [1, 2, 3], "longitude": [0., 0., 0.], "latitude": [0., 0., 0.], "agbd": [1., 2., 3.]})
    for feature in ALL_FEATURES:
        base[feature] = 1.0
    same = base.iloc[[2, 0, 1]].copy()
    same.loc[same.shot_number == 2, ALL_FEATURES[0]] = np.nan
    p1, p2 = tmp_path / "base.parquet", tmp_path / "same.parquet"
    base.to_parquet(p1, index=False); same.to_parquet(p2, index=False)
    common, retention = build_temporal_common_sample({"MULTIYEAR_BASELINE": p1, "SAME_YEAR": p2}, tmp_path / "common.parquet")
    assert common.shot_number.tolist() == [1, 3]
    assert len(common) == 2
    assert retention.set_index("strategy").loc["SAME_YEAR", "available_rows"] == 2


def test_fold_reuse_is_by_shot_identity(tmp_path):
    fold_path = tmp_path / "folds.parquet"
    pd.DataFrame({"shot_number": np.arange(10), "fold": np.tile(np.arange(5), 2), "spatial_block": [str(i) for i in range(10)]}).to_parquet(fold_path, index=False)
    shots = pd.Series([6, 1, 8, 3, 0, 5, 2, 7, 4, 9])
    assigned, status = folds_for_shots(shots, fold_path)
    assert assigned.tolist() == [1, 1, 3, 3, 0, 0, 2, 2, 4, 4]
    assert status.consistent.iloc[0]


def test_filtering_2022_is_exact():
    seconds_2022 = (pd.Timestamp("2022-06-01", tz="UTC") - pd.Timestamp("2018-01-01", tz="UTC")).total_seconds()
    seconds_2021 = (pd.Timestamp("2021-12-31", tz="UTC") - pd.Timestamp("2018-01-01", tz="UTC")).total_seconds()
    dated = add_gedi_acquisition_fields(pd.DataFrame({"delta_time": [seconds_2021, seconds_2022]}))
    assert dated.loc[dated.gedi_year.eq(2022)].index.tolist() == [1]


def test_paired_metrics_use_same_rows_and_sign():
    y = pd.Series([100., 200., 400., 500.])
    multi = np.array([80., 170., 300., 350.])
    same = np.array([90., 180., 350., 450.])
    blocks = pd.Series(["a", "a", "b", "b"])
    result = paired_error_summary(y, multi, same, blocks, bootstrap_iterations=100).iloc[0]
    assert result.mean_abs_error_difference_same_minus_multi < 0
    assert result.fraction_improved_sameyear == 1.0
    high = pd.DataFrame(biomass_metric_rows(y, same, "same")).set_index("biomass_regime")
    assert high.loc["ge_370", "n"] == 2
    assert high.loc["ge_370", "bias"] == -50.0
