import numpy as np
import pandas as pd
import pytest

from biomass_mapping.field_learning import (
    BASE,
    F3,
    create_field_cv_assignments,
    fit_predict_method,
    select_training_plots,
)


def _field(n=60):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "plot_id": [f"p{i}" for i in range(n)],
        "latitude": np.linspace(4.5, 10, n),
        "longitude": np.linspace(-8, -3, n) + rng.normal(0, .15, n),
        "field_agbd": rng.uniform(10, 250, n),
        BASE: rng.uniform(20, 180, n),
        "national_domain_distance": rng.uniform(0, 2, n),
        "national_interval_width": rng.uniform(100, 180, n),
        "region": np.where(np.linspace(4.5, 10, n) < 6.5, "SOUTH",
                           np.where(np.linspace(4.5, 10, n) < 8.5, "CENTRE", "NORTH")),
        "ecological_stratum": "forest-like",
    })
    for feature in F3:
        frame[feature] = rng.normal(size=n)
    return frame


def test_field_split_integrity_and_group_preservation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = _field()
    assignments = create_field_cv_assignments(frame)
    assert assignments.plot_id.is_unique
    assert assignments.random_fold.between(0, 4).all()
    assert assignments.groupby("spatial50_block").spatial50_fold.nunique().max() == 1
    for fold in assignments.spatial50_fold.unique():
        train = set(assignments.loc[assignments.spatial50_fold != fold, "plot_id"])
        test = set(assignments.loc[assignments.spatial50_fold == fold, "plot_id"])
        assert train.isdisjoint(test)


def test_leave_region_out_holds_out_exactly_one_region(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assignments = create_field_cv_assignments(_field())
    for fold, region in enumerate(["SOUTH", "CENTRE", "NORTH"]):
        held_out = assignments.loc[assignments.leave_region_out_fold == fold]
        training = assignments.loc[assignments.leave_region_out_fold != fold]
        assert set(held_out.region) == {region}
        assert region not in set(training.region)
        assert set(held_out.plot_id).isdisjoint(training.plot_id)


def test_model_rejects_plot_leakage():
    frame = _field(20)
    with pytest.raises(ValueError, match="leakage"):
        fit_predict_method("GEDI_RESIDUAL_CORRECTION", "F3", frame.iloc[:15], frame.iloc[10:])


def test_residual_correction_is_finite_and_held_out():
    frame = _field(40)
    frame["field_agbd"] = frame[BASE] + 25 + frame.B11 * 3
    prediction = fit_predict_method(
        "GEDI_RESIDUAL_CORRECTION", "F3", frame.iloc[:30], frame.iloc[30:]
    )
    assert len(prediction) == 10
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("strategy", [
    "RANDOM", "SPATIALLY_BALANCED", "DOMAIN_GUIDED",
    "PREDICTED_BIOMASS_STRATIFIED", "HYBRID",
])
def test_strategic_sampling_is_reproducible_and_target_blind(strategy):
    frame = _field(50)
    first = select_training_plots(frame, 12, strategy, 42)
    modified = frame.copy()
    modified["field_agbd"] = modified.field_agbd.sample(frac=1, random_state=9).to_numpy()
    second = select_training_plots(modified, 12, strategy, 42)
    assert len(first) == 12
    assert len(np.unique(first)) == 12
    assert np.array_equal(first, second)


def test_sample_size_is_capped_without_duplicates():
    frame = _field(18)
    selected = select_training_plots(frame, 50, "RANDOM", 1)
    assert np.array_equal(selected, np.arange(18))


def test_repeated_sample_draws_are_reproducible():
    frame = _field(50)
    first_run = [select_training_plots(frame, 20, "RANDOM", seed) for seed in range(5)]
    second_run = [select_training_plots(frame, 20, "RANDOM", seed) for seed in range(5)]
    assert all(np.array_equal(first, second) for first, second in zip(first_run, second_run))
    assert any(not np.array_equal(first_run[0], draw) for draw in first_run[1:])
