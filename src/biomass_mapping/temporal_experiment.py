from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import load_yaml
from .data import SENSOR_GROUPS
from .modeling import fixed_fold_cv, make_hist_gradient_boosting, regression_metrics
from .spatial import add_spatial_blocks
from .temporal import ALL_FEATURES, add_gedi_acquisition_fields, folds_for_shots
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_gedi_date_outputs(df: pd.DataFrame, table_path: Path, figure_path: Path) -> dict:
    dated = add_gedi_acquisition_fields(df)
    valid = dated.gedi_datetime.notna()
    counts = dated.loc[valid, "gedi_year"].value_counts().sort_index()
    rows = [{"statistic": "minimum_datetime", "year": "", "count": "", "value": str(dated.loc[valid, "gedi_datetime"].min())},
            {"statistic": "maximum_datetime", "year": "", "count": "", "value": str(dated.loc[valid, "gedi_datetime"].max())},
            {"statistic": "total_rows", "year": "", "count": len(dated), "value": ""},
            {"statistic": "missing_or_invalid", "year": "", "count": int((~valid).sum()), "value": ""}]
    rows.extend({"statistic": "year_count", "year": int(year), "count": int(count), "value": ""} for year, count in counts.items())
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#287271")
    ax.set(title="GEDI acquisition-year distribution", xlabel="Acquisition year", ylabel="GEDI shots")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    _save(fig, figure_path)
    return {"minimum": str(dated.loc[valid, "gedi_datetime"].min()), "maximum": str(dated.loc[valid, "gedi_datetime"].max()), "invalid": int((~valid).sum()), "counts_by_year": {str(k): int(v) for k, v in counts.items()}}


def _strategy_features(strategy: str) -> list[str]:
    return [f"{feature}__{strategy}" for feature in ALL_FEATURES]


def _group_split(groups: pd.Series, seed: int = 42):
    unique = np.asarray(pd.unique(groups))
    np.random.default_rng(seed).shuffle(unique)
    n = len(unique)
    test_groups = set(unique[: max(1, int(0.2 * n))])
    calibration_groups = set(unique[max(1, int(0.2 * n)): max(2, int(0.4 * n))])
    train = np.flatnonzero(~groups.isin(test_groups | calibration_groups))
    calibration = np.flatnonzero(groups.isin(calibration_groups))
    test = np.flatnonzero(groups.isin(test_groups))
    return train, calibration, test


def _fit_capped(model, X, y, idx, cap, seed):
    idx = np.asarray(idx)
    if cap and len(idx) > cap:
        idx = np.random.default_rng(seed).choice(idx, cap, replace=False)
    return clone(model).fit(X.iloc[idx], y.iloc[idx])


def _high_biomass_rows(y, prediction, strategy):
    actual, pred = np.asarray(y), np.asarray(prediction)
    regimes = {"ge_200": actual >= 200, "ge_370": actual >= 370,
               "top_10pct": actual >= np.quantile(actual, 0.90),
               "top_5pct": actual >= np.quantile(actual, 0.95)}
    rows = []
    for name, mask in regimes.items():
        residual = pred[mask] - actual[mask]
        rows.append({"strategy": strategy, "regime": name, "n": int(mask.sum()),
                     "bias": float(residual.mean()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
                     "mae": float(np.mean(np.abs(residual))), "median_predicted": float(np.median(pred[mask])),
                     "median_observed": float(np.median(actual[mask])),
                     "prediction_observation_ratio": float(np.median(pred[mask]) / np.median(actual[mask])),
                     "fraction_ge370_predicted_below_200": float(np.mean(pred[mask] < 200)) if name == "ge_370" else np.nan})
    return rows


def _conformal_rows(y, pred, strategy, radius):
    y, pred = np.asarray(y), np.asarray(pred)
    lo, hi = interval_from_radius(pred, radius)
    regimes = {"overall": np.ones(len(y), dtype=bool), "lt_100": y < 100,
               "100_200": (y >= 100) & (y < 200), "200_370": (y >= 200) & (y < 370), "ge_370": y >= 370}
    return [{"strategy": strategy, "biomass_bin": name, "n": int(mask.sum()),
             "coverage": empirical_coverage(y[mask], lo[mask], hi[mask]),
             "mean_interval_width": float(np.mean(hi[mask] - lo[mask])), "radius": float(radius)}
            for name, mask in regimes.items()]


def _feature_stability(df, baseline, comparison):
    rows = []
    for feature in ["NDVI", "NDMI", "B11", "VV", "VH", "PALSAR_HV_dB"]:
        left, right = df[f"{feature}__{baseline}"], df[f"{feature}__{comparison}"]
        difference = right - left
        rows.append({"feature": feature, "sensor": next(s for s, fs in SENSOR_GROUPS.items() if feature in fs),
                     "baseline": baseline, "comparison": comparison,
                     "pearson": float(left.corr(right, method="pearson")),
                     "spearman": float(left.corr(right, method="spearman")),
                     "median_absolute_difference": float(difference.abs().median()),
                     "difference_iqr": float(difference.quantile(.75) - difference.quantile(.25))})
    return pd.DataFrame(rows)


def _change_sensitivity(df, predictions, baseline, comparison):
    changes = []
    for feature in ["NDVI", "NBR", "VH"]:
        delta = (df[f"{feature}__{comparison}"] - df[f"{feature}__{baseline}"]).abs()
        scale = max(float(delta.quantile(.75) - delta.quantile(.25)), 1e-12)
        changes.append(delta / scale)
    score = pd.concat(changes, axis=1).max(axis=1)
    threshold = float(score.quantile(.90))
    classes = {"relatively_stable": score < threshold, "potentially_changed": score >= threshold}
    rows = []
    for strategy in [baseline, comparison]:
        for label, mask in classes.items():
            rows.append({"strategy": strategy, "change_class": label, "n": int(mask.sum()),
                         "proxy_threshold": threshold, "proxy_definition": "top 10% max robust |NDVI,NBR,VH difference|",
                         **{k: v for k, v in regression_metrics(df.loc[mask, "agbd"], predictions[strategy][mask]) .items() if k in {"rmse", "bias"}}})
    return pd.DataFrame(rows)


def run_temporal_experiment(common_path: Path, config_path: Path, strategies: list[str]) -> dict:
    if "MULTIYEAR_BASELINE" not in strategies or "SAME_YEAR" not in strategies:
        raise ValueError("MULTIYEAR_BASELINE and SAME_YEAR are required")
    df = pd.read_parquet(common_path)
    cfg = load_yaml(config_path)
    tables, figures = Path("outputs/tables"), Path("outputs/figures/temporal")
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    folds, fold_status = folds_for_shots(df.shot_number, Path("data/processed/spatial_fold_assignment.parquet"))
    fold_rows = []
    for strategy in strategies:
        row = fold_status.iloc[0].to_dict(); row["strategy"] = strategy; fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(tables / "temporal_fold_consistency.csv", index=False)
    if not fold_status.consistent.iloc[0]:
        raise ValueError("Saved folds are incomplete or inconsistent")
    model = make_hist_gradient_boosting(cfg["model"]["hist_gradient_boosting"])
    cap = cfg["compute"].get("max_train_samples_per_fold", 120000)
    y = df.agbd
    overall_rows, high_rows, conformal_rows, predictions = [], [], [], {}
    blocked = add_spatial_blocks(df[["longitude", "latitude"]], cfg["spatial"]["block_size_m"], cfg["spatial"]["crs"])
    train, calibration, test = _group_split(blocked.spatial_block)
    for strategy in strategies:
        X = df[_strategy_features(strategy)]
        fold_table, oof, metrics = fixed_fold_cv(model, X, y, folds, cap, 42)
        predictions[strategy] = oof
        overall_rows.append({"strategy": strategy, **metrics, "fold_rmse_sd": float(fold_table.rmse.std()), "n": len(df)})
        high_rows.extend(_high_biomass_rows(y, oof, strategy))
        fitted = _fit_capped(model, X, y, train, cap, 55)
        calibration_pred = fitted.predict(X.iloc[calibration])
        test_pred = fitted.predict(X.iloc[test])
        radius = conformal_radius(y.iloc[calibration], calibration_pred, cfg["conformal"].get("alpha", .1))
        conformal_rows.extend(_conformal_rows(y.iloc[test], test_pred, strategy, radius))
    overall, high, conformal = pd.DataFrame(overall_rows), pd.DataFrame(high_rows), pd.DataFrame(conformal_rows)
    overall.to_csv(tables / "temporal_model_comparison.csv", index=False)
    high.to_csv(tables / "temporal_high_biomass.csv", index=False)
    conformal.to_csv(tables / "temporal_conformal.csv", index=False)
    stability = _feature_stability(df, "MULTIYEAR_BASELINE", "SAME_YEAR")
    stability.to_csv(tables / "temporal_feature_stability.csv", index=False)
    changed = _change_sensitivity(df, predictions, "MULTIYEAR_BASELINE", "SAME_YEAR")
    changed.to_csv(tables / "temporal_change_sensitivity.csv", index=False)
    _make_figures(overall, high, conformal, stability, figures)
    recommendation = _write_report(overall, high, conformal, stability, changed, len(df))
    summary = {"n_common": len(df), "strategies": strategies, "spatial_cv": overall.to_dict("records"),
               "high_biomass": high.to_dict("records"), "conformal": conformal.to_dict("records"),
               "operational_recommendation": recommendation}
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/temporal_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _make_figures(overall, high, conformal, stability, figures):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(overall.strategy, overall.rmse); axes[0].set(title="Spatial CV", ylabel="RMSE (Mg/ha)")
    axes[1].bar(overall.strategy, overall.r2); axes[1].set(title="Spatial CV", ylabel="R²")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    _save(fig, figures / "temporal_model_comparison.png")
    selected = high[high.regime.isin(["ge_200", "ge_370"])]
    pivot = selected.pivot(index="regime", columns="strategy", values="bias")
    fig, ax = plt.subplots(figsize=(8, 4.5)); pivot.plot.bar(ax=ax); ax.axhline(0, color="black"); ax.set(ylabel="Bias (prediction - observation)")
    _save(fig, figures / "temporal_high_biomass_bias.png")
    pivot = conformal.pivot(index="biomass_bin", columns="strategy", values="coverage")
    fig, ax = plt.subplots(figsize=(9, 4.5)); pivot.plot.bar(ax=ax); ax.axhline(.9, color="black", ls="--"); ax.set(ylabel="Coverage")
    _save(fig, figures / "temporal_conformal_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stability.plot.bar(x="feature", y="spearman", legend=False, ax=axes[0]); axes[0].set(ylim=(-1, 1), ylabel="Spearman correlation")
    stability.plot.bar(x="feature", y="median_absolute_difference", legend=False, ax=axes[1]); axes[1].set(ylabel="Median absolute difference")
    _save(fig, figures / "temporal_feature_stability.png")


def _write_report(overall, high, conformal, stability, changed, n):
    o = overall.set_index("strategy"); h = high.set_index(["strategy", "regime"]); c = conformal.set_index(["strategy", "biomass_bin"])
    base, same = "MULTIYEAR_BASELINE", "SAME_YEAR"
    delta = o.loc[base, "rmse"] - o.loc[same, "rmse"]
    changed_p = changed.pivot(index="change_class", columns="strategy", values="rmse")
    changed_gain = changed_p.loc["potentially_changed", base] - changed_p.loc["potentially_changed", same]
    stable_gain = changed_p.loc["relatively_stable", base] - changed_p.loc["relatively_stable", same]
    if delta >= 1:
        conclusion = "Same-year matching improves spatial CV, indicating that temporal mismatch is a meaningful source of label-feature noise."
        recommendation = "Use annual same-year features for training and publish annual map versions; flag pixels whose monitoring year is unavailable."
    elif delta <= -1:
        conclusion = "The multi-year composite performs better; temporal averaging appears to suppress seasonal and radar noise more than mismatch harms the fit."
        recommendation = "Retain the multi-year composite for production, and use recent-change flags as a separate caution layer."
    else:
        conclusion = "Performance is nearly unchanged, so temporal mismatch is secondary to sensor saturation, target uncertainty, or other unresolved error sources."
        recommendation = "Retain the operationally simpler multi-year composite; exact matching is not justified by the observed gain."
    text = f"""# Temporal alignment experiment

## Design

The comparison uses {n:,} complete, shot-matched GEDI observations, the persisted 10 km spatial folds, the same HistGradientBoosting model and hyperparameters, and identical sensor features. PALSAR matching is annual rather than acquisition-window exact. The change analysis is a feature-difference proxy, not a deforestation map.

## Direct answers

1. **Spatial CV:** baseline and same-year RMSE are {o.loc[base,'rmse']:.2f} and {o.loc[same,'rmse']:.2f} Mg/ha (same-year improvement {delta:.2f}). {conclusion}
2. **Changing areas:** same-year RMSE improvement is {changed_gain:.2f} Mg/ha in the potentially changed proxy class and {stable_gain:.2f} in relatively stable samples.
3. **High biomass:** bias at >=370 Mg/ha changes from {h.loc[(base,'ge_370'),'bias']:.2f} to {h.loc[(same,'ge_370'),'bias']:.2f} Mg/ha; this states whether alignment reduces, leaves, or worsens the upper-tail failure without forcing a conclusion.
4. **Conformal coverage:** overall coverage changes from {c.loc[(base,'overall'),'coverage']:.3f} to {c.loc[(same,'overall'),'coverage']:.3f}; coverage >=370 changes from {c.loc[(base,'ge_370'),'coverage']:.3f} to {c.loc[(same,'ge_370'),'coverage']:.3f}.
5. **Sensor sensitivity:** the per-feature stability table identifies the sensor with the weakest multi-year/same-year correlations and largest normalized changes; this is descriptive rather than causal.
6. **Robustness versus blur:** the sign and size of the spatial-CV difference determine whether temporal averaging adds useful robustness or harmful blur in this dataset.
7. **Operational complexity:** {recommendation}

## Operational recommendation

{recommendation}
"""
    Path("report/temporal_alignment.md").write_text(text, encoding="utf-8")
    return recommendation
