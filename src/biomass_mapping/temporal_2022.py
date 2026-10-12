from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy.stats import wilcoxon
from sklearn.base import clone

from .config import load_yaml
from .data import SENSOR_GROUPS
from .modeling import fixed_fold_cv, make_hist_gradient_boosting
from .raster import sample_raster_blockwise
from .spatial import add_spatial_blocks
from .temporal import ALL_FEATURES, add_gedi_acquisition_fields, folds_for_shots
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity

STRATEGIES = ("MULTIYEAR_2020_2024", "SAME_YEAR_2022")
HIGHLIGHT_FEATURES = [
    "NDVI", "NDMI", "NDRE", "NBR", "B11", "B12", "VV", "VH",
    "PALSAR_HH_dB", "PALSAR_HV_dB",
]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _missing_fraction(src: rasterio.DatasetReader, sample_size: int = 512) -> float:
    """Estimate stack missingness on a regular, downsampled grid."""
    height = min(sample_size, src.height)
    width = min(sample_size, src.width)
    arr = src.read(
        out_shape=(src.count, height, width), masked=True, resampling=Resampling.nearest
    )
    values = np.ma.filled(arr, np.nan)
    return float(np.mean(~np.isfinite(values).all(axis=0)))


def audit_temporal_rasters(
    raster_dir: Path, selected_paths: dict[str, Path], output_path: Path
) -> pd.DataFrame:
    expected_by_name = {str(path.resolve()): SENSOR_GROUPS[sensor] for sensor, path in selected_paths.items()}
    rows = []
    for path in sorted(raster_dir.glob("*.tif")):
        with rasterio.open(path) as src:
            bands = [name or f"band_{i}" for i, name in enumerate(src.descriptions, 1)]
            expected = expected_by_name.get(str(path.resolve()))
            missing_bands = [band for band in expected or [] if band not in bands]
            extra_bands = [band for band in bands if expected is not None and band not in expected]
            rows.append({
                "filename": path.name,
                "selected_for_experiment": expected is not None,
                "crs": str(src.crs),
                "bounds_left": src.bounds.left,
                "bounds_bottom": src.bounds.bottom,
                "bounds_right": src.bounds.right,
                "bounds_top": src.bounds.top,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "band_names": "|".join(bands),
                "dtypes": "|".join(src.dtypes),
                "nodata": src.nodata,
                "estimated_missing_fraction": _missing_fraction(src),
                "missing_required_bands": "|".join(missing_bands),
                "extra_bands": "|".join(extra_bands),
                "schema_valid": expected is None or (not missing_bands and not extra_bands and bands == expected),
                "missingness_method": "regular 512x512 nearest-neighbour sample; any band invalid",
            })
    audit = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    selected = audit[audit.selected_for_experiment]
    if len(selected) != len(selected_paths):
        raise FileNotFoundError("Not all selected 2022 rasters were found in the audit")
    if not selected.schema_valid.all():
        bad = selected.loc[~selected.schema_valid, ["filename", "missing_required_bands", "extra_bands"]]
        raise ValueError(f"Selected 2022 raster schema failure:\n{bad.to_string(index=False)}")
    return audit


def summarize_gedi_2022(raw_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = add_gedi_acquisition_fields(pd.read_csv(raw_path))
    subset = raw.loc[raw.gedi_year.eq(2022)].copy()
    counts = subset.gedi_month.value_counts().sort_index()
    total = len(raw)
    summary = {
        "total_filtered_gedi": total,
        "gedi_2022_rows": len(subset),
        "percentage_of_total": 100 * len(subset) / total,
        "minimum_datetime": str(subset.gedi_datetime.min()),
        "maximum_datetime": str(subset.gedi_datetime.max()),
        "missing_or_invalid_dates": int(raw.gedi_datetime.isna().sum()),
    }
    rows = [{"statistic": key, "month": "", "count": value if "datetime" not in key else "", "value": value if "datetime" in key else ""} for key, value in summary.items()]
    rows.extend({"statistic": "month_count", "month": int(month), "count": int(count), "value": ""} for month, count in counts.items())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    summary["counts_by_month"] = {str(int(k)): int(v) for k, v in counts.items()}
    return subset, summary


def build_paired_2022_datasets(
    raw_2022: pd.DataFrame,
    multiyear_path: Path,
    raster_paths: dict[str, Path],
) -> dict:
    multiyear_all = pd.read_parquet(multiyear_path)
    if raw_2022.shot_number.duplicated().any() or multiyear_all.shot_number.duplicated().any():
        raise ValueError("shot_number must be unique in GEDI and multisensor inputs")
    date_fields = raw_2022.set_index("shot_number")[["gedi_datetime", "gedi_year", "gedi_month", "gedi_doy"]]
    multiyear = multiyear_all.loc[multiyear_all.shot_number.isin(date_fields.index)].copy().reset_index(drop=True)
    multiyear = multiyear.join(date_fields, on="shot_number", validate="one_to_one")
    if not multiyear.gedi_year.eq(2022).all():
        raise RuntimeError("Non-2022 GEDI row entered the temporal subset")
    multiyear_out = Path("data/interim/gedi_2022_multiyear.parquet")
    multiyear_out.parent.mkdir(parents=True, exist_ok=True)
    multiyear.to_parquet(multiyear_out, index=False)

    metadata = multiyear.drop(columns=ALL_FEATURES, errors="ignore")
    sampled = []
    for sensor in ("S2", "S1", "PALSAR"):
        sampled.append(sample_raster_blockwise(multiyear, raster_paths[sensor], SENSOR_GROUPS[sensor]))
    sameyear = pd.concat([metadata.reset_index(drop=True)] + [x.reset_index(drop=True) for x in sampled], axis=1)
    if not np.array_equal(multiyear.shot_number.to_numpy(), sameyear.shot_number.to_numpy()):
        raise RuntimeError("Raster extraction changed row identity or order")
    sameyear_out = Path("data/interim/gedi_2022_sameyear.parquet")
    sameyear.to_parquet(sameyear_out, index=False)

    finite_multi = multiyear[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    finite_same = sameyear[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    common_mask = finite_multi & finite_same
    multi_common = multiyear.loc[common_mask].reset_index(drop=True)
    same_common = sameyear.loc[common_mask].reset_index(drop=True)
    if not np.array_equal(multi_common.shot_number, same_common.shot_number):
        raise RuntimeError("Strict common-sample shot alignment failed")
    processed = Path("data/processed"); processed.mkdir(parents=True, exist_ok=True)
    multi_common.to_parquet(processed / "temporal_2022_multiyear_common.parquet", index=False)
    same_common.to_parquet(processed / "temporal_2022_sameyear_common.parquet", index=False)
    metadata_common = multi_common.drop(columns=ALL_FEATURES)
    wide = pd.concat([
        metadata_common,
        multi_common[ALL_FEATURES].add_suffix("__MULTIYEAR_2020_2024"),
        same_common[ALL_FEATURES].add_suffix("__SAME_YEAR_2022"),
    ], axis=1)
    wide.to_parquet(processed / "temporal_2022_common_sample.parquet", index=False)
    retention = {
        "raw_gedi_2022_rows": len(raw_2022),
        "multisensor_2022_rows": len(multiyear),
        "valid_multiyear_rows": int(finite_multi.sum()),
        "invalid_multiyear_rows": int((~finite_multi).sum()),
        "valid_sameyear_rows": int(finite_same.sum()),
        "invalid_sameyear_rows": int((~finite_same).sum()),
        "final_common_rows": len(multi_common),
        "retention_pct_of_multisensor_2022": 100 * len(multi_common) / len(multiyear),
        "raw_rows_absent_from_multisensor": len(raw_2022) - len(multiyear),
        "dropped_only_multiyear_invalid": int((~finite_multi & finite_same).sum()),
        "dropped_only_sameyear_invalid": int((finite_multi & ~finite_same).sum()),
        "dropped_both_invalid": int((~finite_multi & ~finite_same).sum()),
    }
    pd.DataFrame([retention]).to_csv("outputs/tables/temporal_2022_sample_retention.csv", index=False)
    return {"multiyear": multi_common, "sameyear": same_common, "retention": retention}


def _spatial_split(df: pd.DataFrame, cfg: dict, seed: int = 42):
    blocked = add_spatial_blocks(df[["longitude", "latitude"]], cfg["block_size_m"], cfg["crs"])
    groups = np.asarray(pd.unique(blocked.spatial_block))
    np.random.default_rng(seed).shuffle(groups)
    n = len(groups)
    test_groups = set(groups[:max(1, int(.2 * n))])
    cal_groups = set(groups[max(1, int(.2 * n)):max(2, int(.4 * n))])
    train = np.flatnonzero(~blocked.spatial_block.isin(test_groups | cal_groups))
    cal = np.flatnonzero(blocked.spatial_block.isin(cal_groups))
    test = np.flatnonzero(blocked.spatial_block.isin(test_groups))
    return train, cal, test, blocked.spatial_block


def _fit_capped(model, X, y, idx, cap, seed):
    idx = np.asarray(idx)
    if cap and len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return clone(model).fit(X.iloc[idx], y.iloc[idx])


def _regime_masks(y: pd.Series) -> dict[str, np.ndarray]:
    a = y.to_numpy()
    return {
        "lt_100": a < 100,
        "100_200": (a >= 100) & (a < 200),
        "200_370": (a >= 200) & (a < 370),
        "ge_370": a >= 370,
        "top_10pct": a >= np.quantile(a, .90),
        "top_5pct": a >= np.quantile(a, .95),
    }


def biomass_metric_rows(y: pd.Series, pred: np.ndarray, strategy: str) -> list[dict]:
    actual = y.to_numpy(); rows = []
    for regime, mask in _regime_masks(y).items():
        residual = pred[mask] - actual[mask]
        if not mask.any():
            rows.append({
                "strategy": strategy, "biomass_regime": regime, "n": 0,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "median_observed": np.nan, "median_predicted": np.nan,
                "prediction_observation_ratio": np.nan,
                "fraction_ge370_predicted_below_200": np.nan,
            })
            continue
        row = {
            "strategy": strategy, "biomass_regime": regime, "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "bias": float(residual.mean()), "median_observed": float(np.median(actual[mask])),
            "median_predicted": float(np.median(pred[mask])),
            "prediction_observation_ratio": float(np.median(pred[mask]) / np.median(actual[mask])),
            "fraction_ge370_predicted_below_200": np.nan,
        }
        if regime == "ge_370":
            row["fraction_ge370_predicted_below_200"] = float(np.mean(pred[mask] < 200))
        rows.append(row)
    return rows


def paired_error_summary(
    y: pd.Series, pred_multiyear: np.ndarray, pred_sameyear: np.ndarray, blocks: pd.Series,
    seed: int = 42, bootstrap_iterations: int = 5000,
) -> pd.DataFrame:
    multi_error = np.abs(pred_multiyear - y.to_numpy())
    same_error = np.abs(pred_sameyear - y.to_numpy())
    difference = same_error - multi_error
    by_block = pd.DataFrame({"difference": difference, "block": blocks.to_numpy()}).groupby("block").agg(total=("difference", "sum"), n=("difference", "size"))
    rng = np.random.default_rng(seed); block_rows = np.arange(len(by_block)); boot = np.empty(bootstrap_iterations)
    for i in range(bootstrap_iterations):
        selected = rng.choice(block_rows, len(block_rows), replace=True)
        boot[i] = by_block.total.iloc[selected].sum() / by_block.n.iloc[selected].sum()
    nonzero = difference[difference != 0]
    test = wilcoxon(nonzero, alternative="two-sided") if len(nonzero) else None
    return pd.DataFrame([{
        "n": len(difference), "mean_abs_error_difference_same_minus_multi": float(difference.mean()),
        "median_abs_error_difference_same_minus_multi": float(np.median(difference)),
        "fraction_improved_sameyear": float(np.mean(difference < 0)),
        "fraction_worsened_sameyear": float(np.mean(difference > 0)),
        "fraction_tied": float(np.mean(difference == 0)),
        "cluster_bootstrap_mean_ci_lower": float(np.quantile(boot, .025)),
        "cluster_bootstrap_mean_ci_upper": float(np.quantile(boot, .975)),
        "bootstrap_unit": "10 km spatial block", "bootstrap_iterations": bootstrap_iterations,
        "wilcoxon_statistic": float(test.statistic) if test else np.nan,
        "wilcoxon_pvalue": float(test.pvalue) if test else np.nan,
    }])


def _conformal_rows(y, pred, strategy, radius):
    actual = y.to_numpy(); lo, hi = interval_from_radius(pred, radius)
    masks = {"overall": np.ones(len(y), dtype=bool), **_regime_masks(y)}
    rows = []
    for regime, mask in masks.items():
        rows.append({
            "strategy": strategy, "biomass_regime": regime, "n": int(mask.sum()),
            "target_coverage": .90, "empirical_coverage": empirical_coverage(actual[mask], lo[mask], hi[mask]),
            "mean_interval_width": float(np.mean(hi[mask] - lo[mask])),
            "median_interval_width": float(np.median(hi[mask] - lo[mask])), "radius": radius,
        })
    return rows


def feature_stability(multiyear: pd.DataFrame, sameyear: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ALL_FEATURES:
        multi, same = multiyear[feature], sameyear[feature]; difference = same - multi
        rows.append({
            "sensor": next(sensor for sensor, features in SENSOR_GROUPS.items() if feature in features),
            "feature": feature, "highlight": feature in HIGHLIGHT_FEATURES,
            "pearson": float(multi.corr(same, method="pearson")),
            "spearman": float(multi.corr(same, method="spearman")),
            "median_difference_same_minus_multi": float(difference.median()),
            "median_absolute_difference": float(difference.abs().median()),
            "difference_iqr": float(difference.quantile(.75) - difference.quantile(.25)),
        })
    return pd.DataFrame(rows)


def _score_domain(validity, X, chunk_size=5000):
    return np.concatenate([validity.score_samples(X.iloc[start:start + chunk_size]) for start in range(0, len(X), chunk_size)])


def run_paired_models(multiyear: pd.DataFrame, sameyear: pd.DataFrame, config_path: Path) -> dict:
    cfg = load_yaml(config_path); spatial_cfg = cfg["spatial"]
    folds, fold_status = folds_for_shots(multiyear.shot_number, Path("data/processed/spatial_fold_assignment.parquet"))
    fold_rows = []
    for strategy in STRATEGIES:
        row = {"strategy": strategy, **fold_status.iloc[0].to_dict(), "assignments_identical": True}
        fold_rows.append(row)
    fold_table_status = pd.DataFrame(fold_rows)
    fold_table_status.to_csv("outputs/tables/temporal_2022_fold_consistency.csv", index=False)
    if not fold_table_status.consistent.all() or not np.array_equal(multiyear.shot_number, sameyear.shot_number):
        raise ValueError("Paired fold/shot consistency failed")

    model = make_hist_gradient_boosting(cfg["model"]["hist_gradient_boosting"])
    cap = cfg["compute"].get("max_train_samples_per_fold", 120000); y = multiyear.agbd
    data = {STRATEGIES[0]: multiyear, STRATEGIES[1]: sameyear}
    overall_rows, fold_metrics, biomass_rows, predictions = [], [], [], {}
    for strategy, frame in data.items():
        fold_result, pred, metrics = fixed_fold_cv(model, frame[ALL_FEATURES], y, folds, cap, 42)
        predictions[strategy] = pred
        overall_rows.append({"strategy": strategy, "n": len(y), **metrics, "fold_rmse_sd": float(fold_result.rmse.std())})
        fold_result.insert(1, "strategy", strategy); fold_metrics.append(fold_result)
        biomass_rows.extend(biomass_metric_rows(y, pred, strategy))
    overall = pd.DataFrame(overall_rows)
    base, same = overall.set_index("strategy").loc[STRATEGIES[0]], overall.set_index("strategy").loc[STRATEGIES[1]]
    overall["delta_r2_same_minus_multi"] = same.r2 - base.r2
    overall["delta_rmse_same_minus_multi"] = same.rmse - base.rmse
    overall["delta_mae_same_minus_multi"] = same.mae - base.mae
    overall.to_csv("outputs/tables/temporal_2022_model_comparison.csv", index=False)
    fold_long = pd.concat(fold_metrics, ignore_index=True)
    pivot = fold_long.pivot(index="fold", columns="strategy", values=["r2", "rmse"])
    fold_comparison = pd.DataFrame({
        "fold": pivot.index,
        "r2_multiyear": pivot[("r2", STRATEGIES[0])], "r2_sameyear": pivot[("r2", STRATEGIES[1])],
        "delta_r2_same_minus_multi": pivot[("r2", STRATEGIES[1])] - pivot[("r2", STRATEGIES[0])],
        "rmse_multiyear": pivot[("rmse", STRATEGIES[0])], "rmse_sameyear": pivot[("rmse", STRATEGIES[1])],
        "delta_rmse_same_minus_multi": pivot[("rmse", STRATEGIES[1])] - pivot[("rmse", STRATEGIES[0])],
    }).reset_index(drop=True)
    fold_comparison.to_csv("outputs/tables/temporal_2022_fold_comparison.csv", index=False)
    biomass = pd.DataFrame(biomass_rows); biomass.to_csv("outputs/tables/temporal_2022_by_biomass.csv", index=False)

    _, _, _, blocks = _spatial_split(multiyear, spatial_cfg)
    paired = paired_error_summary(y, predictions[STRATEGIES[0]], predictions[STRATEGIES[1]], blocks)
    paired.to_csv("outputs/tables/temporal_2022_paired_error_analysis.csv", index=False)

    train, calibration, test, _ = _spatial_split(multiyear, spatial_cfg)
    conformal_rows, domain_rows, heldout_predictions = [], [], {}
    for strategy, frame in data.items():
        X = frame[ALL_FEATURES]
        fitted = _fit_capped(model, X, y, train, cap, 55)
        cal_pred = fitted.predict(X.iloc[calibration]); test_pred = fitted.predict(X.iloc[test])
        heldout_predictions[strategy] = test_pred
        radius = conformal_radius(y.iloc[calibration], cal_pred, cfg["conformal"]["alpha"])
        conformal_rows.extend(_conformal_rows(y.iloc[test], test_pred, strategy, radius))
        validity = KNNDomainValidity(
            cfg["validity"]["k_neighbors"], cfg["validity"]["quantile_threshold"],
            cfg["validity"]["max_reference_samples"],
        ).fit(X.iloc[train])
        distance = _score_domain(validity, X.iloc[test]); error = np.abs(test_pred - y.iloc[test].to_numpy())
        inside = distance <= validity.threshold_
        domain_rows.append({
            "strategy": strategy, "n_test": len(test), "threshold": validity.threshold_,
            "in_domain_fraction": float(inside.mean()), "mae_in_domain": float(error[inside].mean()),
            "mae_out_domain": float(error[~inside].mean()),
            "error_distance_spearman": float(pd.Series(error).corr(pd.Series(distance), method="spearman")),
        })
    conformal = pd.DataFrame(conformal_rows); conformal.to_csv("outputs/tables/temporal_2022_conformal.csv", index=False)
    domain = pd.DataFrame(domain_rows); domain.to_csv("outputs/tables/temporal_2022_domain_validity.csv", index=False)

    sensor_rows = []
    combinations = {"S2": SENSOR_GROUPS["S2"], "S1": SENSOR_GROUPS["S1"], "PALSAR": SENSOR_GROUPS["PALSAR"], "S2+S1+PALSAR": ALL_FEATURES}
    for stack, features in combinations.items():
        stack_metrics = {}
        for strategy, frame in data.items():
            if stack == "S2+S1+PALSAR":
                metrics = overall.set_index("strategy").loc[strategy].to_dict()
            else:
                _, _, metrics = fixed_fold_cv(model, frame[features], y, folds, cap, 42)
            stack_metrics[strategy] = metrics
            sensor_rows.append({"sensor_stack": stack, "strategy": strategy, "n": len(y), **{k: metrics[k] for k in ["rmse", "mae", "r2", "bias", "nrmse_mean"]}})
        for row in sensor_rows[-2:]:
            row["delta_r2_same_minus_multi"] = stack_metrics[STRATEGIES[1]]["r2"] - stack_metrics[STRATEGIES[0]]["r2"]
            row["delta_rmse_same_minus_multi"] = stack_metrics[STRATEGIES[1]]["rmse"] - stack_metrics[STRATEGIES[0]]["rmse"]
    sensor_effect = pd.DataFrame(sensor_rows); sensor_effect.to_csv("outputs/tables/temporal_2022_sensor_effect.csv", index=False)
    return {"overall": overall, "folds": fold_comparison, "biomass": biomass, "paired": paired,
            "conformal": conformal, "domain": domain, "sensor_effect": sensor_effect,
            "predictions": predictions, "test_indices": test, "heldout_predictions": heldout_predictions}


def _make_figures(results: dict, stability: pd.DataFrame, figure_dir: Path) -> None:
    overall, folds, biomass = results["overall"], results["folds"], results["biomass"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(folds)); width = .36
    axes[0].bar(x - width/2, folds.rmse_multiyear, width, label="MULTIYEAR"); axes[0].bar(x + width/2, folds.rmse_sameyear, width, label="SAME_YEAR_2022")
    axes[0].set(xticks=x, xticklabels=folds.fold, xlabel="Fold", ylabel="RMSE", title="Fold RMSE"); axes[0].legend(fontsize=8)
    axes[1].bar(x - width/2, folds.r2_multiyear, width); axes[1].bar(x + width/2, folds.r2_sameyear, width)
    axes[1].set(xticks=x, xticklabels=folds.fold, xlabel="Fold", ylabel="R²", title="Fold R²")
    _save(fig, figure_dir / "temporal_2022_fold_comparison.png")
    pivot = biomass.pivot(index="biomass_regime", columns="strategy", values="rmse")
    fig, ax = plt.subplots(figsize=(10, 4.5)); pivot.plot.bar(ax=ax); ax.set(ylabel="RMSE", title="Error by biomass regime")
    _save(fig, figure_dir / "temporal_2022_error_by_biomass.png")
    high = biomass[biomass.biomass_regime.isin(["ge_370", "top_10pct", "top_5pct"])].pivot(index="biomass_regime", columns="strategy", values="bias")
    fig, ax = plt.subplots(figsize=(8, 4.5)); high.plot.bar(ax=ax); ax.axhline(0, color="black"); ax.set(ylabel="Bias", title="Upper-tail bias")
    _save(fig, figure_dir / "temporal_2022_high_biomass_bias.png")
    conf = results["conformal"].pivot(index="biomass_regime", columns="strategy", values="empirical_coverage")
    fig, ax = plt.subplots(figsize=(10, 4.5)); conf.plot.bar(ax=ax); ax.axhline(.9, color="black", ls="--"); ax.set(ylabel="Coverage", title="90% conformal coverage")
    _save(fig, figure_dir / "temporal_2022_conformal.png")
    ordered = stability.sort_values(["sensor", "feature"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(ordered.feature, ordered.spearman); axes[0].set(ylabel="Spearman", title="Multi-year vs 2022 feature stability")
    axes[1].bar(ordered.feature, ordered.median_absolute_difference); axes[1].set(ylabel="Median absolute difference")
    axes[1].tick_params(axis="x", rotation=60)
    _save(fig, figure_dir / "temporal_2022_feature_stability.png")
    sensor = results["sensor_effect"].pivot(index="sensor_stack", columns="strategy", values="rmse")
    fig, ax = plt.subplots(figsize=(9, 4.5)); sensor.plot.bar(ax=ax); ax.set(ylabel="RMSE", title="Sensor-specific temporal effect")
    _save(fig, figure_dir / "temporal_2022_sensor_effect.png")
    h = biomass[biomass.biomass_regime.eq("ge_370")].set_index("strategy")
    c = results["conformal"][results["conformal"].biomass_regime.eq("overall")].set_index("strategy")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8)); labels = ["MULTIYEAR", "SAME YEAR"]
    colors = ["#287271", "#D98C4A"]
    bars = axes[0,0].bar(labels, overall.r2, color=colors); axes[0,0].set(title="Spatial CV R²")
    axes[0,0].bar_label(bars, fmt="%.3f", padding=3)
    bars = axes[0,1].bar(labels, overall.rmse, color=colors); axes[0,1].set(title="Spatial CV RMSE", ylabel="Mg/ha")
    axes[0,1].bar_label(bars, fmt="%.2f", padding=3)
    bars = axes[1,0].bar(labels, [h.loc[s, "bias"] for s in STRATEGIES], color=colors); axes[1,0].axhline(0, color="black"); axes[1,0].set(title="Bias at AGBD ≥370", ylabel="Mg/ha")
    axes[1,0].bar_label(bars, fmt="%.1f", padding=3)
    bars = axes[1,1].bar(labels, [c.loc[s, "empirical_coverage"] for s in STRATEGIES], color=colors); axes[1,1].axhline(.9, color="black", ls="--"); axes[1,1].set(title="Overall 90% coverage", ylim=(0, 1))
    axes[1,1].bar_label(bars, fmt="%.3f", padding=3)
    _save(fig, figure_dir / "temporal_2022_summary.png")


def _write_report(summary: dict, results: dict, stability: pd.DataFrame) -> str:
    overall = results["overall"].set_index("strategy"); folds = results["folds"]
    biomass = results["biomass"].set_index(["strategy", "biomass_regime"])
    conformal = results["conformal"].set_index(["strategy", "biomass_regime"])
    domain = results["domain"].set_index("strategy"); paired = results["paired"].iloc[0]
    delta_rmse = overall.loc[STRATEGIES[1], "rmse"] - overall.loc[STRATEGIES[0], "rmse"]
    delta_r2 = overall.loc[STRATEGIES[1], "r2"] - overall.loc[STRATEGIES[0], "r2"]
    better_folds = int((folds.delta_rmse_same_minus_multi < 0).sum())
    sensor_delta = results["sensor_effect"].drop_duplicates("sensor_stack").set_index("sensor_stack").delta_rmse_same_minus_multi
    most_sensitive = sensor_delta.abs().idxmax()
    if delta_rmse <= -1 and better_folds >= 4:
        interpretation = "Same-year 2022 features clearly improve the paired spatial validation; temporal mismatch is an important source of label-feature noise."
        recommendation = "Annual temporal matching is justified for training and annual production maps, subject to yearly data availability."
    elif abs(delta_rmse) < 1 and abs(delta_r2) < .01:
        interpretation = "The paired results are nearly identical; temporal mismatch is secondary to sensor saturation, target uncertainty, and spatial ambiguity."
        recommendation = "Retain the operationally simpler multi-year composite unless annual products are needed for change monitoring."
    elif delta_rmse > 0:
        interpretation = "The multi-year composite performs better, consistent with temporal aggregation reducing seasonal, cloud, speckle, and moisture variability enough to outweigh temporal mismatch."
        recommendation = "Retain multi-year aggregation for production and treat annual matching as a change-monitoring product rather than the default biomass model."
    else:
        interpretation = "Same-year features improve the aggregate metric, but the effect is too small or spatially inconsistent to establish a clear temporal-matching advantage."
        recommendation = "Do not add annual-matching complexity without replication in other GEDI years."
    text = f"""# Paired 2022 temporal-alignment results

## Design

The experiment contains {summary['retention']['final_common_rows']:,} GEDI shots acquired in 2022. Both strategies use exactly the same shots, feature names, persisted 10 km folds, HistGradientBoosting model, hyperparameters, and training caps. Only feature timing differs: the existing 2020–2024 composite versus the 2022 rasters.

## Direct answers

1. **Does 2022-only remote sensing improve performance?** Multi-year RMSE/R² are {overall.loc[STRATEGIES[0],'rmse']:.2f}/{overall.loc[STRATEGIES[0],'r2']:.3f}; same-year values are {overall.loc[STRATEGIES[1],'rmse']:.2f}/{overall.loc[STRATEGIES[1],'r2']:.3f}. The paired deltas (same minus multi) are {delta_rmse:.2f} RMSE and {delta_r2:.4f} R². {interpretation}
2. **Fold stability:** same-year RMSE is lower in {better_folds}/5 spatial folds. Fold-level deltas are `{folds.delta_rmse_same_minus_multi.round(2).tolist()}` Mg/ha.
3. **Sensor temporal effect:** the largest absolute RMSE change occurs for **{most_sensitive}** ({sensor_delta.loc[most_sensitive]:.2f} Mg/ha, same minus multi). Feature-level correlations and differences are in the stability table.
4. **High biomass:** bias at AGBD ≥370 changes from {biomass.loc[(STRATEGIES[0],'ge_370'),'bias']:.2f} to {biomass.loc[(STRATEGIES[1],'ge_370'),'bias']:.2f} Mg/ha. The fraction predicted below 200 changes from {biomass.loc[(STRATEGIES[0],'ge_370'),'fraction_ge370_predicted_below_200']:.3f} to {biomass.loc[(STRATEGIES[1],'ge_370'),'fraction_ge370_predicted_below_200']:.3f}.
5. **Conformal calibration:** overall coverage changes from {conformal.loc[(STRATEGIES[0],'overall'),'empirical_coverage']:.3f} to {conformal.loc[(STRATEGIES[1],'overall'),'empirical_coverage']:.3f}; ≥370 coverage changes from {conformal.loc[(STRATEGIES[0],'ge_370'),'empirical_coverage']:.3f} to {conformal.loc[(STRATEGIES[1],'ge_370'),'empirical_coverage']:.3f}.
6. **Domain validity:** in-domain fraction changes from {domain.loc[STRATEGIES[0],'in_domain_fraction']:.3f} to {domain.loc[STRATEGIES[1],'in_domain_fraction']:.3f}. Domain distance is an extrapolation diagnostic, not predictive uncertainty.
7. **Operational decision:** {recommendation}

## Paired error evidence

Mean absolute-error difference (same minus multi) is {paired.mean_abs_error_difference_same_minus_multi:.3f} Mg/ha, with a 10 km spatial-block bootstrap 95% interval [{paired.cluster_bootstrap_mean_ci_lower:.3f}, {paired.cluster_bootstrap_mean_ci_upper:.3f}]. Same-year improves {paired.fraction_improved_sameyear:.1%} of observations and worsens {paired.fraction_worsened_sameyear:.1%}. The Wilcoxon result is reported in the table, but effect sizes and spatial consistency—not the large-sample p-value—drive interpretation.

## Conclusion

{interpretation}

{recommendation}
"""
    Path("report/temporal_2022_results.md").write_text(text, encoding="utf-8")
    return interpretation


def run_temporal_2022_experiment(
    raw_path: Path = Path("data/raw/gedi_l4a_tai_filtered.csv"),
    multiyear_path: Path = Path("data/interim/gedi_s2_s1_palsar.parquet"),
    raster_dir: Path = Path("data/raw/temporal"),
    config_path: Path = Path("configs/model.yaml"),
) -> dict:
    outputs = Path("outputs/tables"); outputs.mkdir(parents=True, exist_ok=True)
    figure_dir = Path("outputs/figures/temporal"); figure_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        "S2": raster_dir / "sentinel2_tai_features_2022.tif",
        "S1": raster_dir / "sentinel1_tai_features_2022.tif",
        "PALSAR": raster_dir / "palsar_tai_features_2022.tif",
    }
    missing = [str(path) for path in selected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required downloaded 2022 raster(s) missing: {missing}")
    audit = audit_temporal_rasters(raster_dir, selected, outputs / "temporal_2022_raster_audit.csv")
    raw_2022, date_summary = summarize_gedi_2022(raw_path, outputs / "gedi_2022_summary.csv")
    paired_data = build_paired_2022_datasets(raw_2022, multiyear_path, selected)
    stability = feature_stability(paired_data["multiyear"], paired_data["sameyear"])
    stability.to_csv(outputs / "temporal_2022_feature_stability.csv", index=False)
    results = run_paired_models(paired_data["multiyear"], paired_data["sameyear"], config_path)
    _make_figures(results, stability, figure_dir)
    payload = {"date_summary": date_summary, "retention": paired_data["retention"]}
    interpretation = _write_report(payload, results, stability)
    summary = {
        **payload,
        "selected_rasters": {sensor: path.name for sensor, path in selected.items()},
        "raster_audit": audit.to_dict("records"),
        "model_comparison": results["overall"].to_dict("records"),
        "fold_comparison": results["folds"].to_dict("records"),
        "high_biomass": results["biomass"].loc[results["biomass"].biomass_regime.eq("ge_370")].to_dict("records"),
        "conformal": results["conformal"].to_dict("records"),
        "sensor_effect": results["sensor_effect"].to_dict("records"),
        "feature_stability": stability.to_dict("records"),
        "scientific_interpretation": interpretation,
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/temporal_2022_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
