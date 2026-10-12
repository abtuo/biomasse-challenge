from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import S2_BANDS, S2_INDICES
from .modeling import regression_metrics
from .uncertainty import conformal_radius, interval_from_radius

BASE = "national_gedi_prediction"
F2 = ["NDVI", "NDMI", "NBR", "NDRE"]
F3 = ["B11", "B12", "NDMI", "NDRE", "elevation", "slope"]
F4 = S2_BANDS + S2_INDICES + ["elevation", "slope", "landcover"]
FEATURE_VARIANTS = {"F1": [BASE], "F2": F2, "F3": F3, "F4": F4}
PRIMARY_CONFIGS = [
    ("GEDI_NATIONAL_BASELINE", "F1"),
    ("FIELD_ONLY_RIDGE", "F3"),
    ("FIELD_ONLY_TREE", "F3"),
    ("GEDI_LINEAR_CALIBRATION", "F1"),
    ("GEDI_RESIDUAL_CORRECTION", "F3"),
    ("STACKED_FIELD_MODEL", "F3"),
]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _region(latitude) -> pd.Categorical:
    return pd.cut(
        latitude, [-np.inf, 6.5, 8.5, np.inf], labels=["SOUTH", "CENTRE", "NORTH"],
        right=False,
    )


def _ecological_stratum(landcover) -> pd.Categorical:
    values = pd.Series(landcover)
    labels = np.select(
        [values.eq(10), values.eq(40), values.isin([20, 30])],
        ["forest-like", "mosaic/agriculture", "savanna/open"], default="other",
    )
    return pd.Categorical(labels, categories=[
        "forest-like", "mosaic/agriculture", "savanna/open", "other",
    ])


def build_clean_field_dataset(
    feature_path: Path = Path("data/interim/field_ci_national_features.parquet"),
    national_predictions: Path = Path("outputs/tables/national_field_predictions.csv"),
    tai_predictions: Path = Path("outputs/tables/tai_transfer_field_predictions.csv"),
) -> pd.DataFrame:
    field = pd.read_parquet(feature_path)
    enrichment = [
        BASE, "national_pi_lower", "national_pi_upper", "national_domain_distance",
        "national_in_domain", "tai_domain_distance", "tai_in_domain",
        "required_features_available", "region", "ecological_stratum",
        "national_interval_width",
    ]
    field = field.drop(columns=[column for column in enrichment if column in field], errors="ignore")
    national = pd.read_csv(national_predictions)[[
        "plot_id", "predicted_agbd", "pi_lower", "pi_upper", "domain_distance", "in_domain",
    ]].rename(columns={
        "predicted_agbd": BASE, "pi_lower": "national_pi_lower",
        "pi_upper": "national_pi_upper", "domain_distance": "national_domain_distance",
        "in_domain": "national_in_domain",
    })
    tai = pd.read_csv(tai_predictions)[[
        "plot_id", "domain_distance", "in_domain", "required_features_available",
    ]].rename(columns={
        "domain_distance": "tai_domain_distance", "in_domain": "tai_in_domain",
    })
    out = field.merge(national, on="plot_id", how="left").merge(tai, on="plot_id", how="left")
    out = out.loc[out[BASE].notna()].copy().reset_index(drop=True)
    out["region"] = _region(out.latitude).astype(str)
    out["ecological_stratum"] = _ecological_stratum(out.landcover).astype(str)
    out["national_interval_width"] = out.national_pi_upper - out.national_pi_lower
    out.to_parquet(feature_path, index=False)
    _field_audit(out)
    return out


def _field_audit(field: pd.DataFrame):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    x, y = transformer.transform(field.longitude, field.latitude)
    coordinates = np.column_stack([x, y])
    distances, _ = cKDTree(coordinates).query(coordinates, k=2)
    rows = [
        {"section": "sample", "metric": "valid_rows", "value": len(field)},
        {"section": "biomass", "metric": "min", "value": field.field_agbd.min()},
        {"section": "biomass", "metric": "median", "value": field.field_agbd.median()},
        {"section": "biomass", "metric": "max", "value": field.field_agbd.max()},
        {"section": "geography", "metric": "median_nearest_plot_distance_km",
         "value": np.median(distances[:, 1]) / 1000},
        {"section": "domain", "metric": "national_out_of_domain",
         "value": int((~field.national_in_domain.astype(bool)).sum())},
        {"section": "domain", "metric": "tai_features_available",
         "value": int(field.required_features_available.sum())},
    ]
    for q in [.05, .1, .25, .5, .75, .9, .95]:
        rows.append({"section": "biomass_quantile", "metric": f"q{q:g}",
                     "value": field.field_agbd.quantile(q)})
    for column in field.columns:
        rows.append({"section": "missingness", "metric": column,
                     "value": int(field[column].isna().sum())})
    pd.DataFrame(rows).to_csv("outputs/tables/field_learning_audit.csv", index=False)
    counts = []
    for grouping in ["region", "ecological_stratum", "landcover", "national_in_domain"]:
        for value, count in field[grouping].value_counts(dropna=False).items():
            counts.append({"grouping": grouping, "group": value, "n": count})
    pd.DataFrame(counts).to_csv("outputs/tables/field_group_counts.csv", index=False)


def _balanced_group_folds(groups, n_splits=5) -> np.ndarray:
    groups = np.asarray(groups)
    folds = np.full(len(groups), -1, dtype="int8")
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        folds[test] = fold
    return folds


def create_field_cv_assignments(field: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    x, y = transformer.transform(field.longitude, field.latitude)
    output = field[["plot_id", "region", "ecological_stratum"]].copy()
    rng_folds = np.full(len(field), -1, dtype="int8")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, test) in enumerate(kfold.split(field)):
        rng_folds[test] = fold
    output["random_fold"] = rng_folds
    for size, label in [(50_000, "spatial50"), (100_000, "spatial100")]:
        blocks = np.char.add(
            np.char.add((np.floor(np.asarray(x) / size).astype(int)).astype(str), "_"),
            (np.floor(np.asarray(y) / size).astype(int)).astype(str),
        )
        output[f"{label}_block"] = blocks
        output[f"{label}_fold"] = _balanced_group_folds(blocks)
    output["leave_region_out_fold"] = pd.Categorical(
        output.region, categories=["SOUTH", "CENTRE", "NORTH"]
    ).codes
    path = Path("data/processed/field_cv_assignments.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)
    return output


def _features(variant: str, include_base: bool = False) -> list[str]:
    features = list(FEATURE_VARIANTS[variant])
    if include_base and BASE not in features:
        features = [BASE] + features
    return features


def _preprocessor(features: list[str], scale=True) -> ColumnTransformer:
    numeric = [feature for feature in features if feature != "landcover"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if "landcover" in features:
        transformers.append(("landcover", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["landcover"]))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _ridge(features: list[str], alpha=20.0) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor(features)), ("model", Ridge(alpha=alpha))])


def _tree(features: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", _preprocessor(features, scale=False)),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=5, min_samples_leaf=12, max_features=.7,
            random_state=42, n_jobs=-1,
        )),
    ])


def fit_predict_method(
    method: str,
    variant: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Training and test sets must both be non-empty")
    if set(train.plot_id) & set(test.plot_id):
        raise ValueError("Field leakage: train and test plot IDs overlap")
    if method == "GEDI_NATIONAL_BASELINE":
        return test[BASE].to_numpy()
    if method == "FIELD_ONLY_RIDGE":
        features = _features(variant)
        return _ridge(features).fit(train[features], train.field_agbd).predict(test[features])
    if method == "FIELD_ONLY_TREE":
        features = _features(variant)
        return _tree(features).fit(train[features], train.field_agbd).predict(test[features])
    if method == "GEDI_LINEAR_CALIBRATION":
        model = LinearRegression().fit(train[[BASE]], train.field_agbd)
        return model.predict(test[[BASE]])
    if method == "GEDI_RESIDUAL_CORRECTION":
        features = _features(variant, include_base=True)
        residual = train.field_agbd - train[BASE]
        correction = _ridge(features, alpha=30).fit(train[features], residual)
        return test[BASE].to_numpy() + correction.predict(test[features])
    if method == "STACKED_FIELD_MODEL":
        features = _features(variant, include_base=True)
        return _ridge(features, alpha=20).fit(train[features], train.field_agbd).predict(test[features])
    raise ValueError(f"Unknown field-learning method: {method}")


def _metric_row(observed, predicted, **metadata) -> dict:
    return {**metadata, "n": len(observed), **regression_metrics(observed, predicted),
            "spearman": float(spearmanr(observed, predicted).statistic)}


def evaluate_field_models(field: pd.DataFrame, assignments: pd.DataFrame):
    data = field.merge(assignments.drop(columns=["region", "ecological_stratum"]), on="plot_id")
    full_configs = [("GEDI_NATIONAL_BASELINE", "F1"), ("GEDI_LINEAR_CALIBRATION", "F1")]
    for variant in ["F2", "F3", "F4"]:
        full_configs.extend([
            ("FIELD_ONLY_RIDGE", variant), ("FIELD_ONLY_TREE", variant),
            ("GEDI_RESIDUAL_CORRECTION", variant), ("STACKED_FIELD_MODEL", variant),
        ])
    protocols = [("SPATIAL_BLOCK_CV_50KM", "spatial50_fold", full_configs)]
    protocols.extend([
        ("RANDOM_CV", "random_fold", PRIMARY_CONFIGS),
        ("SPATIAL_BLOCK_CV_100KM", "spatial100_fold", PRIMARY_CONFIGS),
        ("LEAVE_REGION_OUT", "leave_region_out_fold", PRIMARY_CONFIGS),
    ])
    summary_rows, prediction_rows = [], []
    for protocol, fold_column, configs in protocols:
        for method, variant in configs:
            prediction = np.full(len(data), np.nan)
            for fold in sorted(data[fold_column].unique()):
                train = data.loc[data[fold_column] != fold]
                test = data.loc[data[fold_column] == fold]
                prediction[test.index] = fit_predict_method(method, variant, train, test)
            summary_rows.append(_metric_row(
                data.field_agbd, prediction, protocol=protocol, method=method,
                feature_variant=variant, n_features=len(_features(
                    variant, include_base=method in {"GEDI_RESIDUAL_CORRECTION", "STACKED_FIELD_MODEL"}
                )),
            ))
            if protocol == "SPATIAL_BLOCK_CV_50KM" and (method, variant) in PRIMARY_CONFIGS:
                for index, value in enumerate(prediction):
                    prediction_rows.append({
                        "plot_id": data.iloc[index].plot_id, "method": method,
                        "feature_variant": variant, "observed": data.iloc[index].field_agbd,
                        "predicted": value, "region": data.iloc[index].region,
                    })
    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv("outputs/tables/field_model_comparison.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv("outputs/tables/field_primary_oof_predictions.csv", index=False)
    return comparison, predictions, data


def select_training_plots(
    pool: pd.DataFrame,
    n: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    """Select plot positions without consulting field_agbd."""
    if n >= len(pool):
        return np.arange(len(pool))
    rng = np.random.default_rng(seed)
    if strategy == "RANDOM":
        return np.sort(rng.choice(len(pool), n, replace=False))
    if strategy == "SPATIALLY_BALANCED":
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
        x, y = transformer.transform(pool.longitude, pool.latitude)
        coordinates = StandardScaler().fit_transform(np.column_stack([x, y]))
        selected = [int(rng.integers(len(pool)))]
        minimum = pairwise_distances(coordinates, coordinates[selected]).ravel()
        while len(selected) < n:
            candidate = int(np.argmax(minimum + rng.random(len(pool)) * 1e-9))
            selected.append(candidate)
            new_distance = pairwise_distances(coordinates, coordinates[[candidate]]).ravel()
            minimum = np.minimum(minimum, new_distance)
            minimum[selected] = -np.inf
        return np.sort(selected)
    if strategy == "DOMAIN_GUIDED":
        domain = pool.national_domain_distance.rank(pct=True).to_numpy()
        width = pool.national_interval_width.rank(pct=True).to_numpy()
        score = .75 * domain + .25 * width + rng.random(len(pool)) * .03
        return np.sort(np.argsort(score)[-n:])
    if strategy == "PREDICTED_BIOMASS_STRATIFIED":
        strata = pd.qcut(pool[BASE], q=min(3, pool[BASE].nunique()), labels=False, duplicates="drop")
        selected = []
        groups = [np.flatnonzero(strata.to_numpy() == value) for value in sorted(strata.unique())]
        while len(selected) < n:
            progressed = False
            for group in groups:
                available = np.setdiff1d(group, np.asarray(selected, dtype=int), assume_unique=False)
                if len(available) and len(selected) < n:
                    selected.append(int(rng.choice(available)))
                    progressed = True
            if not progressed:
                break
        return np.sort(selected)
    if strategy == "HYBRID":
        predicted_bin = pd.qcut(pool[BASE], q=min(3, pool[BASE].nunique()), labels=False,
                                duplicates="drop").astype(str)
        groups = pool.region.astype(str) + "|" + predicted_bin
        ranked = pool.national_domain_distance.rank(pct=True).to_numpy() + rng.random(len(pool)) * .1
        buckets = {group: list(indices[np.argsort(ranked[indices])[::-1]])
                   for group, indices in pd.Series(np.arange(len(pool))).groupby(groups).groups.items()}
        selected = []
        while len(selected) < n and any(buckets.values()):
            for group in sorted(buckets):
                if buckets[group] and len(selected) < n:
                    selected.append(int(buckets[group].pop(0)))
        return np.sort(selected)
    raise ValueError(f"Unknown selection strategy: {strategy}")


def run_sample_efficiency(data: pd.DataFrame) -> pd.DataFrame:
    sizes: list[int | str] = [10, 20, 40, 50, 60, 100, 150, 200, "all"]
    configs = [
        ("GEDI_NATIONAL_BASELINE", "F1"), ("GEDI_LINEAR_CALIBRATION", "F1"),
        ("GEDI_RESIDUAL_CORRECTION", "F3"), ("FIELD_ONLY_RIDGE", "F3"),
        ("STACKED_FIELD_MODEL", "F3"),
    ]
    rows = []
    for fold in sorted(data.spatial50_fold.unique()):
        pool = data.loc[data.spatial50_fold != fold].reset_index(drop=True)
        test = data.loc[data.spatial50_fold == fold].reset_index(drop=True)
        for repeat in range(4):
            for requested in sizes:
                n = len(pool) if requested == "all" else min(int(requested), len(pool))
                seed = 10_000 + int(fold) * 100 + repeat * 10 + n
                positions = select_training_plots(pool, n, "RANDOM", seed)
                train = pool.iloc[positions]
                for method, variant in configs:
                    prediction = fit_predict_method(method, variant, train, test)
                    rows.append(_metric_row(
                        test.field_agbd, prediction, requested_n=str(requested), n_train=len(train),
                        fold=int(fold), repeat=repeat, method=method, feature_variant=variant,
                    ))
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_sample_efficiency.csv", index=False)
    _sample_efficiency_figures(result)
    return result


def _sample_efficiency_figures(result: pd.DataFrame):
    order = ["10", "20", "40", "50", "60", "100", "150", "200", "all"]
    for metric, filename, ylabel in [
        ("rmse", "sample_efficiency_rmse.png", "Held-out RMSE (Mg/ha)"),
        ("mae", "sample_efficiency_mae.png", "Held-out MAE (Mg/ha)"),
    ]:
        summary = result.groupby(["requested_n", "method"], as_index=False)[metric].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(order))
        for method, group in summary.groupby("method"):
            indexed = group.set_index("requested_n").reindex(order)
            ax.plot(x, indexed["mean"], marker="o", label=method)
            ax.fill_between(x, indexed["mean"] - indexed["std"],
                            indexed["mean"] + indexed["std"], alpha=.12)
        ax.set(xticks=x, xticklabels=order, xlabel="Field training plots", ylabel=ylabel,
               title="Spatially held-out field sample efficiency")
        ax.legend(fontsize=8)
        _save(fig, Path("outputs/figures/field_learning") / filename)


def run_sampling_strategies(data: pd.DataFrame) -> pd.DataFrame:
    strategies = [
        "RANDOM", "SPATIALLY_BALANCED", "DOMAIN_GUIDED",
        "PREDICTED_BIOMASS_STRATIFIED", "HYBRID",
    ]
    rows = []
    for fold in sorted(data.spatial50_fold.unique()):
        pool = data.loc[data.spatial50_fold != fold].reset_index(drop=True)
        test = data.loc[data.spatial50_fold == fold].reset_index(drop=True)
        for n in [20, 40, 60]:
            for repeat in range(3):
                for strategy in strategies:
                    seed = 20_000 + int(fold) * 100 + repeat * 10 + n
                    positions = select_training_plots(pool, min(n, len(pool)), strategy, seed)
                    train = pool.iloc[positions]
                    prediction = fit_predict_method("GEDI_RESIDUAL_CORRECTION", "F3", train, test)
                    rows.append(_metric_row(
                        test.field_agbd, prediction, n_train=len(train), fold=int(fold), repeat=repeat,
                        strategy=strategy, method="GEDI_RESIDUAL_CORRECTION", feature_variant="F3",
                    ))
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_sampling_strategy_comparison.csv", index=False)
    summary = result.groupby(["n_train", "strategy"], as_index=False).rmse.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for strategy, group in summary.groupby("strategy"):
        ax.errorbar(group.n_train, group["mean"], yerr=group["std"], marker="o", capsize=3,
                    label=strategy)
    ax.set(xlabel="Field-plot budget", ylabel="Held-out RMSE (Mg/ha)",
           title="Leakage-safe field selection strategies")
    ax.legend(fontsize=8)
    _save(fig, Path("outputs/figures/field_learning/sampling_strategy_comparison.png"))
    return result


def run_distance_analysis(data: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("FIELD_ONLY_RIDGE", "F3"), ("GEDI_RESIDUAL_CORRECTION", "F3"),
        ("STACKED_FIELD_MODEL", "F3"),
    ]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    rows = []
    for fold in sorted(data.spatial50_fold.unique()):
        train = data.loc[data.spatial50_fold != fold].reset_index(drop=True)
        test = data.loc[data.spatial50_fold == fold].reset_index(drop=True)
        train_x, train_y = transformer.transform(train.longitude, train.latitude)
        test_x, test_y = transformer.transform(test.longitude, test.latitude)
        geographic = cKDTree(np.column_stack([train_x, train_y])).query(
            np.column_stack([test_x, test_y]), k=1
        )[0] / 1000
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(imputer.fit_transform(train[F3]))
        test_features = scaler.transform(imputer.transform(test[F3]))
        feature_distance = NearestNeighbors(n_neighbors=1).fit(train_features).kneighbors(
            test_features
        )[0][:, 0]
        for method, variant in methods:
            prediction = fit_predict_method(method, variant, train, test)
            for index in range(len(test)):
                rows.append({
                    "plot_id": test.iloc[index].plot_id, "fold": int(fold), "method": method,
                    "observed": test.iloc[index].field_agbd, "predicted": prediction[index],
                    "absolute_error": abs(prediction[index] - test.iloc[index].field_agbd),
                    "nearest_training_geographic_km": geographic[index],
                    "nearest_training_feature_distance": feature_distance[index],
                })
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_generalization_distance.csv", index=False)
    for x, filename, xlabel in [
        ("nearest_training_geographic_km", "error_vs_field_distance.png",
         "Nearest training plot (km)"),
        ("nearest_training_feature_distance", "error_vs_feature_distance.png",
         "Nearest training plot in standardized F3 space"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for axis, (method, group) in zip(axes, result.groupby("method")):
            axis.scatter(group[x], group.absolute_error, s=16, alpha=.55)
            rho = group[x].corr(group.absolute_error, method="spearman")
            axis.set(title=f"{method}\nSpearman={rho:.2f}", xlabel=xlabel,
                     ylabel="Absolute error (Mg/ha)")
        _save(fig, Path("outputs/figures/field_learning") / filename)
    return result


def run_regional_transfer(data: pd.DataFrame) -> pd.DataFrame:
    methods = [("GEDI_LINEAR_CALIBRATION", "F1"), ("GEDI_RESIDUAL_CORRECTION", "F3")]
    rows = []
    # Global spatial CV provides the reference within-population correction.
    for method, variant in methods:
        for fold in sorted(data.spatial50_fold.unique()):
            train = data.loc[data.spatial50_fold != fold]
            test = data.loc[data.spatial50_fold == fold]
            prediction = fit_predict_method(method, variant, train, test)
            for region, positions in test.groupby("region").groups.items():
                local = test.index.get_indexer(positions)
                rows.append(_metric_row(
                    test.loc[positions, "field_agbd"], prediction[local], source_region="GLOBAL",
                    test_region=region, evaluation="GLOBAL_SPATIAL_CV", method=method,
                ))
    # Source-region training. One source-region fold is withheld for a leakage-safe within-region test.
    for source in ["SOUTH", "NORTH"]:
        source_data = data.loc[data.region == source]
        source_folds = sorted(source_data.spatial50_fold.unique())
        within_fold = source_folds[0]
        for method, variant in methods:
            within_train = source_data.loc[source_data.spatial50_fold != within_fold]
            within_test = source_data.loc[source_data.spatial50_fold == within_fold]
            if len(within_train) and len(within_test):
                prediction = fit_predict_method(method, variant, within_train, within_test)
                rows.append(_metric_row(
                    within_test.field_agbd, prediction, source_region=source, test_region=source,
                    evaluation="WITHIN_REGION_HELD_OUT", method=method,
                ))
            for target in [region for region in ["SOUTH", "CENTRE", "NORTH"] if region != source]:
                target_data = data.loc[data.region == target]
                prediction = fit_predict_method(method, variant, source_data, target_data)
                source_lat = source_data.latitude.mean()
                target_lat = target_data.latitude.mean()
                rows.append(_metric_row(
                    target_data.field_agbd, prediction, source_region=source, test_region=target,
                    evaluation="CROSS_REGION", method=method,
                    centroid_latitude_distance_km=abs(source_lat - target_lat) * 111.1,
                ))
    # Region-specific models are evaluated only within their own spatial folds.
    for region, region_data in data.groupby("region"):
        unique_folds = sorted(region_data.spatial50_fold.unique())
        if len(unique_folds) < 2 or len(region_data) < 20:
            continue
        for method, variant in methods:
            observed, predicted = [], []
            for fold in unique_folds:
                train = region_data.loc[region_data.spatial50_fold != fold]
                test = region_data.loc[region_data.spatial50_fold == fold]
                if len(train) < 10 or len(test) == 0:
                    continue
                observed.extend(test.field_agbd)
                predicted.extend(fit_predict_method(method, variant, train, test))
            rows.append(_metric_row(
                observed, predicted, source_region=region, test_region=region,
                evaluation="REGION_SPECIFIC_SPATIAL_CV", method=method,
            ))
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_regional_transfer.csv", index=False)
    plot = result[result.evaluation.isin(["WITHIN_REGION_HELD_OUT", "CROSS_REGION"])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = plot.source_region + "→" + plot.test_region + "\n" + plot.method.str.replace("GEDI_", "")
    ax.bar(np.arange(len(plot)), plot.rmse)
    ax.set(xticks=np.arange(len(plot)), xticklabels=labels, ylabel="RMSE (Mg/ha)",
           title="Within-region and cross-region field correction")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, Path("outputs/figures/field_learning/field_regional_transfer.png"))
    return result


def run_biomass_analysis(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    q33, q67 = data.field_agbd.quantile([1 / 3, 2 / 3])
    labels = pd.cut(data.field_agbd, [-np.inf, q33, q67, np.inf],
                    labels=["LOW", "MIDDLE", "UPPER"], include_lowest=True)
    lookup = pd.DataFrame({"plot_id": data.plot_id, "field_biomass_regime": labels.astype(str)})
    merged = predictions.merge(lookup, on="plot_id")
    rows = []
    for (method, regime), group in merged.groupby(["method", "field_biomass_regime"]):
        rows.append(_metric_row(
            group.observed, group.predicted, method=method, biomass_regime=regime,
            lower_threshold=q33, upper_threshold=q67,
        ))
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_learning_by_biomass.csv", index=False)
    return result


def run_complexity_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["RIDGE", "SHALLOW_RF", "REGULARIZED_HGBR"]:
        prediction = np.full(len(data), np.nan)
        for fold in sorted(data.spatial50_fold.unique()):
            train = data.loc[data.spatial50_fold != fold]
            test = data.loc[data.spatial50_fold == fold]
            if name == "RIDGE":
                model = _ridge(F3, alpha=20)
            elif name == "SHALLOW_RF":
                model = _tree(F3)
            else:
                model = Pipeline([
                    ("preprocessor", _preprocessor(F3)),
                    ("model", HistGradientBoostingRegressor(
                        learning_rate=.05, max_iter=120, max_leaf_nodes=7,
                        min_samples_leaf=15, l2_regularization=10, random_state=42,
                    )),
                ])
            fitted = model.fit(train[F3], train.field_agbd)
            prediction[test.index] = fitted.predict(test[F3])
        rows.append(_metric_row(data.field_agbd, prediction, model=name, feature_variant="F3"))
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_model_complexity.csv", index=False)
    return result


def run_uncertainty(field: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prediction_lookup = field.set_index("plot_id")
    field_regime = pd.cut(
        field.field_agbd,
        [-np.inf, field.field_agbd.quantile(1 / 3), field.field_agbd.quantile(2 / 3), np.inf],
        labels=["LOW", "MIDDLE", "UPPER"], include_lowest=True,
    )
    regime_lookup = pd.Series(field_regime.to_numpy(), index=field.plot_id)
    detail = []
    for fold in sorted(data.spatial50_fold.unique()):
        train = data.loc[data.spatial50_fold != fold].reset_index(drop=True)
        test = data.loc[data.spatial50_fold == fold].reset_index(drop=True)
        # Cross-fitted training residuals calibrate a field correction without seeing outer-test targets.
        inner_groups = train.spatial50_block
        inner_splits = min(3, inner_groups.nunique())
        calibration_prediction = np.full(len(train), np.nan)
        if inner_splits >= 2:
            splitter = GroupKFold(n_splits=inner_splits)
            for inner_train, inner_cal in splitter.split(train, groups=inner_groups):
                calibration_prediction[inner_cal] = fit_predict_method(
                    "GEDI_RESIDUAL_CORRECTION", "F3",
                    train.iloc[inner_train], train.iloc[inner_cal],
                )
        else:
            # This branch is unlikely, but preserves strict separation with a deterministic split.
            split = len(train) // 2
            calibration_prediction[split:] = fit_predict_method(
                "GEDI_RESIDUAL_CORRECTION", "F3", train.iloc[:split], train.iloc[split:]
            )
            calibration_prediction[:split] = fit_predict_method(
                "GEDI_RESIDUAL_CORRECTION", "F3", train.iloc[split:], train.iloc[:split]
            )
        radius = conformal_radius(train.field_agbd, calibration_prediction, .1)
        corrected = fit_predict_method("GEDI_RESIDUAL_CORRECTION", "F3", train, test)
        field_lo, field_hi = interval_from_radius(corrected, radius)
        national_rows = prediction_lookup.loc[test.plot_id]
        for index, plot_id in enumerate(test.plot_id):
            for method, lower, upper, prediction in [
                ("GEDI_DERIVED", national_rows.iloc[index].national_pi_lower,
                 national_rows.iloc[index].national_pi_upper, test.iloc[index][BASE]),
                ("FIELD_RESIDUAL", field_lo[index], field_hi[index], corrected[index]),
            ]:
                detail.append({
                    "plot_id": plot_id, "fold": int(fold), "method": method,
                    "observed": test.iloc[index].field_agbd, "predicted": prediction,
                    "lower": lower, "upper": upper, "region": test.iloc[index].region,
                    "biomass_regime": str(regime_lookup.loc[plot_id]),
                })
    detail = pd.DataFrame(detail)
    for method, group in detail.groupby("method"):
        covered = (group.observed >= group.lower) & (group.observed <= group.upper)
        rows.append({"method": method, "grouping": "overall", "group": "ALL", "n": len(group),
                     "coverage": covered.mean(), "mean_width": (group.upper - group.lower).mean()})
        for grouping in ["region", "biomass_regime"]:
            for value, subset in group.groupby(grouping):
                covered = (subset.observed >= subset.lower) & (subset.observed <= subset.upper)
                rows.append({"method": method, "grouping": grouping, "group": value,
                             "n": len(subset), "coverage": covered.mean(),
                             "mean_width": (subset.upper - subset.lower).mean()})
    result = pd.DataFrame(rows)
    result.to_csv("outputs/tables/field_uncertainty_comparison.csv", index=False)
    return result


def nearest_gedi_matches(
    field: pd.DataFrame,
    gedi_path: Path = Path("data/raw/gedi_l4a_cote_divoire_filtered.csv"),
) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    field_x, field_y = transformer.transform(field.longitude, field.latitude)
    field_coordinates = np.column_stack([field_x, field_y])
    best_distance = np.full(len(field), np.inf)
    best_agbd = np.full(len(field), np.nan)
    best_shot = np.full(len(field), -1, dtype="int64")
    for chunk in pd.read_csv(
        gedi_path, usecols=["longitude", "latitude", "agbd", "shot_number"], chunksize=250_000
    ):
        x, y = transformer.transform(chunk.longitude, chunk.latitude)
        distance, index = cKDTree(np.column_stack([x, y])).query(field_coordinates, k=1)
        improve = distance < best_distance
        best_distance[improve] = distance[improve]
        best_agbd[improve] = chunk.agbd.to_numpy()[index[improve]]
        best_shot[improve] = chunk.shot_number.to_numpy(dtype="int64")[index[improve]]
    result = field[["plot_id", "field_agbd", BASE]].copy()
    result["nearest_gedi_distance_m"] = best_distance
    result["nearest_gedi_agbd"] = best_agbd
    result["nearest_gedi_shot_number"] = best_shot
    result["field_minus_nearest_gedi"] = result.field_agbd - result.nearest_gedi_agbd
    result["field_minus_national_prediction"] = result.field_agbd - result[BASE]
    result.to_csv("outputs/tables/field_target_nearest_gedi.csv", index=False)
    _write_target_mismatch_report(result)
    return result


def _write_target_mismatch_report(result: pd.DataFrame):
    lines = ["# Field target versus GEDI target", "", "## Matching density", ""]
    for threshold in [25, 50, 100, 250, 500, 1000]:
        lines.append(f"- Plots with a national GEDI shot within {threshold} m: "
                     f"{int((result.nearest_gedi_distance_m <= threshold).sum())}.")
    close = result[result.nearest_gedi_distance_m <= 100]
    lines.extend([
        "", "## Interpretation", "",
        (f"The median nearest-shot distance is {result.nearest_gedi_distance_m.median():.1f} m. "
         f"For the {len(close)} plots within 100 m, median field minus nearest GEDI AGBD is "
         f"{close.field_minus_nearest_gedi.median():.2f} Mg/ha and Spearman association is "
         f"{close.field_agbd.corr(close.nearest_gedi_agbd, method='spearman'):.3f}."),
        ("Nearby does not mean support-matched: plot area is absent, GEDI footprints and plot geometry "
         "differ, acquisition dates differ, and GEDI AGBD is itself model-derived. These comparisons "
         "separate possible target/support mismatch from map error descriptively; they do not validate "
         "one target as truth at the other's support."),
    ])
    Path("report/field_target_mismatch.md").write_text("\n".join(lines), encoding="utf-8")


def _stable_gain(sample_efficiency: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    baseline = sample_efficiency[sample_efficiency.method == "GEDI_NATIONAL_BASELINE"][
        ["requested_n", "fold", "repeat", "rmse"]
    ].rename(columns={"rmse": "baseline_rmse"})
    corrected = sample_efficiency[sample_efficiency.method != "GEDI_NATIONAL_BASELINE"].merge(
        baseline, on=["requested_n", "fold", "repeat"]
    )
    corrected["beats_baseline"] = corrected.rmse < corrected.baseline_rmse
    corrected["relative_gain"] = (corrected.baseline_rmse - corrected.rmse) / corrected.baseline_rmse
    stability = corrected.groupby(["requested_n", "method"], as_index=False).agg(
        mean_relative_gain=("relative_gain", "mean"),
        fraction_beating_baseline=("beats_baseline", "mean"), mean_rmse=("rmse", "mean"),
    )
    order = ["10", "20", "40", "50", "60", "100", "150", "200", "all"]
    stable = stability[
        (stability.mean_relative_gain >= .02) & (stability.fraction_beating_baseline >= .75)
    ].copy()
    stable["order"] = stable.requested_n.map({value: index for index, value in enumerate(order)})
    first = None if stable.empty else str(stable.sort_values("order").iloc[0].requested_n)
    return first, stability


def build_decision_summary(
    comparison: pd.DataFrame,
    sample_efficiency: pd.DataFrame,
    strategies: pd.DataFrame,
    regional: pd.DataFrame,
    complexity: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    first_stable, stability = _stable_gain(sample_efficiency)
    rows = []
    for method in sample_efficiency.method.unique():
        row = {"method": method}
        for n in ["20", "50", "100", "all"]:
            values = sample_efficiency[
                (sample_efficiency.method == method) & (sample_efficiency.requested_n == n)
            ]
            row[f"rmse_n_{n}"] = values.rmse.mean()
            row[f"mae_n_{n}"] = values.mae.mean()
        spatial = comparison[
            (comparison.protocol == "SPATIAL_BLOCK_CV_50KM") & (comparison.method == method)
            & (comparison.feature_variant == ("F1" if "BASELINE" in method or "CALIBRATION" in method else "F3"))
        ]
        row["all_plot_spatial_rmse"] = spatial.rmse.iloc[0] if len(spatial) else np.nan
        stable_method = stability[stability.method == method]
        row["best_fraction_beating_baseline"] = stable_method.fraction_beating_baseline.max(
        ) if len(stable_method) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    best_strategy = strategies.groupby("strategy").rmse.mean().idxmin()
    result["best_selection_strategy"] = best_strategy
    result["first_stable_gain_n"] = first_stable
    result["best_complexity_model"] = complexity.sort_values("rmse").iloc[0].model
    cross = regional[regional.evaluation == "CROSS_REGION"]
    within = regional[regional.evaluation.str.contains("WITHIN|REGION_SPECIFIC", regex=True)]
    result["mean_cross_region_rmse"] = cross.rmse.mean()
    result["mean_within_region_rmse"] = within.rmse.mean()
    result.to_csv("outputs/tables/field_learning_decision_summary.csv", index=False)
    _decision_figure(result)
    return result, first_stable


def _decision_figure(summary: pd.DataFrame):
    columns = ["rmse_n_20", "rmse_n_50", "rmse_n_100", "rmse_n_all"]
    plot = summary.set_index("method")[columns]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot.plot.bar(ax=ax)
    ax.set(ylabel="Spatially held-out RMSE (Mg/ha)", title="Field-learning decision summary")
    ax.legend(["n=20", "n=50", "n=100", "all"], fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, Path("outputs/figures/field_learning/field_learning_summary.png"))


def _write_reports(
    field: pd.DataFrame,
    comparison: pd.DataFrame,
    sample_efficiency: pd.DataFrame,
    strategies: pd.DataFrame,
    regional: pd.DataFrame,
    distances: pd.DataFrame,
    uncertainty: pd.DataFrame,
    decision: pd.DataFrame,
    first_stable: str | None,
):
    primary = comparison[
        (comparison.protocol == "SPATIAL_BLOCK_CV_50KM")
        & comparison.apply(lambda row: (row.method, row.feature_variant) in PRIMARY_CONFIGS, axis=1)
    ].sort_values("rmse")
    baseline = primary[primary.method == "GEDI_NATIONAL_BASELINE"].iloc[0]
    best = primary.iloc[0]
    strategy_summary = strategies.groupby("strategy").agg(rmse=("rmse", "mean"),
                                                            rmse_sd=("rmse", "std")).sort_values("rmse")
    best_strategy = strategy_summary.index[0]
    geographic_rho = distances.groupby("method").apply(
        lambda group: group.absolute_error.corr(group.nearest_training_geographic_km,
                                                method="spearman"), include_groups=False
    )
    feature_rho = distances.groupby("method").apply(
        lambda group: group.absolute_error.corr(group.nearest_training_feature_distance,
                                                method="spearman"), include_groups=False
    )
    field_uncertainty = uncertainty[
        (uncertainty.method == "FIELD_RESIDUAL") & (uncertainty.grouping == "overall")
    ].iloc[0]
    gedi_uncertainty = uncertainty[
        (uncertainty.method == "GEDI_DERIVED") & (uncertainty.grouping == "overall")
    ].iloc[0]
    report = f"""# Field-learning experiment

## 1. Motivation

This experiment asks whether a small, independent field sample can improve the national GEDI-trained model, how much data are needed, and whether corrections transfer spatially and ecologically.

## 2. Experimental design

There are {len(field)} plots with valid national predictions (AGBD {field.field_agbd.min():.1f}–{field.field_agbd.max():.1f} Mg/ha): {int((field.region == 'SOUTH').sum())} South, {int((field.region == 'CENTRE').sum())} Centre, and {int((field.region == 'NORTH').sum())} North. Primary evidence uses persisted 50 km spatial folds; random CV is diagnostic, 100 km blocks are a sensitivity, and each region is held out in turn. F3 was predeclared for headline curves, preventing evaluation-fold feature selection.

## 3. GEDI baseline

The unchanged national prediction has spatial-CV RMSE {baseline.rmse:.2f}, MAE {baseline.mae:.2f}, bias {baseline.bias:.2f}, and descriptive R² {baseline.r2:.3f} Mg/ha.

## 4. Field-only learning

The best primary configuration was **{best.method}** with RMSE {best.rmse:.2f} and MAE {best.mae:.2f}. Field-only models are compared with Ridge, shallow RF, and regularized HGBR in `field_model_complexity.csv`; flexible models are not assumed superior on 252 clustered plots.

## 5. Calibration and residual correction

Linear calibration, F3 residual correction, and F3 stacking were fitted only on each training fold. Their results show whether field data are best used to replace, calibrate, or locally correct the GEDI representation.

## 6. Sample efficiency

The first sample size satisfying the declared stable-gain rule (>=2% mean RMSE gain and improvement in >=75% of spatial repeats) was **{first_stable or 'not reached'}**. Each small-n estimate aggregates 20 deterministic, spatially held-out evaluations.

## 7. Strategic sampling

The lowest mean RMSE strategy was **{best_strategy}** ({strategy_summary.iloc[0].rmse:.2f} Mg/ha). Predicted-biomass stratification uses the existing national prediction, never held-out field truth.

## 8. Regional transfer

Within-region and cross-region results are in `field_regional_transfer.csv`. Local corrections should be treated as transportable only where cross-region rows remain competitive with the unchanged baseline; source-region success alone is insufficient.

## 9. Distance to calibration

Geographic-distance/error Spearman correlations range from {geographic_rho.min():.2f} to {geographic_rho.max():.2f}; feature-distance/error correlations range from {feature_rho.min():.2f} to {feature_rho.max():.2f}. These gradients are descriptive and do not establish a hard transfer radius.

## 10. Uncertainty

GEDI-derived field coverage was {gedi_uncertainty.coverage:.3f} with mean width {gedi_uncertainty.mean_width:.1f} Mg/ha. Leakage-safe field-residual coverage was {field_uncertainty.coverage:.3f} with width {field_uncertainty.mean_width:.1f}. Region and biomass-stratum coverage remain uncertain because group sizes are small.

## 11. Operational recommendation

Use the national GEDI model as the representation backbone and add field correction only where spatially held-out gains are stable. Field campaign design should follow the best leakage-safe selection strategy and deliberately improve North/Centre, OOD, and upper-predicted-biomass coverage.

## 12. Limits

Plots are geographically clustered, plot area and detailed forest-inventory metadata are absent, landcover has only seven mosaic/agriculture plots, and no ecological category is a sourced formal ecoregion. Target-support and temporal mismatch remain. Results describe this 252-plot sample and should not be converted into universal learning thresholds.
"""
    Path("report/field_learning_experiment.md").write_text(report, encoding="utf-8")
    _write_campaign(field, sample_efficiency, strategies, regional, first_stable, best_strategy)


def _write_campaign(field, sample_efficiency, strategies, regional, first_stable, best_strategy):
    counts = field.region.value_counts()
    cross = regional[regional.evaluation == "CROSS_REGION"]
    within = regional[regional.evaluation.str.contains("WITHIN|REGION_SPECIFIC", regex=True)]
    transfer_gap = cross.rmse.mean() - within.rmse.mean()
    target_n = first_stable if first_stable not in (None, "all") else "at least 100"
    additional_n = 80 if target_n == 60 else 100
    calibration_n = round(additional_n * 0.75)
    validation_n = additional_n - calibration_n
    text = f"""# Recommended field campaign

1. **Campaign size.** The current archive already contains {len(field)} valid plots. Acquire a next tranche of **{additional_n} additional plots**: approximately **{calibration_n}** for model updating and **{validation_n}** kept as an untouched validation reserve. The {target_n}-plot calibration allocation reflects the first stable learning-curve gain, not a universal threshold.
2. **Regional allocation.** Current counts are South {counts.get('SOUTH', 0)}, Centre {counts.get('CENTRE', 0)}, North {counts.get('NORTH', 0)}. As a concrete starting quota, place about 40 new plots in North, 25 in environmentally underrepresented Centre settings, and 15 in South, emphasizing under-covered conditions rather than existing clusters. Apply the 75/25 calibration/validation split within each region.
3. **OOD and uncertainty.** Use domain distance and mapped uncertainty as acquisition-time covariates. The best tested selection strategy was **{best_strategy}**; operational selection should keep geographic quotas so OOD targeting does not collapse into one cluster.
4. **High biomass.** Deliberately oversample the upper national-prediction stratum, but define selection from mapped prediction/uncertainty rather than unknown field AGBD.
5. **Spatial layout.** Prefer dispersed plots across 50–100 km blocks, with a smaller number of local replicate clusters for measurement and support-variance estimation.
6. **Transfer.** Mean cross-region RMSE exceeded the within-region summary by {transfer_gap:.2f} Mg/ha. Local corrections should not be exported without held-out regional evidence.
7. **Required metadata.** Record plot area and geometry, DBH, height, species and wood density, GPS accuracy, measurement date, allometric equation/version, disturbance history, protocol, and plot/measurement uncertainty. Preserve tree-level data and permanent plot identifiers where permitted.

Re-evaluate the learning curve after each acquisition tranche; stop or redirect sampling when spatially held-out gains plateau rather than committing to a fixed total in advance.
"""
    Path("report/field_campaign_recommendation.md").write_text(text, encoding="utf-8")


def run_field_learning_experiment() -> dict:
    for path in [Path("outputs/tables"), Path("outputs/figures/field_learning"), Path("report")]:
        path.mkdir(parents=True, exist_ok=True)
    field = build_clean_field_dataset()
    assignments = create_field_cv_assignments(field)
    comparison, predictions, data = evaluate_field_models(field, assignments)
    sample_efficiency = run_sample_efficiency(data)
    strategies = run_sampling_strategies(data)
    regional = run_regional_transfer(data)
    distances = run_distance_analysis(data)
    biomass = run_biomass_analysis(data, predictions)
    complexity = run_complexity_sensitivity(data)
    uncertainty = run_uncertainty(field, data)
    target = nearest_gedi_matches(field)
    decision, first_stable = build_decision_summary(
        comparison, sample_efficiency, strategies, regional, complexity
    )
    _write_reports(
        field, comparison, sample_efficiency, strategies, regional, distances, uncertainty,
        decision, first_stable,
    )
    summary = {
        "valid_field_plots": len(field), "regional_counts": field.region.value_counts().to_dict(),
        "biomass_range": [field.field_agbd.min(), field.field_agbd.max()],
        "primary_model_comparison": comparison[
            comparison.protocol == "SPATIAL_BLOCK_CV_50KM"
        ].to_dict("records"),
        "first_stable_gain_n": first_stable,
        "best_sampling_strategy": strategies.groupby("strategy").rmse.mean().idxmin(),
        "distance_error_spearman": {
            method: {
                "geographic": group.absolute_error.corr(
                    group.nearest_training_geographic_km, method="spearman"
                ),
                "feature": group.absolute_error.corr(
                    group.nearest_training_feature_distance, method="spearman"
                ),
            } for method, group in distances.groupby("method")
        },
        "uncertainty": uncertainty.to_dict("records"),
        "nearest_gedi_median_distance_m": target.nearest_gedi_distance_m.median(),
        "best_complexity": complexity.sort_values("rmse").iloc[0].to_dict(),
        "upper_biomass": biomass[biomass.biomass_regime == "UPPER"].to_dict("records"),
    }
    Path("outputs/metrics/field_learning_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: float(value)
                   if isinstance(value, np.floating) else int(value)
                   if isinstance(value, np.integer) else str(value)), encoding="utf-8",
    )
    return summary
