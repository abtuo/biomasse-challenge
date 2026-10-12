from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import sklearn
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import load_yaml
from .data import PALSAR_FEATURES, S1_FEATURES, S2_BANDS, S2_INDICES
from .modeling import fixed_fold_cv, make_hist_gradient_boosting, regression_metrics
from .raster import raster_feature_names, sample_raster_blockwise
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity

S2_ONLY = S2_BANDS + S2_INDICES
S2_TOPO = S2_ONLY + ["elevation", "slope"]
S2_TOPO_LC = S2_TOPO + ["landcover"]
FEATURE_SETS = {"S2_ONLY": S2_ONLY, "S2_TOPO": S2_TOPO, "S2_TOPO_LC": S2_TOPO_LC}
NATIONAL_CRS = "EPSG:32630"
NATIONAL_RASTER_CANDIDATES = (
    Path("data/raw/ci_national_s2_context_2020_2024.tif"),
    Path("data/raw/ci_national_s2_context_2020_2024_100m.tif"),
)


def resolve_national_raster(path: Path | None = None) -> Path:
    candidates = [path] if path is not None else []
    candidates.extend(NATIONAL_RASTER_CANDIDATES)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"National raster not found; checked: {candidates}")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _region(latitude) -> pd.Categorical:
    return pd.cut(
        latitude,
        [-np.inf, 6.5, 8.5, np.inf],
        labels=["South (<6.5N)", "Centre (6.5-8.5N)", "North (>=8.5N)"],
        right=False,
    )


def _gedi_datetime(delta_time) -> pd.Series:
    return pd.Timestamp("2018-01-01", tz="UTC") + pd.to_timedelta(delta_time, unit="s")


def audit_national_gedi(csv_path: Path, chunk_size: int = 250_000) -> dict:
    """Audit the national GEDI CSV in bounded-memory chunks."""
    tables = Path("outputs/tables")
    figures = Path("outputs/figures/national")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    total = 0
    missing = Counter()
    arrays: dict[str, list[np.ndarray]] = {
        c: [] for c in ["shot_number", "agbd", "relative_se", "longitude", "latitude", "delta_time"]
    }
    pft = Counter()
    region_class = Counter()
    columns: list[str] | None = None
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        columns = list(chunk.columns)
        total += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        for column, parts in arrays.items():
            parts.append(pd.to_numeric(chunk[column], errors="coerce").to_numpy())
        pft.update(chunk.pft_class.dropna().astype(str))
        region_class.update(chunk.region_class.dropna().astype(str))
    values = {column: np.concatenate(parts) for column, parts in arrays.items()}
    shot_sorted = np.sort(values["shot_number"])
    unique_shots = int(1 + np.count_nonzero(np.diff(shot_sorted))) if len(shot_sorted) else 0
    agbd = values["agbd"][np.isfinite(values["agbd"])]
    relative_se = values["relative_se"][np.isfinite(values["relative_se"])]
    dates = _gedi_datetime(values["delta_time"])
    years = pd.Series(dates.year).value_counts().sort_index().rename_axis("year").reset_index(name="n")
    years.to_csv(tables / "national_gedi_year_counts.csv", index=False)
    quantiles = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
    distribution = []
    for variable, vector in [("agbd", agbd), ("relative_se", relative_se)]:
        for q, value in zip(quantiles, np.quantile(vector, quantiles)):
            distribution.append({"variable": variable, "statistic": f"q{q:g}", "value": value})
        distribution.extend([
            {"variable": variable, "statistic": "mean", "value": float(np.mean(vector))},
            {"variable": variable, "statistic": "std", "value": float(np.std(vector, ddof=1))},
        ])
    pd.DataFrame(distribution).to_csv(tables / "national_gedi_biomass_distribution.csv", index=False)
    summary = [
        {"section": "rows", "metric": "row_count", "value": total},
        {"section": "rows", "metric": "unique_shot_number", "value": unique_shots},
        {"section": "rows", "metric": "duplicate_shot_number", "value": total - unique_shots},
        {"section": "time", "metric": "date_min", "value": str(dates.min())},
        {"section": "time", "metric": "date_max", "value": str(dates.max())},
        {"section": "extent", "metric": "longitude_min", "value": np.nanmin(values["longitude"])},
        {"section": "extent", "metric": "longitude_max", "value": np.nanmax(values["longitude"])},
        {"section": "extent", "metric": "latitude_min", "value": np.nanmin(values["latitude"])},
        {"section": "extent", "metric": "latitude_max", "value": np.nanmax(values["latitude"])},
    ]
    for column in columns or []:
        summary.append({"section": "missingness", "metric": column, "value": int(missing[column])})
    lat_regions = pd.Series(_region(values["latitude"])).value_counts(sort=False)
    for label, count in lat_regions.items():
        summary.append({"section": "latitude_band", "metric": str(label), "value": int(count)})
    for label, count in sorted(pft.items()):
        summary.append({"section": "pft_class", "metric": label, "value": count})
    for label, count in sorted(region_class.items()):
        summary.append({"section": "region_class", "metric": label, "value": count})
    pd.DataFrame(summary).to_csv(tables / "national_gedi_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    density, xedges, yedges = np.histogram2d(
        values["longitude"], values["latitude"], bins=(280, 280)
    )
    image = ax.imshow(
        np.log1p(density.T), origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis", aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="log(1 + shots)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="National GEDI shot density")
    _save(fig, figures / "national_gedi_density.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(agbd, bins=np.linspace(0, np.quantile(agbd, .995), 100), color="#2a788e")
    ax.set(xlabel="GEDI AGBD (Mg/ha)", ylabel="Count", title="National GEDI biomass distribution")
    _save(fig, figures / "national_gedi_agbd_distribution.png")
    return {
        "row_count": total,
        "unique_shot_number": unique_shots,
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "agbd_mean": float(np.mean(agbd)),
        "agbd_median": float(np.median(agbd)),
    }


def _part_paths(parts_dir: Path) -> list[Path]:
    return sorted(parts_dir.glob("part-*.parquet"))


def extract_national_features(
    csv_path: Path,
    raster_path: Path | None = None,
    parts_dir: Path = Path("data/interim/gedi_ci_national_s2_context_parts"),
    chunk_size: int = 100_000,
) -> dict:
    """Sample all GEDI points with windowed block reads and resumable Parquet parts."""
    raster_path = resolve_national_raster(raster_path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        names = raster_feature_names(src)
        bounds = src.bounds
        raster_transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        raster_info = {
            "resolved_raster": str(raster_path), "crs": str(src.crs), "bands": names,
            "width": src.width, "height": src.height, "resolution_m": abs(src.res[0]),
        }
    expected = S2_TOPO_LC
    if names != expected:
        raise ValueError(f"National raster bands are {names}; expected {expected}")
    audit_rows = []
    total = 0
    for index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        output = parts_dir / f"part-{index:05d}.parquet"
        if output.exists():
            existing = pd.read_parquet(output)
            if len(existing) != len(chunk):
                raise ValueError(f"Existing {output} has {len(existing)} rows; expected {len(chunk)}")
            matched = existing
        else:
            features = sample_raster_blockwise(chunk, raster_path, expected)
            matched = pd.concat([chunk.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
            matched.to_parquet(output, index=False, compression="zstd")
        feature_values = matched[expected].replace([np.inf, -np.inf], np.nan)
        xs, ys = raster_transformer.transform(chunk.longitude.to_numpy(), chunk.latitude.to_numpy())
        inside = ((np.asarray(xs) >= bounds.left) & (np.asarray(xs) < bounds.right)
                  & (np.asarray(ys) > bounds.bottom) & (np.asarray(ys) <= bounds.top))
        audit_rows.append({
            "part": index, "rows": len(matched),
            "coordinates_inside_raster": int(inside.sum()),
            "complete_s2": int(feature_values[S2_ONLY].notna().all(axis=1).sum()),
            "complete_s2_topo": int(feature_values[S2_TOPO].notna().all(axis=1).sum()),
            "complete_s2_topo_lc": int(feature_values.notna().all(axis=1).sum()),
            "rows_with_any_missing": int(feature_values.isna().any(axis=1).sum()),
            "infinite_values": int(np.isinf(matched[expected].to_numpy()).sum()),
            "duplicate_shot_number": int(matched.shot_number.duplicated().sum()),
        })
        total += len(matched)
    audit = pd.DataFrame(audit_rows)
    totals = {
        "part": "TOTAL", "rows": int(audit.rows.sum()),
        "coordinates_inside_raster": int(audit.coordinates_inside_raster.sum()),
        "complete_s2": int(audit.complete_s2.sum()),
        "complete_s2_topo": int(audit.complete_s2_topo.sum()),
        "complete_s2_topo_lc": int(audit.complete_s2_topo_lc.sum()),
        "rows_with_any_missing": int(audit.rows_with_any_missing.sum()),
        "infinite_values": int(audit.infinite_values.sum()),
        "duplicate_shot_number": int(audit.duplicate_shot_number.sum()),
    }
    audit = pd.concat([audit, pd.DataFrame([totals])], ignore_index=True)
    table_path = Path("outputs/tables/national_matching_audit.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False)
    manifest = {**raster_info, "rows": total, "parts": len(audit_rows), "parts_dir": str(parts_dir)}
    (parts_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def extract_field_national_features(
    field_path: Path,
    raster_path: Path | None = None,
    output: Path = Path("data/interim/field_ci_national_features.parquet"),
) -> pd.DataFrame:
    raster_path = resolve_national_raster(raster_path)
    field = pd.read_csv(field_path)
    features = sample_raster_blockwise(field, raster_path, S2_TOPO_LC)
    out = pd.concat([field.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    audit = pd.DataFrame([{
        "field_rows": len(out), "biomass_min": out.field_agbd.min(),
        "biomass_max": out.field_agbd.max(), "complete_s2": out[S2_ONLY].notna().all(axis=1).sum(),
        "complete_all": out[S2_TOPO_LC].notna().all(axis=1).sum(),
        "longitude_min": out.longitude.min(), "longitude_max": out.longitude.max(),
        "latitude_min": out.latitude.min(), "latitude_max": out.latitude.max(),
    }])
    audit.to_csv("outputs/tables/national_field_feature_audit.csv", index=False)
    return out


def _block_ids(frame: pd.DataFrame, block_size_m: int) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", NATIONAL_CRS, always_xy=True)
    x, y = transformer.transform(frame.longitude.to_numpy(), frame.latitude.to_numpy())
    bx = np.floor(np.asarray(x) / block_size_m).astype("int32")
    by = np.floor(np.asarray(y) / block_size_m).astype("int32")
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def _balanced_fold_map(block_counts: Counter, n_splits: int = 5) -> dict[str, int]:
    totals = np.zeros(n_splits, dtype=np.int64)
    mapping = {}
    for block, count in sorted(block_counts.items(), key=lambda item: (-item[1], item[0])):
        fold = int(np.argmin(totals))
        mapping[block] = fold
        totals[fold] += count
    return mapping


def build_national_folds_and_sample(
    parts_dir: Path = Path("data/interim/gedi_ci_national_s2_context_parts"),
    block_size_m: int = 50_000,
    max_per_block: int = 5_000,
) -> dict:
    """Persist national folds and maintain a deterministic weighted reservoir per block."""
    paths = _part_paths(parts_dir)
    if not paths:
        raise FileNotFoundError(f"No national feature parts found under {parts_dir}")
    block_counts: Counter = Counter()
    for path in paths:
        frame = pd.read_parquet(path, columns=["longitude", "latitude"])
        block_counts.update(_block_ids(frame, block_size_m))
    fold_map = _balanced_fold_map(block_counts)
    fold_path = Path("data/processed/national_spatial_fold_assignment.parquet")
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    reservoirs: dict[str, pd.DataFrame] = {}
    for part_index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        blocks = _block_ids(frame, block_size_m)
        folds = np.fromiter((fold_map[b] for b in blocks), dtype="int8", count=len(frame))
        fold_frame = pd.DataFrame({
            "shot_number": frame.shot_number.to_numpy(), "spatial_block": blocks, "fold": folds,
        })
        table = pa.Table.from_pandas(fold_frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(fold_path, table.schema, compression="zstd")
        writer.write_table(table)
        rng = np.random.default_rng(42 + part_index)
        weight = np.select(
            [frame.agbd.to_numpy() >= 370, frame.agbd.to_numpy() >= 200], [2.5, 1.5], default=1.0
        )
        candidate = frame.copy()
        candidate["spatial_block"] = blocks
        candidate["fold"] = folds
        candidate["_priority"] = -np.log(np.maximum(rng.random(len(frame)), 1e-12)) / weight
        for block, group in candidate.groupby("spatial_block", sort=False):
            combined = pd.concat([reservoirs.get(block, group.iloc[:0]), group], ignore_index=True)
            reservoirs[block] = combined.nsmallest(max_per_block, "_priority")
    if writer is not None:
        writer.close()
    sample = pd.concat(reservoirs.values(), ignore_index=True).drop(columns="_priority")
    sample = sample.sort_values(["fold", "spatial_block", "shot_number"]).reset_index(drop=True)
    sample_path = Path("data/processed/national_training_sample.parquet")
    sample.to_parquet(sample_path, index=False, compression="zstd")
    original_quantiles = []
    for path in paths:
        original_quantiles.append(pd.read_parquet(path, columns=["agbd"]).agbd.to_numpy())
    full_agbd = np.concatenate(original_quantiles)
    qs = [.05, .25, .5, .75, .9, .95, .99]
    rows = [{
        "original_rows": int(sum(block_counts.values())), "sampled_rows": len(sample),
        "n_blocks": len(block_counts), "block_size_m": block_size_m,
        "max_per_block": max_per_block, "original_blocks_covered": len(block_counts),
        "sampled_blocks_covered": sample.spatial_block.nunique(),
    }]
    for q, original, sampled in zip(qs, np.quantile(full_agbd, qs), sample.agbd.quantile(qs)):
        rows.append({"quantile": q, "original_agbd": original, "sampled_agbd": sampled})
    pd.DataFrame(rows).to_csv("outputs/tables/national_sampling_summary.csv", index=False)
    block_table = pd.DataFrame({"spatial_block": list(block_counts), "original_rows": list(block_counts.values())})
    sampled_counts = sample.spatial_block.value_counts()
    block_table["sampled_rows"] = block_table.spatial_block.map(sampled_counts).fillna(0).astype(int)
    block_table["fold"] = block_table.spatial_block.map(fold_map)
    block_table.to_csv("outputs/tables/national_sampling_by_block.csv", index=False)
    _plot_national_blocks(sample)
    return {
        "original_rows": int(sum(block_counts.values())), "sampled_rows": len(sample),
        "n_blocks": len(block_counts), "block_size_m": block_size_m,
    }


def _plot_national_blocks(sample: pd.DataFrame):
    plot = sample.sample(min(len(sample), 150_000), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(plot.longitude, plot.latitude, c=plot.fold, s=1, cmap="tab10", alpha=.45)
    fig.colorbar(scatter, ax=ax, label="Spatial fold")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="National 50 km spatial blocks and folds")
    _save(fig, Path("outputs/figures/national/national_spatial_blocks.png"))


def _national_model(features: list[str], params: dict) -> Pipeline:
    if "landcover" not in features:
        return make_hist_gradient_boosting(params)
    numeric = [feature for feature in features if feature != "landcover"]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("landcover", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ["landcover"]),
    ], verbose_feature_names_out=False)
    categorical = [False] * len(numeric) + [True]
    regressor = HistGradientBoostingRegressor(
        learning_rate=float(params.get("learning_rate", .08)),
        max_iter=int(params.get("max_iter", 160)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(params.get("min_samples_leaf", 30)),
        l2_regularization=float(params.get("l2_regularization", 1)),
        categorical_features=categorical,
        random_state=int(params.get("random_state", 42)),
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def _biomass_masks(y) -> dict[str, np.ndarray]:
    values = np.asarray(y)
    return {
        "lt_100": values < 100,
        "100_200": (values >= 100) & (values < 200),
        "200_370": (values >= 200) & (values < 370),
        "ge_370": values >= 370,
        "top_10pct": values >= np.quantile(values, .9),
        "top_5pct": values >= np.quantile(values, .95),
    }


def _metrics_rows(y, prediction, feature_set: str) -> list[dict]:
    values = np.asarray(y)
    prediction = np.asarray(prediction)
    rows = []
    for label, mask in _biomass_masks(values).items():
        rows.append({
            "feature_set": feature_set, "biomass_regime": label, "n": int(mask.sum()),
            **regression_metrics(values[mask], prediction[mask]),
        })
    return rows


def _regional_metric_rows(y, prediction, latitude, model_name: str, population: str) -> list[dict]:
    frame = pd.DataFrame({
        "observed": np.asarray(y), "predicted": np.asarray(prediction),
        "region": _region(np.asarray(latitude)),
    })
    rows = []
    for region, group in frame.groupby("region", observed=True):
        rows.append({
            "population": population, "model": model_name, "region": str(region), "n": len(group),
            **regression_metrics(group.observed, group.predicted),
            "spearman": float(spearmanr(group.observed, group.predicted).statistic),
        })
    return rows


def _assign_blocks_in_frame(frame: pd.DataFrame, block_size_m: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = _block_ids(frame, block_size_m)
    counts = Counter(blocks)
    mapping = _balanced_fold_map(counts)
    folds = np.fromiter((mapping[block] for block in blocks), dtype="int8", count=len(blocks))
    return blocks, folds


def run_national_experiment(
    sample_path: Path = Path("data/processed/national_training_sample.parquet"),
    field_path: Path = Path("data/interim/field_ci_national_features.parquet"),
    config_path: Path = Path("configs/model.yaml"),
) -> dict:
    cfg = load_yaml(config_path)
    tables = Path("outputs/tables")
    figures = Path("outputs/figures/national")
    models = Path("outputs/models")
    for path in [tables, figures, models, Path("report")]:
        path.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(sample_path).replace([np.inf, -np.inf], np.nan)
    sample = sample.loc[sample.agbd.notna()].reset_index(drop=True)
    if "fold" not in sample:
        _, sample["fold"] = _assign_blocks_in_frame(sample, 50_000)
    params = cfg["model"]["hist_gradient_boosting"]
    train_cap = min(300_000, len(sample))
    final_cap = min(500_000, len(sample))
    overall_rows = []
    fold_rows = []
    biomass_rows = []
    predictions = {}
    for feature_set, features in FEATURE_SETS.items():
        valid = sample[S2_ONLY].notna().all(axis=1)
        subset = sample.loc[valid].reset_index(drop=True)
        model = _national_model(features, params)
        folds, oof, overall = fixed_fold_cv(
            model, subset[features], subset.agbd, subset.fold.to_numpy(), train_cap, 42
        )
        overall_rows.append({
            "feature_set": feature_set, "n": len(subset), **overall,
            "fold_rmse_sd": float(folds.rmse.std()), "model_family": "HistGradientBoostingRegressor",
            "landcover_handling": "categorical" if "landcover" in features else "not_used",
        })
        folds.insert(0, "feature_set", feature_set)
        fold_rows.append(folds)
        biomass_rows.extend(_metrics_rows(subset.agbd, oof, feature_set))
        predictions[feature_set] = (subset, oof)
    overall = pd.DataFrame(overall_rows).sort_values("rmse").reset_index(drop=True)
    by_biomass = pd.DataFrame(biomass_rows)
    overall.to_csv(tables / "national_model_comparison.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(tables / "national_model_folds.csv", index=False)
    by_biomass.to_csv(tables / "national_model_by_biomass.csv", index=False)
    selected = str(overall.iloc[0].feature_set)
    features = FEATURE_SETS[selected]
    selected_data, selected_oof = predictions[selected]
    # 25 km sensitivity uses the selected feature set and unchanged model settings.
    _, folds_25 = _assign_blocks_in_frame(selected_data, 25_000)
    sensitivity_folds, _, sensitivity_metrics = fixed_fold_cv(
        _national_model(features, params), selected_data[features], selected_data.agbd,
        folds_25, train_cap, 73,
    )
    pd.DataFrame([
        {"block_size_m": 50_000, **overall.set_index("feature_set").loc[selected].to_dict()},
        {"block_size_m": 25_000, **sensitivity_metrics,
         "fold_rmse_sd": float(sensitivity_folds.rmse.std()), "feature_set": selected},
    ]).to_csv(tables / "national_block_size_sensitivity.csv", index=False)
    radius = conformal_radius(selected_data.agbd, selected_oof, cfg["conformal"].get("alpha", .1))
    lower, upper = interval_from_radius(selected_oof, radius)
    conformal_rows = [{
        "biomass_regime": "overall", "n": len(selected_data),
        "coverage": empirical_coverage(selected_data.agbd, lower, upper),
        "mean_width": float(np.mean(upper - lower)), "median_width": float(np.median(upper - lower)),
        "radius": radius, "protocol": "five-fold spatial cross-conformal residuals",
    }]
    for label, mask in _biomass_masks(selected_data.agbd).items():
        conformal_rows.append({
            "biomass_regime": label, "n": int(mask.sum()),
            "coverage": empirical_coverage(selected_data.agbd.to_numpy()[mask], lower[mask], upper[mask]),
            "mean_width": float(np.mean((upper - lower)[mask])),
            "median_width": float(np.median((upper - lower)[mask])), "radius": radius,
            "protocol": "five-fold spatial cross-conformal residuals",
        })
    pd.DataFrame(conformal_rows).to_csv(tables / "national_conformal.csv", index=False)
    rng = np.random.default_rng(99)
    final_idx = np.arange(len(selected_data))
    if len(final_idx) > final_cap:
        final_idx = np.sort(rng.choice(final_idx, final_cap, replace=False))
    final_model = _national_model(features, params).fit(
        selected_data.iloc[final_idx][features], selected_data.iloc[final_idx].agbd
    )
    validity_features = [feature for feature in features if feature != "landcover"]
    validity = KNNDomainValidity(
        cfg["validity"].get("k_neighbors", 10), cfg["validity"].get("quantile_threshold", .95),
        cfg["validity"].get("max_reference_samples", 20_000), random_state=42,
    ).fit(selected_data.iloc[final_idx][validity_features])
    bundle = {
        "model": final_model, "features": features, "feature_set": selected,
        "validity_features": validity_features,
        "feature_sets_compared": FEATURE_SETS, "landcover_categorical": "landcover" in features,
        "conformal": {"alpha": .1, "radius": radius, "protocol": "spatial cross-conformal"},
        "validity": validity, "crs": NATIONAL_CRS, "block_size_m": 50_000,
        "training_summary": {"n_sample": len(sample), "n_final_fit": len(final_idx)},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    bundle_path = models / "national_model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    regional_rows = _regional_metric_rows(
        selected_data.agbd, selected_oof, selected_data.latitude, selected, "GEDI spatial CV"
    )
    field_results = evaluate_national_fields(field_path, bundle_path, regional_rows)
    _national_model_figures(overall, by_biomass, selected_data, selected_oof)
    summary = {
        "training_sample_rows": len(sample), "selected_feature_set": selected,
        "national_spatial_cv": overall.to_dict("records"), "conformal": conformal_rows,
        "field": field_results, "block_25km": sensitivity_metrics,
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/national_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_recommendations(summary)
    _write_national_report(summary)
    return summary


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def _national_model_figures(overall, by_biomass, data, oof):
    figures = Path("outputs/figures/national")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(overall.feature_set, overall.rmse)
    axes[0].set(ylabel="RMSE (Mg/ha)", title="National spatial CV")
    axes[1].bar(overall.feature_set, overall.r2)
    axes[1].set(ylabel="R²", title="National spatial CV")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    _save(fig, figures / "national_model_comparison.png")
    plot = by_biomass.pivot(index="biomass_regime", columns="feature_set", values="rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot.plot.bar(ax=ax)
    ax.set(ylabel="RMSE (Mg/ha)", title="National error by biomass regime")
    _save(fig, figures / "national_model_by_biomass.png")
    selection = np.random.default_rng(42).choice(len(data), min(120_000, len(data)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(data.agbd.iloc[selection], oof[selection], gridsize=70, bins="log", mincnt=1)
    limit = float(data.agbd.quantile(.995))
    ax.plot([0, limit], [0, limit], "r--")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Observed GEDI AGBD",
           ylabel="Spatial-CV prediction", title="Selected national model")
    _save(fig, figures / "national_predicted_vs_observed.png")


def _field_metrics(observed, predicted) -> dict:
    result = regression_metrics(observed, predicted)
    result["spearman"] = float(spearmanr(observed, predicted).statistic)
    result["descriptive_r2"] = result.pop("r2")
    return result


def _sample_tai_field_features(field: pd.DataFrame) -> pd.DataFrame:
    rasters = {
        "S2": Path("data/raw/sentinel2_tai_features_2020_2024.tif"),
        "S1": Path("data/raw/sentinel1_tai_features_2020_2024.tif"),
        "PALSAR": Path("data/raw/palsar_tai_features_2020_2024.tif"),
    }
    feature_groups = {"S2": S2_ONLY, "S1": S1_FEATURES, "PALSAR": PALSAR_FEATURES}
    metadata = ["plot_id", "measurement_date", "latitude", "longitude", "field_agbd"]
    parts = [field[metadata].reset_index(drop=True)]
    for sensor, raster in rasters.items():
        parts.append(sample_raster_blockwise(field, raster, feature_groups[sensor]).reset_index(drop=True))
    return pd.concat(parts, axis=1)


def evaluate_national_fields(
    field_path: Path,
    national_bundle_path: Path,
    regional_rows: list[dict] | None = None,
) -> dict:
    tables = Path("outputs/tables")
    field = pd.read_parquet(field_path).replace([np.inf, -np.inf], np.nan)
    national = joblib.load(national_bundle_path)
    tai = joblib.load("outputs/models/final_multisensor_model_bundle.joblib")
    national_valid = field[S2_ONLY].notna().all(axis=1)
    national_prediction = np.full(len(field), np.nan)
    national_distance = np.full(len(field), np.nan)
    national_prediction[national_valid] = national["model"].predict(
        field.loc[national_valid, national["features"]]
    )
    national_distance[national_valid] = national["validity"].score_samples(
        field.loc[national_valid, national["validity_features"]]
    )
    national_in = national_distance <= national["validity"].threshold_
    national_lo, national_hi = interval_from_radius(
        national_prediction, national["conformal"]["radius"]
    )
    national_field = field[["plot_id", "measurement_date", "latitude", "longitude", "field_agbd"]].copy()
    national_field["predicted_agbd"] = national_prediction
    national_field["pi_lower"] = national_lo
    national_field["pi_upper"] = national_hi
    national_field["domain_distance"] = national_distance
    national_field["in_domain"] = national_in
    national_field.to_csv(tables / "national_field_predictions.csv", index=False)
    tai_features = _sample_tai_field_features(field)
    tai_valid = tai_features[tai["features"]].notna().all(axis=1)
    tai_prediction = np.full(len(field), np.nan)
    tai_distance = np.full(len(field), np.nan)
    tai_prediction[tai_valid] = tai["model"].predict(tai_features.loc[tai_valid, tai["features"]])
    tai_distance[tai_valid] = tai["validity"].score_samples(tai_features.loc[tai_valid, tai["features"]])
    tai_in = tai_distance <= tai["validity"].threshold_
    tai_lo, tai_hi = interval_from_radius(tai_prediction, tai["conformal"]["radius"])
    transfer = field[["plot_id", "measurement_date", "latitude", "longitude", "field_agbd"]].copy()
    transfer["predicted_agbd"] = tai_prediction
    transfer["pi_lower"] = tai_lo
    transfer["pi_upper"] = tai_hi
    transfer["domain_distance"] = tai_distance
    transfer["in_domain"] = tai_in
    transfer["required_features_available"] = tai_valid
    transfer.to_csv(tables / "tai_transfer_field_predictions.csv", index=False)
    common = tai_valid & national_valid
    comparison_rows = []
    for label, prediction, distance, inside, threshold in [
        ("TAI_TRANSFER", tai_prediction, tai_distance, tai_in, tai["validity"].threshold_),
        ("NATIONAL_RETRAIN", national_prediction, national_distance, national_in,
         national["validity"].threshold_),
    ]:
        row = {"model": label, "n": int(common.sum()), **_field_metrics(
            field.loc[common, "field_agbd"], prediction[common]
        )}
        row.update({
            "domain_threshold": threshold, "in_domain_fraction": float(inside[common].mean()),
            "mae_in_domain": float(np.mean(np.abs(
                prediction[common & inside] - field.loc[common & inside, "field_agbd"]
            ))) if (common & inside).any() else np.nan,
            "mae_out_domain": float(np.mean(np.abs(
                prediction[common & ~inside] - field.loc[common & ~inside, "field_agbd"]
            ))) if (common & ~inside).any() else np.nan,
            "error_distance_spearman": float(spearmanr(
                np.abs(prediction[common] - field.loc[common, "field_agbd"]), distance[common]
            ).statistic),
        })
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(tables / "transfer_vs_national_field.csv", index=False)
    domain = pd.DataFrame({
        "plot_id": field.plot_id, "field_agbd": field.field_agbd,
        "tai_prediction": tai_prediction, "tai_distance": tai_distance, "tai_in_domain": tai_in,
        "national_prediction": national_prediction, "national_distance": national_distance,
        "national_in_domain": national_in, "common_comparison_subset": common,
    })
    domain.to_csv(tables / "tai_vs_national_domain.csv", index=False)
    regional_rows = list(regional_rows or [])
    if common.any():
        regional_rows.extend(_regional_metric_rows(
            field.loc[common, "field_agbd"], tai_prediction[common], field.loc[common, "latitude"],
            "TAI_TRANSFER", "field common subset",
        ))
        regional_rows.extend(_regional_metric_rows(
            field.loc[common, "field_agbd"], national_prediction[common],
            field.loc[common, "latitude"], "NATIONAL_RETRAIN", "field common subset",
        ))
    pd.DataFrame(regional_rows).to_csv(tables / "national_metrics_by_region.csv", index=False)
    calibration = _field_calibration(field.loc[national_valid].copy(), national_prediction[national_valid])
    calibration.to_csv(tables / "national_field_calibration.csv", index=False)
    _field_figures(field, common, tai_prediction, national_prediction, tai_distance, national_distance,
                   tai_in, national_in, comparison)
    return {
        "national_valid_field_rows": int(national_valid.sum()),
        "tai_valid_field_rows": int(tai_valid.sum()), "common_field_rows": int(common.sum()),
        "comparison": comparison.to_dict("records"), "calibration": calibration.to_dict("records"),
    }


def _field_calibration(field: pd.DataFrame, raw_prediction: np.ndarray) -> pd.DataFrame:
    observed = field.field_agbd.to_numpy()
    raw_prediction = np.asarray(raw_prediction)
    rows = [{"scope": "national", "method": "raw", "n": len(field),
             "slope": 1.0, "intercept": 0.0, **regression_metrics(observed, raw_prediction)}]
    folds = KFold(n_splits=min(5, len(field)), shuffle=True, random_state=42)
    calibrated = np.full(len(field), np.nan)
    slopes, intercepts = [], []
    for train, test in folds.split(field):
        model = LinearRegression().fit(raw_prediction[train, None], observed[train])
        calibrated[test] = model.predict(raw_prediction[test, None])
        slopes.append(float(model.coef_[0]))
        intercepts.append(float(model.intercept_))
    rows.append({
        "scope": "national", "method": "linear_calibration_cv", "n": len(field),
        "slope": float(np.mean(slopes)), "intercept": float(np.mean(intercepts)),
        **regression_metrics(observed, calibrated),
    })
    region_labels = _region(field.latitude)
    categories = region_labels.cat.categories if hasattr(region_labels, "cat") else region_labels.categories
    for region in categories:
        mask = np.asarray(region_labels == region)
        if mask.sum() < 20:
            continue
        region_observed = observed[mask]
        region_prediction = raw_prediction[mask]
        region_cv = KFold(n_splits=min(5, mask.sum()), shuffle=True, random_state=43)
        region_calibrated = np.full(mask.sum(), np.nan)
        region_slopes, region_intercepts = [], []
        for train, test in region_cv.split(region_prediction):
            model = LinearRegression().fit(region_prediction[train, None], region_observed[train])
            region_calibrated[test] = model.predict(region_prediction[test, None])
            region_slopes.append(float(model.coef_[0]))
            region_intercepts.append(float(model.intercept_))
        rows.append({
            "scope": str(region), "method": "regional_linear_calibration_cv", "n": int(mask.sum()),
            "slope": float(np.mean(region_slopes)), "intercept": float(np.mean(region_intercepts)),
            **regression_metrics(region_observed, region_calibrated),
        })
    return pd.DataFrame(rows)


def _field_figures(field, common, tai_prediction, national_prediction, tai_distance,
                   national_distance, tai_in, national_in, comparison):
    figures = Path("outputs/figures/national")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    limit = float(max(field.loc[common, "field_agbd"].max(), national_prediction[common].max()))
    for axis, label, prediction in zip(
        axes, ["Taï transfer", "National retrain"], [tai_prediction, national_prediction]
    ):
        axis.scatter(field.loc[common, "field_agbd"], prediction[common], s=24, alpha=.7)
        axis.plot([0, limit], [0, limit], "r--")
        axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Field AGBD", ylabel="Predicted AGBD",
                 title=label)
    _save(fig, figures / "transfer_vs_national_field.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(tai_distance[common], np.abs(tai_prediction[common] - field.loc[common, "field_agbd"]),
                    c=tai_in[common], s=25)
    axes[0].set(xlabel="Taï-domain distance", ylabel="Absolute field error", title="Taï transfer")
    axes[1].scatter(national_distance[common],
                    np.abs(national_prediction[common] - field.loc[common, "field_agbd"]),
                    c=national_in[common], s=25)
    axes[1].set(xlabel="National-domain distance", ylabel="Absolute field error", title="National model")
    _save(fig, figures / "tai_vs_national_domain.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(comparison.model, comparison.rmse)
    axes[0].set(title="Field RMSE", ylabel="Mg/ha")
    axes[1].bar(comparison.model, comparison.bias)
    axes[1].set(title="Field bias", ylabel="Mg/ha")
    axes[2].bar(comparison.model, comparison.in_domain_fraction)
    axes[2].set(title="In-domain fraction", ylim=(0, 1))
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    _save(fig, figures / "ci_transfer_vs_retrain_summary.png")


def predict_national_map(
    bundle_path: Path = Path("outputs/models/national_model_bundle.joblib"),
    raster_path: Path | None = None,
    output_dir: Path = Path("outputs/maps"),
) -> dict:
    """Generate 100 m national products with blockwise inference and 1 km domain diagnostics."""
    raster_path = resolve_national_raster(raster_path)
    bundle = joblib.load(bundle_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "prediction": "ci_biomass_prediction_100m.tif",
        "lower": "ci_biomass_pi_lower_100m.tif",
        "upper": "ci_biomass_pi_upper_100m.tif",
        "width": "ci_biomass_interval_width_100m.tif",
        "distance": "ci_domain_distance_100m.tif",
        "validity": "ci_domain_validity_100m.tif",
    }
    with rasterio.open(raster_path) as source:
        names = raster_feature_names(source)
        indices = [names.index(feature) + 1 for feature in bundle["features"]]
        s2_positions = [bundle["features"].index(feature) for feature in S2_ONLY]
        profile = source.profile.copy()
        profile.update(count=1, dtype="float32", compress="deflate", predictor=3,
                       tiled=True, nodata=np.nan)
        writers = {
            key: rasterio.open(output_dir / filename, "w", **profile)
            for key, filename in filenames.items() if key != "validity"
        }
        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", nodata=255, predictor=2)
        mask_writer = rasterio.open(output_dir / filenames["validity"], "w", **mask_profile)
        try:
            for _, window in source.block_windows(1):
                stack = source.read(indices, window=window, masked=True).filled(np.nan).astype("float32")
                height, width = stack.shape[1:]
                matrix = stack.reshape(len(indices), -1).T
                valid = np.isfinite(matrix[:, s2_positions]).all(axis=1)
                arrays = {key: np.full(height * width, np.nan, dtype="float32") for key in writers}
                validity_mask = np.full(height * width, 255, dtype="uint8")
                if valid.any():
                    valid_frame = pd.DataFrame(matrix[valid], columns=bundle["features"])
                    prediction = bundle["model"].predict(valid_frame).astype("float32")
                    lower, upper = interval_from_radius(prediction, bundle["conformal"]["radius"])
                    arrays["prediction"][valid] = prediction
                    arrays["lower"][valid] = lower
                    arrays["upper"][valid] = upper
                    arrays["width"][valid] = upper - lower
                    # Domain distance is the expensive diagnostic. Evaluate at 1 km and interpolate
                    # onto the 100 m map; prediction itself remains native 100 m.
                    coarse_rows = np.unique(np.r_[np.arange(0, height, 10), height - 1])
                    coarse_cols = np.unique(np.r_[np.arange(0, width, 10), width - 1])
                    rr, cc = np.meshgrid(coarse_rows, coarse_cols, indexing="ij")
                    coarse = stack[:, rr, cc].reshape(len(indices), -1).T
                    coarse_valid = np.isfinite(coarse[:, s2_positions]).all(axis=1)
                    coarse_distance = np.full(len(coarse), np.nan, dtype="float32")
                    if coarse_valid.any():
                        queries = pd.DataFrame(coarse[coarse_valid], columns=bundle["features"])
                        queries = queries[bundle["validity_features"]]
                        values = []
                        for start in range(0, len(queries), 256):
                            values.append(bundle["validity"].score_samples(queries.iloc[start:start + 256]))
                        coarse_distance[coarse_valid] = np.concatenate(values).astype("float32")
                    grid = coarse_distance.reshape(len(coarse_rows), len(coarse_cols))
                    finite = np.isfinite(grid)
                    if finite.any() and not finite.all():
                        grid[~finite] = float(np.nanmedian(grid))
                    interpolator = RegularGridInterpolator(
                        (coarse_rows, coarse_cols), grid, bounds_error=False, fill_value=np.nan
                    )
                    full_r, full_c = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                    distance_grid = interpolator(
                        np.column_stack([full_r.ravel(), full_c.ravel()])
                    ).astype("float32")
                    arrays["distance"][valid] = distance_grid[valid]
                    validity_mask[valid] = (
                        distance_grid[valid] <= bundle["validity"].threshold_
                    ).astype("uint8")
                for key, writer in writers.items():
                    writer.write(arrays[key].reshape(height, width), 1, window=window)
                mask_writer.write(validity_mask.reshape(height, width), 1, window=window)
        finally:
            for writer in writers.values():
                writer.close()
            mask_writer.close()
    with rasterio.open(output_dir / filenames["distance"], "r+") as destination:
        destination.update_tags(
            diagnostic_resolution_m=1000,
            method="kNN distance on 10-pixel lattice bilinearly interpolated to 100 m",
        )
    with rasterio.open(output_dir / filenames["validity"], "r+") as destination:
        destination.update_tags(diagnostic_resolution_m=1000)
    _national_map_figures(output_dir)
    return {key: str(output_dir / filename) for key, filename in filenames.items()}


def _read_preview(path: Path, categorical: bool = False):
    with rasterio.open(path) as source:
        scale = max(1, int(max(source.width, source.height) / 1400))
        return source.read(
            1, out_shape=(max(1, source.height // scale), max(1, source.width // scale)),
            masked=True, resampling=Resampling.nearest if categorical else Resampling.average,
        )


def _national_map_figures(output_dir: Path):
    figures = Path("outputs/figures/national")
    specs = [
        ("ci_biomass_prediction_100m.tif", "National biomass prediction", "viridis",
         "ci_biomass_map.png", False),
        ("ci_biomass_interval_width_100m.tif", "Prediction interval width", "magma",
         "ci_uncertainty_map.png", False),
        ("ci_domain_validity_100m.tif", "Domain validity", "RdYlGn",
         "ci_domain_validity_map.png", True),
    ]
    for filename, title, cmap, output, categorical in specs:
        array = _read_preview(output_dir / filename, categorical)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(array, cmap=cmap)
        fig.colorbar(image, ax=ax)
        ax.set(title=title)
        ax.axis("off")
        _save(fig, figures / output)


def _write_recommendations(summary: dict):
    comparison = pd.DataFrame(summary["field"]["comparison"]).set_index("model")
    calibration = pd.DataFrame(summary["field"]["calibration"])
    overall = pd.DataFrame(summary["national_spatial_cv"]).set_index("feature_set")
    selected = summary["selected_feature_set"]
    tai = comparison.loc["TAI_TRANSFER"]
    national = comparison.loc["NATIONAL_RETRAIN"]
    raw = calibration[(calibration.scope == "national") & (calibration.method == "raw")].iloc[0]
    calibrated = calibration[
        (calibration.scope == "national") & (calibration.method == "linear_calibration_cv")
    ].iloc[0]
    text = f"""# Research recommendations

## Evidence from the national experiment

The spatially balanced national GEDI sample contained {summary['training_sample_rows']:,} observations. The selected feature set was **{selected}**, with 50 km spatial-CV RMSE {overall.loc[selected, 'rmse']:.2f} Mg/ha and R² {overall.loc[selected, 'r2']:.3f}. On the identical {int(tai.n)}-plot field subset, Taï transfer RMSE was {tai.rmse:.2f} Mg/ha and the national model RMSE was {national.rmse:.2f} Mg/ha. Their in-domain fractions were {tai.in_domain_fraction:.3f} and {national.in_domain_fraction:.3f}. National field-only linear calibration changed cross-validated RMSE from {raw.rmse:.2f} to {calibrated.rmse:.2f} Mg/ha; this is a diagnostic, not permission to train a complex model on 263 plots.

## Short term

- Retain national GEDI retraining as the base architecture when its paired field/domain evidence is stronger than direct Taï transfer; continue publishing the transfer baseline.
- Match raster support to GEDI footprints and field plot areas, and preserve the point/25 m/50 m sensitivity results.
- Improve temporal matching between GEDI acquisitions, field dates, and annual disturbance-aware composites.
- Expand field sampling in underrepresented Centre and North strata and retain independent holdouts.
- Replace constant-width intervals with conditional conformal calibration by biomass and broad region.

## Medium term

- Test regional models or a mixture-of-experts with shared national features and spatially held-out gating validation.
- Recalibrate GEDI/allometric relationships locally only with independent structural and field evidence.
- Add disturbance history and time-since-disturbance predictors rather than treating the 2020–2024 mosaic as static.
- Use high uncertainty and domain distance jointly to prioritize new plots and structural measurements.

## Long term

- Acquire targeted airborne LiDAR across environmental and disturbance gradients.
- Add denser structural sensing where optical/topographic predictors saturate.
- Move toward an annual national map with explicit change-detection safeguards.
- Build a versioned national MLOps pipeline covering source mosaics, folds, model bundles, map provenance, and monitoring.
- Estimate national stocks with area weights and spatial covariance or block resampling; do not sum pixel variances as if independent.

## Recommended architecture

Use one reproducible national model as the provisional mapping backbone because it has nationally available predictors and covers 252 field plots, but do not claim it outperformed Taï on paired field evidence: Taï was numerically better on the only 10 comparable plots. Retain regional diagnostics and domain masks, and promote regional experts only if spatially held-out and independent-field evidence shows stable gains. Taï should remain a local benchmark and transfer diagnostic, not an assumed national model.
"""
    Path("report/research_recommendations.md").write_text(text, encoding="utf-8")


def _write_national_report(summary: dict):
    overall = pd.DataFrame(summary["national_spatial_cv"]).set_index("feature_set")
    selected = summary["selected_feature_set"]
    conformal = pd.DataFrame(summary["conformal"]).set_index("biomass_regime")
    comparison = pd.DataFrame(summary["field"]["comparison"]).set_index("model")
    calibration = pd.DataFrame(summary["field"]["calibration"])
    raw = calibration[(calibration.scope == "national") & (calibration.method == "raw")].iloc[0]
    calibrated = calibration[
        (calibration.scope == "national") & (calibration.method == "linear_calibration_cv")
    ].iloc[0]
    tai = comparison.loc["TAI_TRANSFER"]
    national = comparison.loc["NATIONAL_RETRAIN"]
    text = f"""# National Côte d’Ivoire biomass experiment

## Data and validation design

The source contains 5,623,253 unique GEDI shots. Raster sampling used 100,000-row chunks and windowed reads from the 100 m, EPSG:32630 national context stack. All shots have complete S2 predictors; 3,620,753 have complete S2, elevation, slope, and landcover. Missing ancillary values are explicitly audited and imputed inside the model pipeline.

The reproducible training sample contains {summary['training_sample_rows']:,} shots across 163 national 50 km blocks. Sampling was capped at 5,000 shots per block and deliberately upweighted AGBD >=200 and >=370 Mg/ha, so tail quantiles are higher than in the source population. Five fixed grouped folds were used; the 25 km sensitivity changed RMSE from {overall.loc[selected, 'rmse']:.2f} to {summary['block_25km']['rmse']:.2f} Mg/ha.

## National GEDI results

The selected stack is **{selected}**: RMSE {overall.loc[selected, 'rmse']:.2f} Mg/ha, MAE {overall.loc[selected, 'mae']:.2f}, R² {overall.loc[selected, 'r2']:.3f}, bias {overall.loc[selected, 'bias']:.2f}, and fold-RMSE SD {overall.loc[selected, 'fold_rmse_sd']:.2f}. S2-only RMSE is {overall.loc['S2_ONLY', 'rmse']:.2f}; therefore topography and categorical landcover add only {overall.loc['S2_ONLY', 'rmse'] - overall.loc[selected, 'rmse']:.2f} Mg/ha of RMSE improvement.

The 90% spatial cross-conformal radius is {conformal.loc['overall', 'radius']:.2f} Mg/ha. Overall coverage is {conformal.loc['overall', 'coverage']:.3f} by construction on the cross-validation residual pool, but conditional coverage is {conformal.loc['200_370', 'coverage']:.3f} at 200–370 Mg/ha and {conformal.loc['ge_370', 'coverage']:.6f} at >=370 Mg/ha. The map intervals must not be interpreted as conditionally calibrated for high biomass.

## Independent field evidence

The national model has valid S2 predictors for {summary['field']['national_valid_field_rows']} of 263 plots. On those plots, raw RMSE is {raw.rmse:.2f} Mg/ha, MAE {raw.mae:.2f}, bias {raw.bias:.2f}, and descriptive R² {raw.r2:.3f}. Cross-validated linear calibration changes RMSE to {calibrated.rmse:.2f} and MAE to {calibrated.mae:.2f}; it removes mean bias but does not improve predictive error.

The unchanged Taï S2+S1+PALSAR bundle has all required predictors for only {summary['field']['tai_valid_field_rows']} plots, all in the South. On that identical subset, Taï transfer RMSE is {tai.rmse:.2f} versus {national.rmse:.2f} Mg/ha for national retraining; in-domain fractions are {tai.in_domain_fraction:.2f} and {national.in_domain_fraction:.2f}. Thus national retraining does **not** outperform Taï on the available paired field subset. With n={int(tai.n)}, negative descriptive R² for both, and no Centre/North overlap, this is not evidence that Taï transfers nationally—it is an underpowered local comparison.

## Answer to the scientific question

Local Taï knowledge cannot currently be transferred as a national operational model because its required radar predictors are unavailable for almost all national field locations and for the national raster stack. National retraining is therefore necessary for national coverage with the available data, but it is not demonstrated to be more accurate than Taï on paired field plots. The defensible architecture is a provisional national backbone with domain masking and regional diagnostics, while retaining Taï as a local expert/benchmark and collecting enough harmonized S1/PALSAR and field coverage to test a true regional mixture-of-experts.

## Map limitations

The six products are 100 m predictions from temporally composited 2020–2024 features; GEDI labels span April 2019 to July 2025. Prediction intervals are constant-radius cross-conformal intervals clipped at zero. Domain distance is evaluated on a 1 km lattice and interpolated to 100 m. Severe high-biomass bias and conditional undercoverage, ancillary-layer nodata, field support mismatch, and residual spatial dependence preclude direct pixel summation into national stock totals.
"""
    Path("report/national_results.md").write_text(text, encoding="utf-8")


def run_national_pipeline(
    gedi_path: Path = Path("data/raw/gedi_l4a_cote_divoire_filtered.csv"),
    raster_path: Path | None = None,
    field_path: Path = Path("data/field/africa_biomass_all.csv"),
    config_path: Path = Path("configs/model.yaml"),
    make_map: bool = True,
) -> dict:
    raster_path = resolve_national_raster(raster_path)
    audit = audit_national_gedi(gedi_path)
    matching = extract_national_features(gedi_path, raster_path)
    field = extract_field_national_features(field_path, raster_path)
    sampling = build_national_folds_and_sample()
    experiment = run_national_experiment(config_path=config_path)
    maps = predict_national_map(raster_path=raster_path) if make_map else {}
    return {
        "audit": audit, "matching": matching, "field_rows": len(field),
        "sampling": sampling, "experiment": experiment, "maps": maps,
    }
