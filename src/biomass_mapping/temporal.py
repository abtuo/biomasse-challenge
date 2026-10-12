from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import SENSOR_GROUPS
from .raster import sample_raster_blockwise

GEDI_EPOCH = pd.Timestamp("2018-01-01T00:00:00Z")
SENSOR_PREFIXES = {"S2": "s2", "S1": "s1", "PALSAR": "palsar"}
ALL_FEATURES = [feature for features in SENSOR_GROUPS.values() for feature in features]


def delta_time_to_datetime(delta_time: pd.Series) -> pd.Series:
    """Convert GEDI seconds since 2018-01-01 UTC, coercing invalid values."""
    numeric = pd.to_numeric(delta_time, errors="coerce")
    valid = numeric.where(np.isfinite(numeric) & (numeric >= 0))
    return GEDI_EPOCH + pd.to_timedelta(valid, unit="s", errors="coerce")


def add_gedi_acquisition_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "delta_time" not in df:
        raise ValueError("GEDI data must contain delta_time")
    out = df.copy()
    out["gedi_datetime"] = delta_time_to_datetime(out["delta_time"])
    out["gedi_year"] = out["gedi_datetime"].dt.year.astype("Int64")
    out["gedi_month"] = out["gedi_datetime"].dt.month.astype("Int64")
    out["gedi_doy"] = out["gedi_datetime"].dt.dayofyear.astype("Int64")
    return out


def temporal_window_bounds(datetimes: pd.Series, days: int) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")
    dt = pd.to_datetime(datetimes, utc=True, errors="coerce")
    delta = pd.to_timedelta(days, unit="D")
    return pd.DataFrame({"window_start": dt - delta, "window_end": dt + delta}, index=datetimes.index)


def yearly_raster_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for sensor, prefix in SENSOR_PREFIXES.items():
        for path in sorted(root.glob(f"{prefix}_*.tif")):
            token = path.stem.rsplit("_", 1)[-1]
            if token.isdigit() and len(token) == 4:
                rows.append({"sensor": sensor, "year": int(token), "path": str(path), "exists": True})
    return pd.DataFrame(rows, columns=["sensor", "year", "path", "exists"])


def yearly_raster_lookup(root: Path, sensor: str, year: int) -> Path | None:
    if sensor not in SENSOR_PREFIXES:
        raise ValueError(f"Unknown sensor: {sensor}")
    path = root / f"{SENSOR_PREFIXES[sensor]}_{int(year)}.tif"
    return path if path.exists() else None


def sample_yearly_sensor(
    df: pd.DataFrame,
    raster_root: Path,
    sensor: str,
    year_col: str = "gedi_year",
    missing_year: str = "nan",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the raster matching each row's year without changing row identity/order."""
    if missing_year not in {"nan", "raise"}:
        raise ValueError("missing_year must be 'nan' or 'raise'")
    if year_col not in df:
        raise ValueError(f"Missing year column: {year_col}")
    features = SENSOR_GROUPS[sensor]
    sampled = pd.DataFrame(np.nan, index=df.index, columns=features, dtype="float64")
    availability = []
    years = sorted(int(year) for year in df[year_col].dropna().unique())
    for year in years:
        mask = df[year_col].eq(year)
        path = yearly_raster_lookup(raster_root, sensor, year)
        availability.append({"sensor": sensor, "year": year, "n_rows": int(mask.sum()), "available": path is not None, "path": str(path) if path else ""})
        if path is None:
            if missing_year == "raise":
                raise FileNotFoundError(f"Missing {sensor} yearly raster for {year}: {raster_root}")
            continue
        values = sample_raster_blockwise(df.loc[mask], path, features)
        sampled.loc[mask, features] = values.to_numpy()
    return sampled, pd.DataFrame(availability)


def build_same_year_dataset(
    baseline: pd.DataFrame,
    raster_root: Path,
    output_path: Path,
    missing_year: str = "nan",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = add_gedi_acquisition_fields(baseline)
    metadata = dated.drop(columns=[c for c in ALL_FEATURES if c in dated], errors="ignore")
    pieces, availability = [], []
    for sensor in SENSOR_GROUPS:
        values, status = sample_yearly_sensor(dated, raster_root, sensor, missing_year=missing_year)
        pieces.append(values)
        availability.append(status)
    out = pd.concat([metadata] + pieces, axis=1)
    if len(out) != len(baseline) or not np.array_equal(out["shot_number"], baseline["shot_number"]):
        raise RuntimeError("Temporal extraction changed GEDI row identity or order")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out, pd.concat(availability, ignore_index=True)


def build_temporal_common_sample(
    strategy_paths: dict[str, Path], output_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a complete-case, shot-matched wide table for fair strategy comparison."""
    if len(strategy_paths) < 2:
        raise ValueError("At least two temporal strategies are required")
    frames = {name: pd.read_parquet(path) for name, path in strategy_paths.items()}
    for name, frame in frames.items():
        if frame["shot_number"].duplicated().any():
            raise ValueError(f"Duplicate shot_number in {name}")
    first_name = next(iter(frames))
    first = frames[first_name]
    common_ids = set(first.loc[first[ALL_FEATURES].notna().all(axis=1), "shot_number"])
    retention = []
    for name, frame in frames.items():
        complete = frame[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        ids = set(frame.loc[complete, "shot_number"])
        common_ids &= ids
        retention.append({"strategy": name, "total_rows": len(frame), "available_rows": int(complete.sum()), "unavailable_rows": int((~complete).sum())})
    ordered_ids = first.loc[first.shot_number.isin(common_ids), "shot_number"]
    metadata_columns = [c for c in first.columns if c not in ALL_FEATURES and c != "shot_number"]
    wide = first.set_index("shot_number").loc[ordered_ids, metadata_columns].reset_index()
    for name, frame in frames.items():
        values = frame.set_index("shot_number").loc[ordered_ids, ALL_FEATURES].copy()
        values.columns = [f"{c}__{name}" for c in ALL_FEATURES]
        wide = pd.concat([wide.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    for row in retention:
        row["common_rows"] = len(wide)
        row["rows_lost_from_strategy"] = row["available_rows"] - len(wide)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output_path, index=False)
    return wide, pd.DataFrame(retention)


def folds_for_shots(shots: pd.Series, fold_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    folds = pd.read_parquet(fold_path)
    if shots.duplicated().any() or folds.shot_number.duplicated().any():
        raise ValueError("shot_number must be unique")
    found = shots.isin(folds.shot_number)
    assigned = np.full(len(shots), -1, dtype=int)
    if found.any():
        assigned[found.to_numpy()] = folds.set_index("shot_number").loc[shots[found], "fold"].to_numpy(dtype=int)
    status = pd.DataFrame({
        "n_rows": [len(shots)], "all_shots_found": [bool(found.all())],
        "n_missing": [int((~found).sum())], "n_folds": [len(np.unique(assigned[assigned >= 0]))],
        "assignment_source": [str(fold_path)], "consistent": [bool(found.all() and set(np.unique(assigned)) == set(range(5)))],
    })
    return assigned, status
