from __future__ import annotations

from pathlib import Path

import pandas as pd

from .temporal import (
    ALL_FEATURES,
    SENSOR_PREFIXES,
    add_gedi_acquisition_fields,
    build_same_year_dataset,
    build_temporal_common_sample,
    folds_for_shots,
)
from .temporal_experiment import run_temporal_experiment, write_gedi_date_outputs


def prepare_temporal_alignment(
    source_path: Path = Path("data/interim/gedi_s2_s1_palsar.parquet"),
    raster_root: Path = Path("data/raw/temporal"),
    config_path: Path = Path("configs/model.yaml"),
    run_experiment: bool = True,
) -> dict:
    tables = Path("outputs/tables")
    figures = Path("outputs/figures/temporal")
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(source_path)
    dates = write_gedi_date_outputs(source, tables / "gedi_acquisition_date_summary.csv", figures / "gedi_acquisition_year_distribution.png")
    baseline = add_gedi_acquisition_fields(source)
    baseline_path = Path("data/interim/gedi_multisensor_multiyear.parquet")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(baseline_path, index=False)
    years = sorted(int(year) for year in baseline.gedi_year.dropna().unique())
    year_counts = baseline.gedi_year.value_counts().to_dict()
    availability_rows, required = [], []
    for sensor, prefix in SENSOR_PREFIXES.items():
        for year in years:
            path = raster_root / f"{prefix}_{year}.tif"
            available = path.exists()
            availability_rows.append({"sensor": sensor, "year": year, "n_gedi_rows": int(year_counts[year]), "available": available, "path": str(path)})
            if not available:
                required.append(str(path))
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(tables / "temporal_raster_availability.csv", index=False)
    _, fold_status = folds_for_shots(baseline.shot_number, Path("data/processed/spatial_fold_assignment.parquet"))
    fold_status.insert(0, "strategy", "MULTIYEAR_BASELINE")
    fold_status.to_csv(tables / "temporal_fold_consistency.csv", index=False)
    complete_baseline = baseline[ALL_FEATURES].notna().all(axis=1)
    retention = []
    for year in years:
        mask = baseline.gedi_year.eq(year)
        retention.append({"strategy": "MULTIYEAR_BASELINE", "gedi_year": year, "total_rows": int(mask.sum()),
                          "available_rows": int((mask & complete_baseline).sum()), "common_rows": "", "loss_reason": "feature nodata" if not complete_baseline[mask].all() else ""})
        missing = availability[(availability.year == year) & ~availability.available].sensor.tolist()
        retention.append({"strategy": "SAME_YEAR", "gedi_year": year, "total_rows": int(mask.sum()),
                          "available_rows": 0 if missing else "pending_extraction", "common_rows": "",
                          "loss_reason": "missing yearly raster: " + ", ".join(missing) if missing else "pending extraction"})
    retention_table = pd.DataFrame(retention)
    same_year_path = Path("data/interim/gedi_multisensor_same_year.parquet")
    common_path = Path("data/processed/temporal_common_sample.parquet")
    summary = {"dates": dates, "available_strategies": ["MULTIYEAR_BASELINE"], "required_exports": required,
               "folds_reused": bool(fold_status.consistent.iloc[0]), "experiment_run": False}
    # Once at least one annual raster exists for every sensor, extract all rows.
    # Missing sensor-years remain NaN and are reported; they are never substituted.
    can_extract = bool(availability.groupby("sensor").available.any().all())
    if can_extract:
        _, extraction_availability = build_same_year_dataset(baseline, raster_root, same_year_path)
        extraction_availability.to_csv(tables / "temporal_raster_availability.csv", index=False)
        common, overall_retention = build_temporal_common_sample(
            {"MULTIYEAR_BASELINE": baseline_path, "SAME_YEAR": same_year_path}, common_path)
        # Preserve year-specific retention and append the overall comparison rows.
        overall_retention["gedi_year"] = "ALL"
        overall_retention["loss_reason"] = "nodata or unavailable pixel"
        retention_table = pd.concat([retention_table, overall_retention], ignore_index=True, sort=False)
        summary["available_strategies"].append("SAME_YEAR")
        summary["common_rows"] = len(common)
        if run_experiment and len(common) > 0:
            summary["experiment"] = run_temporal_experiment(common_path, config_path, summary["available_strategies"])
            summary["experiment_run"] = True
    retention_table.to_csv(tables / "temporal_sample_retention.csv", index=False)
    if required and not summary["experiment_run"]:
        _write_pending_report(dates, availability, required, fold_status)
    return summary


def _write_pending_report(dates, availability, required, fold_status):
    missing_by_sensor = availability.loc[~availability.available].groupby("sensor").year.apply(list).to_dict()
    exports = "\n".join(f"- `{path}`" for path in required)
    counts = "\n".join(f"- {year}: {count:,}" for year, count in dates["counts_by_year"].items())
    text = f"""# Temporal alignment experiment

## Status

The local baseline preparation and validation are complete, but the temporal model comparison is **not yet estimable** because no annual sensor rasters are present. No temporal performance result is inferred from the multi-year baseline alone.

GEDI spans {dates['minimum']} through {dates['maximum']}, with {dates['invalid']} missing or invalid acquisition dates. Counts by year are:

{counts}

The persisted 10 km folds were reused successfully: {bool(fold_status.consistent.iloc[0])}. Missing sensor-years are `{missing_by_sensor}`. GEDI includes 2025 shots, so 2025 must not be silently replaced with 2024.

## Required exports

Run the three scripts in `gee/temporal/` and place the successful exports at:

{exports}

Earth Engine may report that some PALSAR years contain no HH+HV acquisitions. Those years must remain explicitly unavailable; annual PALSAR is the finest matching used here and is not equivalent to an exact ±6-month window. The ±6-month and ±3-month strategies are not implemented from annual rasters because doing so would mislabel yearly composites as acquisition-window composites.

## Scientific and operational recommendation

The central question—whether temporal mismatch explains current error—remains unresolved until same-year rasters exist. Retain the 2020–2024 multi-year production composite in the meantime. Do not change the model, folds, or training sample to manufacture a temporal comparison. After exporting, run `biomass temporal-prepare`; it will build the common sample and execute the fixed-fold experiment automatically.
"""
    Path("report/temporal_alignment.md").write_text(text, encoding="utf-8")
