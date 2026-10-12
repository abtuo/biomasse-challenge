from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import typer
from rich import print

from .config import load_yaml
from .data import validate_s2_schema
from .field_learning import run_field_learning_experiment
from .field_validation import run_field_validation
from .figures import (
    save_coverage_plot,
    save_density_by_resolution,
)
from .gedi import coverage_by_resolution, load_gedi_csv
from .modeling import make_rf, random_holdout, spatial_cv
from .multisensor import (
    extract_multisensor_dataset,
    multisensor_audit,
    predict_multisensor_map,
    run_multisensor_experiment,
)
from .national import (
    audit_national_gedi,
    build_national_folds_and_sample,
    extract_field_national_features,
    extract_national_features,
    predict_national_map,
    run_national_experiment,
    run_national_pipeline,
)
from .raster import sample_raster_at_points
from .s2_experiment import audit_dataset, make_eda_figures, run_experiment, saturation_analysis
from .spatial import add_spatial_blocks
from .support_experiment import build_support_datasets, run_support_experiment
from .temporal_2022 import run_temporal_2022_experiment
from .temporal_workflow import prepare_temporal_alignment
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity

app = typer.Typer(add_completion=False, help="Forest biomass mapping pipeline.")

@app.command("gedi-coverage")
def gedi_coverage(
    gedi: Path = typer.Option(..., exists=True),
    config: Path = typer.Option("configs/study_area.yaml"),
):
    cfg = load_yaml(config)
    df = load_gedi_csv(str(gedi))
    roi = cfg["roi"]
    bounds = (roi["west"], roi["south"], roi["east"], roi["north"])
    table = coverage_by_resolution(
        df,
        bounds,
        cfg["crs_utm"],
        cfg["coverage_scales_m"],
    )

    out_csv = Path("data/processed/gedi_coverage_by_resolution.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_coverage_plot(table, fig_dir / "gedi_coverage_by_resolution.png")
    save_density_by_resolution(table, fig_dir / "gedi_density_by_resolution.png")

    print(table.to_string(index=False))
    print("[bold green]GEDI coverage analysis complete.[/bold green]")

@app.command("extract-raster")
def extract_raster(
    gedi: Path = typer.Option(..., exists=True),
    raster: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
):
    df = load_gedi_csv(str(gedi))
    out = sample_raster_at_points(df, raster)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"[green]Wrote {len(out):,} rows to {output}[/green]")

@app.command("eda")
def eda(
    dataset: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option("outputs/figures"),
):
    df = pd.read_parquet(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = validate_s2_schema(df)
    audit = audit_dataset(df, features, Path("outputs/tables"))
    make_eda_figures(df, features, output_dir, 10000, "EPSG:32629")
    print(f"[green]EDA complete: {len(df):,} rows, {len(features)} S2 predictors, "
          f"{audit['duplicate_shot_number']:,} duplicated shots.[/green]")

@app.command("saturation")
def saturation(
    dataset: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option("outputs"),
    bin_width: float = typer.Option(25.0),
):
    df = pd.read_parquet(dataset)
    features = validate_s2_schema(df)
    result = saturation_analysis(df, features, output_dir / "tables", output_dir / "figures", bin_width)
    print(json.dumps(result, indent=2))

@app.command("experiment-s2")
def experiment_s2(
    dataset: Path = typer.Option(..., exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    """Reproduce the complete Sentinel-2 audit, EDA and modelling experiment."""
    result = run_experiment(dataset, config)
    print(json.dumps(result, indent=2))
    print("[bold green]Sentinel-2 experiment complete.[/bold green]")

@app.command("extract-multisensor")
def extract_multisensor(
    dataset: Path = typer.Option(..., exists=True, help="Matched GEDI + Sentinel-2 parquet."),
    sentinel1: Path = typer.Option(..., exists=True),
    palsar: Path = typer.Option(..., exists=True),
    output: Path = typer.Option("data/interim/gedi_s2_s1_palsar.parquet"),
):
    out = extract_multisensor_dataset(dataset, sentinel1, palsar, output)
    audit = multisensor_audit(out, Path("outputs/tables"))
    print(f"[green]Wrote {len(out):,} coordinate-aligned rows to {output}.[/green]")
    print(pd.DataFrame(audit["audit"]).to_string(index=False))

@app.command("experiment-multisensor")
def experiment_multisensor(
    dataset: Path = typer.Option(..., exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    result = run_multisensor_experiment(dataset, config)
    print(json.dumps(result, indent=2))
    print("[bold green]Multisensor experiment complete.[/bold green]")

@app.command("predict-map-multisensor")
def predict_map_multisensor(
    model: Path = typer.Option("outputs/models/final_multisensor_model_bundle.joblib", exists=True),
    sentinel2: Path = typer.Option(..., exists=True),
    sentinel1: Path = typer.Option(..., exists=True),
    palsar: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option("outputs/maps"),
):
    predict_multisensor_map(model, sentinel2, sentinel1, palsar, output_dir)
    print(f"[green]Block-wise 25 m maps and previews written under {output_dir.parent}.[/green]")

@app.command("validate-field")
def validate_field(
    field_data: Path = typer.Option("data/field/africa_biomass_roi.csv", exists=True),
    gedi: Path = typer.Option("data/raw/gedi_l4a_tai_filtered.csv", exists=True),
    map_dir: Path = typer.Option("outputs/maps", exists=True),
):
    result = run_field_validation(field_data, gedi, map_dir)
    print("[bold]Independent field plots[/bold]")
    print(result["field"].to_string(index=False))
    print("\n[bold]Model vs field[/bold]")
    print(pd.DataFrame([result["model_metrics"]]).to_string(index=False))
    print("\n[bold]Nearest GEDI distances[/bold]")
    print(result["nearest"][["plot_id","nearest_distance_m","gedi_agbd","temporal_difference_days"]].to_string(index=False))
    print("\n[bold]GEDI threshold metrics[/bold]")
    print(result["gedi_metrics"].to_string(index=False))
    print("\n[bold]Interval coverage[/bold]")
    print(pd.DataFrame([result["coverage"]]).to_string(index=False))
    print("\n[green]Field validation complete; reports and tables written under outputs/ and report/.[/green]")

@app.command("extract-support")
def extract_support(
    dataset: Path = typer.Option("data/interim/gedi_s2_s1_palsar.parquet", exists=True),
    sentinel2: Path = typer.Option("data/raw/sentinel2_tai_features_2020_2024.tif", exists=True),
    sentinel1: Path = typer.Option("data/raw/sentinel1_tai_features_2020_2024.tif", exists=True),
    palsar: Path = typer.Option("data/raw/palsar_tai_features_2020_2024.tif", exists=True),
):
    result=build_support_datasets(dataset,{"sentinel2":sentinel2,"sentinel1":sentinel1,"palsar":palsar})
    print(json.dumps({"paths":{k:str(v) for k,v in result["paths"].items()},"common_path":str(result["common_path"]),"retention":result["retention"]},indent=2))

@app.command("experiment-support")
def experiment_support(
    dataset: Path = typer.Option("data/processed/support_common_sample.parquet", exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    result=run_support_experiment(dataset,config);print(json.dumps(result,indent=2));print("[bold green]Full footprint-support experiment complete.[/bold green]")

@app.command("national-audit")
def national_audit(
    gedi: Path = typer.Option("data/raw/gedi_l4a_cote_divoire_filtered.csv", exists=True),
):
    result = audit_national_gedi(gedi)
    print(json.dumps(result, indent=2))

@app.command("national-extract")
def national_extract(
    gedi: Path = typer.Option("data/raw/gedi_l4a_cote_divoire_filtered.csv", exists=True),
    raster: Path | None = typer.Option(None),
    field_data: Path = typer.Option("data/field/africa_biomass_all.csv", exists=True),
):
    result = extract_national_features(gedi, raster)
    field = extract_field_national_features(field_data, raster)
    print(json.dumps({"matching": result, "field_rows": len(field)}, indent=2))

@app.command("national-sample")
def national_sample(
    parts_dir: Path = typer.Option("data/interim/gedi_ci_national_s2_context_parts", exists=True),
    block_size_m: int = typer.Option(50_000),
    max_per_block: int = typer.Option(5_000),
):
    result = build_national_folds_and_sample(parts_dir, block_size_m, max_per_block)
    print(json.dumps(result, indent=2))

@app.command("national-experiment")
def national_experiment(
    sample: Path = typer.Option("data/processed/national_training_sample.parquet", exists=True),
    field_features: Path = typer.Option("data/interim/field_ci_national_features.parquet", exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    result = run_national_experiment(sample, field_features, config)
    print(json.dumps(result, indent=2, default=str))

@app.command("national-map")
def national_map(
    model: Path = typer.Option("outputs/models/national_model_bundle.joblib", exists=True),
    raster: Path | None = typer.Option(None),
    output_dir: Path = typer.Option("outputs/maps"),
):
    result = predict_national_map(model, raster, output_dir)
    print(json.dumps(result, indent=2))

@app.command("national-all")
def national_all(
    gedi: Path = typer.Option("data/raw/gedi_l4a_cote_divoire_filtered.csv", exists=True),
    raster: Path | None = typer.Option(None),
    field_data: Path = typer.Option("data/field/africa_biomass_all.csv", exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
    make_map: bool = typer.Option(True),
):
    result = run_national_pipeline(gedi, raster, field_data, config, make_map)
    print(json.dumps(result, indent=2, default=str))

@app.command("experiment-field-learning")
def experiment_field_learning():
    """Run leakage-safe small-field-data learning and acquisition-design experiments."""
    result = run_field_learning_experiment()
    print(json.dumps(result, indent=2, default=str))
    print("[bold green]Field-learning experiment complete.[/bold green]")

@app.command("temporal-prepare")
def temporal_prepare(
    dataset: Path = typer.Option("data/interim/gedi_s2_s1_palsar.parquet", exists=True),
    raster_dir: Path = typer.Option("data/raw/temporal"),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    """Prepare dates/annual features and run the temporal experiment when exports exist."""
    result = prepare_temporal_alignment(dataset, raster_dir, config, run_experiment=True)
    print(json.dumps(result, indent=2))
    if result["required_exports"]:
        print("[yellow]Annual rasters are missing; see report/temporal_alignment.md for exact exports.[/yellow]")
    else:
        print("[bold green]Temporal alignment experiment complete.[/bold green]")

@app.command("experiment-temporal-2022")
def experiment_temporal_2022(
    gedi: Path = typer.Option("data/raw/gedi_l4a_tai_filtered.csv", exists=True),
    multiyear: Path = typer.Option("data/interim/gedi_s2_s1_palsar.parquet", exists=True),
    raster_dir: Path = typer.Option("data/raw/temporal", exists=True),
    config: Path = typer.Option("configs/model.yaml", exists=True),
):
    """Run the strict paired 2022 multi-year versus same-year experiment."""
    result = run_temporal_2022_experiment(gedi, multiyear, raster_dir, config)
    print(json.dumps(result, indent=2, default=str))
    print("[bold green]Paired 2022 temporal experiment complete.[/bold green]")

@app.command("train")
def train(
    dataset: Path = typer.Option(..., exists=True),
    config: Path = typer.Option("configs/model.yaml"),
):
    cfg = load_yaml(config)
    df = pd.read_parquet(dataset)
    df = add_spatial_blocks(
        df,
        block_size_m=cfg["spatial"]["block_size_m"],
        crs=cfg["spatial"]["crs"],
    )

    excluded = set(cfg["features"]["exclude"]) | {"x_utm", "y_utm", "spatial_block"}
    features = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not features:
        raise typer.BadParameter("No numeric predictor columns found.")

    X = df[features]
    y = df[cfg["target"]]
    groups = df["spatial_block"]

    model = make_rf(cfg["model"].get("random_forest", cfg["model"]))

    folds, oof, overall = spatial_cv(
        model,
        X,
        y,
        groups,
        n_splits=cfg["spatial"]["n_splits"],
    )
    random_metrics = random_holdout(model, X, y)

    # Calibrate conformal radius on OOF residuals.
    radius = conformal_radius(
        y,
        oof,
        alpha=cfg["conformal"]["alpha"],
    )
    lo, hi = interval_from_radius(oof, radius)
    oof_coverage = empirical_coverage(y, lo, hi)

    # Fit final model + domain validity.
    model.fit(X, y)
    validity = KNNDomainValidity(
        n_neighbors=cfg["validity"]["k_neighbors"],
        threshold_quantile=cfg["validity"]["quantile_threshold"],
    ).fit(X)

    out_dir = Path("outputs/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "conformal_radius": radius,
            "validity": validity,
            "target": cfg["target"],
        },
        out_dir / "model_bundle.joblib",
    )
    folds.to_csv(out_dir / "spatial_cv_folds.csv", index=False)

    metrics = {
        "spatial_cv_overall": overall,
        "random_holdout": random_metrics,
        "conformal_radius": radius,
        "oof_interval_coverage": oof_coverage,
        "n_samples": len(df),
        "n_features": len(features),
        "features": features,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("[bold green]Training complete.[/bold green]")

@app.command("predict-map")
def predict_map(
    model: Path = typer.Option(..., exists=True),
    raster: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option("outputs/maps"),
):
    bundle = joblib.load(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster) as src:
        arr = src.read(masked=True).astype("float32")
        names = [
            d if d not in (None, "") else f"band_{i+1}"
            for i, d in enumerate(src.descriptions)
        ]
        mapping = {n: i for i, n in enumerate(names)}

        missing = [f for f in bundle["features"] if f not in mapping]
        if missing:
            raise typer.BadParameter(f"Raster is missing model bands: {missing}")

        stack = np.stack([arr[mapping[f]].filled(np.nan) for f in bundle["features"]])
        h, w = stack.shape[1:]
        X = stack.reshape(len(bundle["features"]), -1).T
        valid = np.isfinite(X).all(axis=1)

        pred = np.full(h*w, np.nan, dtype="float32")
        low = np.full(h*w, np.nan, dtype="float32")
        high = np.full(h*w, np.nan, dtype="float32")
        dist = np.full(h*w, np.nan, dtype="float32")

        p = bundle["model"].predict(X[valid])
        lo, hi = interval_from_radius(p, bundle["conformal_radius"])
        dd = bundle["validity"].score_samples(X[valid])

        pred[valid] = p
        low[valid] = lo
        high[valid] = hi
        dist[valid] = dd

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", compress="deflate", nodata=np.nan)

        outputs = {
            "biomass_prediction.tif": pred.reshape(h, w),
            "biomass_pi_lower.tif": low.reshape(h, w),
            "biomass_pi_upper.tif": high.reshape(h, w),
            "biomass_interval_width.tif": (high-low).reshape(h, w),
            "domain_distance.tif": dist.reshape(h, w),
        }

        for name, data in outputs.items():
            with rasterio.open(output_dir / name, "w", **profile) as dst:
                dst.write(data, 1)

    print(f"[green]Maps written to {output_dir}[/green]")


if __name__ == "__main__":
    app()
