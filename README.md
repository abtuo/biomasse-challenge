# Biomass Challenge — Côte d’Ivoire

## Objective

This repository contains a reproducible research pipeline for mapping forest
above-ground biomass density (AGBD) in Côte d’Ivoire. It combines GEDI L4A
observations with Sentinel-2, Sentinel-1, PALSAR, terrain, and land-cover
predictors, with spatial validation, uncertainty estimation, domain-validity
diagnostics, and independent field calibration.

The repository contains code and lightweight scientific documentation only.
Large rasters, GEDI extracts, field tables, fitted models, and generated maps
are stored separately on Google Drive and are intentionally excluded from Git.

## Main experiments

- Taï local study and GEDI coverage analysis
- Sentinel-2 baseline and saturation diagnostics
- Sentinel-1/PALSAR sensor ablation
- GEDI footprint-support sensitivity
- temporal and strict 2022 same-year sensitivity
- national Côte d’Ivoire retraining and mapping
- independent field validation
- field-learning, sample-efficiency, transfer, and acquisition-design study

The reports in [`report/`](report/) summarize the completed experiments and
their limitations. Generated numerical outputs can be reproduced locally after
the external data have been restored.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Install the package and development tools:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The optional field-data preparation script also needs Hugging Face Datasets:

```bash
pip install -e ".[data,dev]"
```

## External data

Copy the separately distributed data into the following ignored directories:

```text
data/
├── raw/
│   ├── gedi_l4a_tai_filtered.csv
│   ├── gedi_l4a_cote_divoire_filtered.csv
│   ├── sentinel2_tai_features_2020_2024.tif
│   ├── sentinel1_tai_features_2020_2024.tif
│   ├── palsar_tai_features_2020_2024.tif
│   └── temporal/                  # annual sensor exports
├── field/
│   └── africa_biomass_all.csv
├── interim/                       # generated feature tables
└── processed/                     # folds and training samples
```

The code never requires a personal absolute path. Alternative inputs can be
provided through CLI options; run `biomass COMMAND --help` for details.

## Quick start

Inspect all commands:

```bash
biomass --help
```

Run the core Taï Sentinel-2 workflow:

```bash
biomass gedi-coverage \
  --gedi data/raw/gedi_l4a_tai_filtered.csv \
  --config configs/study_area.yaml

biomass extract-raster \
  --gedi data/raw/gedi_l4a_tai_filtered.csv \
  --raster data/raw/sentinel2_tai_features_2020_2024.tif \
  --output data/interim/gedi_s2.parquet

biomass experiment-s2 \
  --dataset data/interim/gedi_s2.parquet \
  --config configs/model.yaml
```

The equivalent starter workflow is available in
[`scripts/run_all.sh`](scripts/run_all.sh) and
[`scripts/run_all.ps1`](scripts/run_all.ps1).

## Experiment entry points

```bash
# Multisensor feature extraction and ablation
biomass extract-multisensor --help
biomass experiment-multisensor --help

# Footprint-support sensitivity
biomass extract-support --help
biomass experiment-support --help

# Temporal sensitivity
biomass temporal-prepare --help
biomass experiment-temporal-2022 --help

# National extraction, retraining, and mapping
biomass national-audit
biomass national-all --help

# Independent field validation and learning curves
biomass validate-field --help
biomass experiment-field-learning
```

All generated tables, figures, models, and maps are written below `outputs/`
and remain untracked. Experiment narratives are written below `report/`.

## Google Earth Engine exports

The scripts under [`gee/`](gee/) prepare the required GEDI and sensor exports:

- `01_export_gedi.js`
- `02_export_sentinel2.js`
- `03_export_sentinel1.js`
- `04_export_palsar.js`
- annual exports under `gee/temporal/`

Exported products belong in `data/raw/` or `data/raw/temporal/`, not in Git.

## Validation and reproducibility

```bash
python -m ruff check src tests scripts
python -m pytest -q
```

Random seeds, spatial block definitions, model settings, conformal settings,
and compute limits are recorded in [`configs/model.yaml`](configs/model.yaml).
Study-area bounds and GEDI quality settings are in
[`configs/study_area.yaml`](configs/study_area.yaml). See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the methodological checklist.

Core safeguards include:

- spatially blocked validation as the primary estimate of transfer performance;
- fold-local preprocessing, fitting, calibration, and feature handling;
- deterministic subsampling and persisted spatial assignments;
- uncertainty and domain-validity diagnostics alongside point accuracy;
- no use of held-out field targets for acquisition-strategy selection.

## Repository structure

```text
configs/                 model and study-area configuration
gee/                     Earth Engine export scripts
scripts/                 reusable preparation and orchestration utilities
src/biomass_mapping/     installable Python package and CLI
tests/                   unit and leakage-safety tests
report/                  lightweight experiment reports and report assets
data/                    ignored local inputs and intermediates
outputs/                 ignored generated results
```

## Data and artifact policy

Do not commit raw or derived data, GeoTIFFs, Parquet datasets, serialized model
bundles, national maps, credentials, or downloaded Earth Engine exports. Keep
these in the external project storage and use the directory layout above so the
pipeline remains reproducible without embedding machine-specific paths.
