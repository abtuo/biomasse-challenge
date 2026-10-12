#!/usr/bin/env bash
set -euo pipefail

biomass gedi-coverage   --gedi data/raw/gedi_l4a_tai_filtered.csv   --config configs/study_area.yaml

biomass extract-raster   --gedi data/raw/gedi_l4a_tai_filtered.csv   --raster data/raw/sentinel2_tai_features_2020_2024.tif   --output data/interim/gedi_s2.parquet

biomass eda   --dataset data/interim/gedi_s2.parquet   --output-dir outputs/figures

biomass train   --dataset data/interim/gedi_s2.parquet   --config configs/model.yaml

biomass predict-map   --model outputs/models/model_bundle.joblib   --raster data/raw/sentinel2_tai_features_2020_2024.tif   --output-dir outputs/maps
