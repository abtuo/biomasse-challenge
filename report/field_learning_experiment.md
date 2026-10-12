# Field-learning experiment

## 1. Motivation

This experiment asks whether a small, independent field sample can improve the national GEDI-trained model, how much data are needed, and whether corrections transfer spatially and ecologically.

## 2. Experimental design

There are 252 plots with valid national predictions (AGBD 0.9–391.1 Mg/ha): 82 South, 118 Centre, and 52 North. Primary evidence uses persisted 50 km spatial folds; random CV is diagnostic, 100 km blocks are a sensitivity, and each region is held out in turn. F3 was predeclared for headline curves, preventing evaluation-fold feature selection.

## 3. GEDI baseline

The unchanged national prediction has spatial-CV RMSE 56.29, MAE 41.44, bias -2.51, and descriptive R² 0.368 Mg/ha.

## 4. Field-only learning

The best primary configuration was **STACKED_FIELD_MODEL** with RMSE 53.43 and MAE 38.45. Field-only models are compared with Ridge, shallow RF, and regularized HGBR in `field_model_complexity.csv`; flexible models are not assumed superior on 252 clustered plots.

## 5. Calibration and residual correction

Linear calibration, F3 residual correction, and F3 stacking were fitted only on each training fold. Their results show whether field data are best used to replace, calibrate, or locally correct the GEDI representation.

## 6. Sample efficiency

The first sample size satisfying the declared stable-gain rule (>=2% mean RMSE gain and improvement in >=75% of spatial repeats) was **60**. Each small-n estimate aggregates 20 deterministic, spatially held-out evaluations.

## 7. Strategic sampling

The lowest mean RMSE strategy was **SPATIALLY_BALANCED** (53.11 Mg/ha). Predicted-biomass stratification uses the existing national prediction, never held-out field truth.

## 8. Regional transfer

Within-region and cross-region results are in `field_regional_transfer.csv`. Local corrections should be treated as transportable only where cross-region rows remain competitive with the unchanged baseline; source-region success alone is insufficient.

## 9. Distance to calibration

Geographic-distance/error Spearman correlations range from -0.00 to 0.08; feature-distance/error correlations range from -0.13 to -0.00. These gradients are descriptive and do not establish a hard transfer radius.

## 10. Uncertainty

GEDI-derived field coverage was 0.873 with mean width 151.4 Mg/ha. Leakage-safe field-residual coverage was 0.960 with width 213.4. Region and biomass-stratum coverage remain uncertain because group sizes are small.

## 11. Operational recommendation

Use the national GEDI model as the representation backbone and add field correction only where spatially held-out gains are stable. Field campaign design should follow the best leakage-safe selection strategy and deliberately improve North/Centre, OOD, and upper-predicted-biomass coverage.

## 12. Limits

Plots are geographically clustered, plot area and detailed forest-inventory metadata are absent, landcover has only seven mosaic/agriculture plots, and no ecological category is a sourced formal ecoregion. Target-support and temporal mismatch remain. Results describe this 252-plot sample and should not be converted into universal learning thresholds.
