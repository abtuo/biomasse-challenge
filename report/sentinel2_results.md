# Sentinel-2 biomass experiment results

## Dataset and audit

The matched dataset contains **490,158 GEDI L4A observations** and 14 Sentinel-2 predictors. GEDI AGBD is an estimated reference, not field ground truth. There were 0 duplicated `shot_number` values. The maximum AGBD was 3849.0 Mg/ha; high observations were retained.

## Exploratory analysis and saturation

The empirical curve-flattening analysis used 25 Mg/ha bins. The cross-feature descriptive saturation region was 275.0 Mg/ha. This dataset-specific diagnostic does not establish a universal physical threshold and optical predictors should not be assumed to resolve high-biomass forest fully.

## Random versus spatial validation

The selected model was **hist_gradient_boosting**, chosen by spatial CV RMSE. Spatial CV achieved RMSE 98.94 Mg/ha, MAE 65.69, R² 0.252, and bias -0.22. The diagnostic random split RMSE was 97.98; spatial minus random RMSE was 0.96 Mg/ha, quantifying the random split's optimism. Held-out folds were never subsampled; model fitting used at most 120,000 training rows per fold.

## Uncertainty

Group-separated train, calibration, and test blocks were used for 90% split conformal prediction. Test coverage was 0.869, with mean interval width 245.92 Mg/ha. Coverage fell to 0.674 for AGBD at or above 200 Mg/ha, showing that the constant-width interval does not fully capture high-biomass error.

## Domain of validity

Standardized k-nearest-neighbour distance was fitted to training predictors. Its 95th-percentile reference threshold is 1.020; 94.8% of the spatial test observations were inside-domain. In-domain MAE was 70.05 Mg/ha and out-of-domain MAE was 63.98 Mg/ha; Spearman correlation between distance and absolute error was -0.194. Distance did not identify larger errors in this test, so it should be retained as an extrapolation flag rather than interpreted as predictive uncertainty.

## Limitations and next steps

Spatial blocks reduce, but cannot eliminate, dependence and geographic sampling bias. Sentinel-2 reflectance can saturate and GEDI labels contain retrieval uncertainty. The operational sequence should next add Sentinel-1, repeat the same spatial validation and ablation, and then evaluate the incremental contribution of ALOS PALSAR L-band data.
