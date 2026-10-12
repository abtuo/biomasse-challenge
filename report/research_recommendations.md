# Research recommendations

## Evidence from the national experiment

The spatially balanced national GEDI sample contained 753,159 observations. The selected feature set was **S2_TOPO_LC**, with 50 km spatial-CV RMSE 93.30 Mg/ha and R² 0.158. On the identical 10-plot field subset, Taï transfer RMSE was 45.74 Mg/ha and the national model RMSE was 58.87 Mg/ha. Their in-domain fractions were 0.400 and 0.600. National field-only linear calibration changed cross-validated RMSE from 56.29 to 56.35 Mg/ha; this is a diagnostic, not permission to train a complex model on 263 plots.

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
