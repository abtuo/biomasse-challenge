# Paired 2022 temporal-alignment results

## Design

The experiment contains 164,468 GEDI shots acquired in 2022. Both strategies use exactly the same shots, feature names, persisted 10 km folds, HistGradientBoosting model, hyperparameters, and training caps. Only feature timing differs: the existing 2020–2024 composite versus the 2022 rasters.

## Direct answers

1. **Does 2022-only remote sensing improve performance?** Multi-year RMSE/R² are 96.10/0.267; same-year values are 98.00/0.238. The paired deltas (same minus multi) are 1.90 RMSE and -0.0293 R². The multi-year composite performs better, consistent with temporal aggregation reducing seasonal, cloud, speckle, and moisture variability enough to outweigh temporal mismatch.
2. **Fold stability:** same-year RMSE is lower in 0/5 spatial folds. Fold-level deltas are `[1.83, 1.87, 1.77, 2.23, 1.79]` Mg/ha.
3. **Sensor temporal effect:** the largest absolute RMSE change occurs for **PALSAR** (3.36 Mg/ha, same minus multi). Feature-level correlations and differences are in the stability table.
4. **High biomass:** bias at AGBD ≥370 changes from -263.95 to -271.44 Mg/ha. The fraction predicted below 200 changes from 0.367 to 0.443.
5. **Conformal calibration:** overall coverage changes from 0.900 to 0.903; ≥370 coverage changes from 0.039 to 0.037.
6. **Domain validity:** in-domain fraction changes from 0.960 to 0.959. Domain distance is an extrapolation diagnostic, not predictive uncertainty.
7. **Operational decision:** Retain multi-year aggregation for production and treat annual matching as a change-monitoring product rather than the default biomass model.

## Paired error evidence

Mean absolute-error difference (same minus multi) is 1.891 Mg/ha, with a 10 km spatial-block bootstrap 95% interval [1.601, 2.195]. Same-year improves 46.7% of observations and worsens 53.3%. The Wilcoxon result is reported in the table, but effect sizes and spatial consistency—not the large-sample p-value—drive interpretation.

## Conclusion

The multi-year composite performs better, consistent with temporal aggregation reducing seasonal, cloud, speckle, and moisture variability enough to outweigh temporal mismatch.

Retain multi-year aggregation for production and treat annual matching as a change-monitoring product rather than the default biomass model.
