# National Côte d’Ivoire biomass experiment

## Data and validation design

The source contains 5,623,253 unique GEDI shots. Raster sampling used 100,000-row chunks and windowed reads from the 100 m, EPSG:32630 national context stack. All shots have complete S2 predictors; 3,620,753 have complete S2, elevation, slope, and landcover. Missing ancillary values are explicitly audited and imputed inside the model pipeline.

The reproducible training sample contains 753,159 shots across 163 national 50 km blocks. Sampling was capped at 5,000 shots per block and deliberately upweighted AGBD >=200 and >=370 Mg/ha, so tail quantiles are higher than in the source population. Five fixed grouped folds were used; the 25 km sensitivity changed RMSE from 93.30 to 93.21 Mg/ha.

## National GEDI results

The selected stack is **S2_TOPO_LC**: RMSE 93.30 Mg/ha, MAE 45.15, R² 0.158, bias 0.08, and fold-RMSE SD 5.33. S2-only RMSE is 93.82; therefore topography and categorical landcover add only 0.52 Mg/ha of RMSE improvement.

The 90% spatial cross-conformal radius is 81.08 Mg/ha. Overall coverage is 0.900 by construction on the cross-validation residual pool, but conditional coverage is 0.274 at 200–370 Mg/ha and 0.000138 at >=370 Mg/ha. The map intervals must not be interpreted as conditionally calibrated for high biomass.

## Independent field evidence

The national model has valid S2 predictors for 252 of 263 plots. On those plots, raw RMSE is 56.29 Mg/ha, MAE 41.44, bias -2.51, and descriptive R² 0.368. Cross-validated linear calibration changes RMSE to 56.35 and MAE to 41.59; it removes mean bias but does not improve predictive error.

The unchanged Taï S2+S1+PALSAR bundle has all required predictors for only 10 plots, all in the South. On that identical subset, Taï transfer RMSE is 45.74 versus 58.87 Mg/ha for national retraining; in-domain fractions are 0.40 and 0.60. Thus national retraining does **not** outperform Taï on the available paired field subset. With n=10, negative descriptive R² for both, and no Centre/North overlap, this is not evidence that Taï transfers nationally—it is an underpowered local comparison.

## Answer to the scientific question

Local Taï knowledge cannot currently be transferred as a national operational model because its required radar predictors are unavailable for almost all national field locations and for the national raster stack. National retraining is therefore necessary for national coverage with the available data, but it is not demonstrated to be more accurate than Taï on paired field plots. The defensible architecture is a provisional national backbone with domain masking and regional diagnostics, while retaining Taï as a local expert/benchmark and collecting enough harmonized S1/PALSAR and field coverage to test a true regional mixture-of-experts.

## Map limitations

The six products are 100 m predictions from temporally composited 2020–2024 features; GEDI labels span April 2019 to July 2025. Prediction intervals are constant-radius cross-conformal intervals clipped at zero. Domain distance is evaluated on a 1 km lattice and interpolated to 100 m. Severe high-biomass bias and conditional undercoverage, ancillary-layer nodata, field support mismatch, and residual spatial dependence preclude direct pixel summation into national stock totals.
