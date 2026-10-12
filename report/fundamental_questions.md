# Fundamental questions

## 1. Why can random R² be illusory?

Nearby samples share forest structure, disturbance history, soils, and image artifacts. A random split therefore puts spatial neighbours in train and test and can reward interpolation. The persisted 10 km GroupKFold design withholds whole areas, so its R² and RMSE are the primary evidence. The random-minus-spatial R² and spatial-minus-random RMSE tables quantify optimism directly.

## 2. How do GEDI allometric equations affect the target?

GEDI L4A AGBD is a model-based estimate derived from waveform metrics and calibration equations; it is not error-free field truth. Errors may vary with forest type, structure, beam, sensitivity, and biomass. Empirical verification requires independent, spatially matched field plots, stratified residual checks, calibration slope/intercept, and comparison against `agbd_se`; agreement with GEDI alone cannot validate the underlying allometry.

## 3. What causes sensor saturation?

Optical reflectance becomes less responsive after canopy closure and high leaf area. C-band interacts mainly with smaller canopy elements and moisture and often loses structural sensitivity in dense stands. Longer-wavelength L-band interacts more with woody structure and may retain useful response farther into the biomass range. These are established physical tendencies; numerical thresholds in this study are empirical and site-specific.

## 4. How can extrapolation be detected without ground truth?

Standardized kNN distance compares a prediction pixel with the training feature distribution and flags remote feature combinations. Spatial coverage, per-feature ranges, and environmental strata provide complementary checks. This diagnoses extrapolation, not accuracy: small distance with large errors can indicate target noise, support mismatch, or sensor information saturation.

## 5. Can regional stock be obtained by summing pixels?

Only with explicit area conversion, masking, bias correction, and uncertainty propagation at the intended support. Pixel errors are spatially correlated, so adding independent pixel variances is invalid and usually overstates effective sample size. Defensible totals need spatial block/bootstrap resampling, model-assisted estimators, or geostatistical covariance models, plus uncertainty from GEDI labels, model calibration, map area, and carbon conversion. Independent field validation and a clearly defined inference population are prerequisites for an operational stock estimate.

## 6. What spatial property does the model predict?

The label represents a GEDI footprint (roughly 25 m diameter), while Sentinel pixels are 20 m, PALSAR pixels are 25 m, and geolocation is imperfect. Center-pixel extraction can therefore mismatch the waveform support. Neighborhood means test 25 m and 50 m supports: improvement at broader support suggests the model represents local landscape context rather than strictly footprint biomass. Map pixels inherit the trained support relationship and should not automatically be interpreted as independent field plots.
