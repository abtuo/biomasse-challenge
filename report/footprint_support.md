# GEDI footprint-support sensitivity

## Implemented geometry

POINT uses the existing center-pixel extraction. SUPPORT_25 uses pixel centres inside a 12.5 m radius; SUPPORT_50 uses pixel centres inside a 25 m radius. If a 12.5 m circle contains no centre on the 20–25 m grids, the nearest pixel is used. These are practical discrete-grid approximations, not exact integrals over the circular GEDI waveform footprint.

## Data consistency

Of 490,158 GEDI rows, POINT retained 490,158, SUPPORT_25 retained 490,157, and SUPPORT_50 retained 490,086. The matched comparison contains 490,086 rows (99.985%). The 1 and 72 incomplete rows under the averaged supports arise where expanded circular windows encounter raster boundaries or nodata; they are reported rather than silently removed. Every matched shot was mapped by `shot_number` to its original saved spatial fold.

## Results and direct answers

1. **Does local averaging improve spatial CV?** POINT, SUPPORT_25, and SUPPORT_50 RMSE were 98.12, 97.96, and 97.32 Mg/ha. Relative to POINT, the gains were only 0.16 and 0.80 Mg/ha, respectively.
2. **Does it reduce high-biomass underestimation?** Mean residuals above 370 Mg/ha were -269.49, -268.76, and -265.37 Mg/ha. The absolute-bias improvements were 0.73 and 4.12 Mg/ha; none removes the upper-tail failure.
3. **Does it improve conformal coverage?** Overall coverage was 0.869, 0.869, and 0.870; this is effectively unchanged and below the nominal 0.90. Coverage at AGBD >=370 Mg/ha was only 0.034, 0.040, and 0.051, so broader support does not repair upper-tail interval failure.
4. **Does domain behavior change?** In-domain fractions were 0.945, 0.945, and 0.943. Small differences should not be overinterpreted.
5. **Is POINT materially worse than 25 m?** No. The 0.16 Mg/ha RMSE and 0.73 Mg/ha high-biomass bias gains are negligible under the declared thresholds. The comparison uses the same rows, folds, features, and hyperparameters.
6. **Does 50 m help or over-smooth?** It gives the best numerical scores and reduces feature variance, while retaining mean feature correlation 0.933 with POINT (versus 0.992 at 25 m). This is weak evidence of useful contextual smoothing, not a material predictive change.
7. **Effective learned support and decision.** The 50 m gains (0.80 Mg/ha RMSE; 4.12 Mg/ha absolute >=370 bias) fall short of the predeclared 1 and 5 Mg/ha thresholds. Coverage does not deteriorate, but passing that safeguard alone is insufficient. Therefore **POINT is retained**: the experiment does not establish that broader landscape support is operationally superior to center-pixel extraction.

The rule requires at least 1 Mg/ha RMSE improvement, 5 Mg/ha improvement in absolute >=370 bias, and no more than 0.01 coverage loss before preferring a larger support.

## Support-mismatch limitations

GEDI's footprint is not identical to a raster pixel. Sentinel-2/Sentinel-1 are 20 m, PALSAR is 25 m, and their pixel centres and grids differ. GEDI geolocation is uncertain, a circular footprint is discretized onto square pixels, and all sensors are temporally composited. The open field subset also lacks plot area. Consequently, the model estimates a GEDI-referenced remote-sensing response whose effective support may include surrounding landscape context; it should not automatically be described as exact plot biomass.
