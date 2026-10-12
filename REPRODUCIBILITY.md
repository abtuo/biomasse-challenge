# Reproducibility checklist

## Frozen decisions already made
- Study area: southwest Côte d'Ivoire around Taï.
- ROI: [-7.85, 4.95, -6.35, 6.45].
- Metric CRS: UTM 29N / EPSG:32629.
- GEDI quality filter:
  - l4_quality_flag == 1
  - degrade_flag == 0
  - agbd > 0
  - agbd_se > 0
  - agbd_se / agbd <= 0.5
- Coverage scales: 100, 250, 500, 1000, 2000, 5000 m.
- Primary validation: spatial block cross-validation.
- Main uncertainty target: empirical coverage of calibrated prediction intervals.
- Domain-of-validity: separate diagnostic from predictive uncertainty.

## Current GEDI coverage result
Using 490,159 valid unique footprints:

| Resolution | Coverage |
|---:|---:|
| 100 m | 12.4% |
| 250 m | 39.4% |
| 500 m | 71.6% |
| 1 km | 87.0% |
| 2 km | 94.8% |
| 5 km | 99.7% |

## Next empirical milestones
1. Finish Sentinel-2 export.
2. Match raster features to GEDI footprints.
3. Plot correlations and high-biomass saturation.
4. Train RF baseline under random and spatial validation.
5. Calibrate conformal intervals and verify coverage.
6. Map feature-space extrapolation.
7. Add Sentinel-1, then ALOS PALSAR, via sensor ablation.
8. Validate/calibrate against independent field plots if obtained.
