# Independent field validation

## Dataset and independence

Ten field plots strictly inside the ROI were used only after final model selection. They were not used for training, tuning, conformal calibration, feature selection, or model selection. Measurements date from 2022-12-14 to 2022-12-15 and range from 59.17 to 126.45 Mg/ha.

## Model versus field

For n=10, MAE was 23.14 Mg/ha, RMSE 35.76, bias 14.13, Spearman correlation -0.079, and descriptive R² -2.748. R² is not a stable generalization estimate at this sample size.

## GEDI matching and temporal support

GEDI acquisition dates were reconstructed from `delta_time` as seconds since 2018-01-01 00:00. Nearest distances ranged from 41.3 to 1458.4 m (median 1022.3 m). Match counts were {'25': 0, '50': 1, '100': 1, '250': 1, '500': 1}. Absolute field–GEDI temporal differences ranged from 1069 to 1070 days (median 1069). Threshold-specific statistics are retained even where n is zero or too small for correlation.

## Interval transfer and domain validity

The GEDI-calibrated intervals covered 10/10 field values (100.0%), with mean width 224.27 Mg/ha; this is a transfer diagnostic, not a formal 90% field-truth guarantee. Exact fitted-bundle kNN distances from native sensor point samples classified 4/10 field plots as in-domain. The interpolated map-domain value is retained separately. The two largest errors (83.77 and 69.95 Mg/ha) were out-of-domain, but other out-of-domain plots had small errors. With n=10, no error-distance relationship is fitted or claimed.

## Spatial-support sensitivity and limitations

The downloaded table has coordinates, date, and biomass but no plot area or detailed allometric-method metadata. Strict equivalence with a ~25 m GEDI footprint therefore cannot be assumed. No local GEDI median could be calculated because no plot had at least two shots within 100 m. Conclusions are an independent consistency check, not absolute pixel-level validation. The sample is too small for strong generalization claims, but is more independent of the GEDI-derived target than internal validation. Dedicated field campaigns or access to complete national-inventory metadata are needed.
