### Independent field validation

#### 1. Dataset and independence from training

Ten ROI plots measured on 14–15 December 2022 were held out from every training, tuning, calibration, feature-selection, and model-selection step. Field AGBD ranged from 59.17 to 126.45 Mg/ha.

#### 2. Spatial and temporal matching protocol

Nearest GEDI shots were found in EPSG:32629. Distances ranged from 41.3 to 1458.4 m, with only one match within 250 m. GEDI times reconstructed from `delta_time` differed from field dates by about 1069 days.

#### 3. Model versus field results

For n=10, MAE was 23.14 Mg/ha, RMSE 35.76, bias 14.13, Spearman correlation -0.079, and descriptive R² -2.748. These estimates are unstable at n=10.

#### 4. GEDI versus field consistency

There were 1, 1, and 1 nearest matches within 50, 100, and 250 m. Because all three thresholds contain the same single plot, field-versus-GEDI performance cannot be estimated. That shot differed from the field value by -85.22 Mg/ha.

#### 5. Prediction interval transfer

GEDI-calibrated intervals covered 10/10 field observations, but their mean width was 224.27 Mg/ha. This is not a formal field-truth coverage guarantee.

#### 6. Domain of validity

Exact bundle-based feature distances classified 4/10 plots as in-domain. The two largest errors were out-of-domain, but out-of-domain status did not invariably imply a large error.

#### 7. Limitations

The table lacks plot area and detailed allometric metadata; spatial support equivalence is unknown. No local GEDI median was possible within 100 m, the temporal gap was about three years, and n=10 precludes broad claims.

#### 8. Operational implication

These plots provide an independent consistency check, not definitive pixel-level validation. Dedicated field campaigns with plot geometry, allometry, contemporaneous acquisitions, and broader spatial coverage are required before operational accuracy or stock claims.
