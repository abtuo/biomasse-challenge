# Temporal alignment experiment

## Status

The local baseline preparation and validation are complete, but the temporal model comparison is **not yet estimable** because no annual sensor rasters are present. No temporal performance result is inferred from the multi-year baseline alone.

GEDI spans 2019-04-18 06:55:26.183710001+00:00 through 2025-06-20 02:25:38.179655999+00:00, with 0 missing or invalid acquisition dates. Counts by year are:

- 2019: 65,321
- 2020: 12,319
- 2021: 81,409
- 2022: 164,468
- 2023: 53,751
- 2024: 44,141
- 2025: 68,749

The persisted 10 km folds were reused successfully: True. Missing sensor-years are `{'PALSAR': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'S1': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'S2': [2019, 2020, 2021, 2022, 2023, 2024, 2025]}`. GEDI includes 2025 shots, so 2025 must not be silently replaced with 2024.

## Required exports

Run the three scripts in `gee/temporal/` and place the successful exports at:

- `data\raw\temporal\s2_2019.tif`
- `data\raw\temporal\s2_2020.tif`
- `data\raw\temporal\s2_2021.tif`
- `data\raw\temporal\s2_2022.tif`
- `data\raw\temporal\s2_2023.tif`
- `data\raw\temporal\s2_2024.tif`
- `data\raw\temporal\s2_2025.tif`
- `data\raw\temporal\s1_2019.tif`
- `data\raw\temporal\s1_2020.tif`
- `data\raw\temporal\s1_2021.tif`
- `data\raw\temporal\s1_2022.tif`
- `data\raw\temporal\s1_2023.tif`
- `data\raw\temporal\s1_2024.tif`
- `data\raw\temporal\s1_2025.tif`
- `data\raw\temporal\palsar_2019.tif`
- `data\raw\temporal\palsar_2020.tif`
- `data\raw\temporal\palsar_2021.tif`
- `data\raw\temporal\palsar_2022.tif`
- `data\raw\temporal\palsar_2023.tif`
- `data\raw\temporal\palsar_2024.tif`
- `data\raw\temporal\palsar_2025.tif`

Earth Engine may report that some PALSAR years contain no HH+HV acquisitions. Those years must remain explicitly unavailable; annual PALSAR is the finest matching used here and is not equivalent to an exact ±6-month window. The ±6-month and ±3-month strategies are not implemented from annual rasters because doing so would mislabel yearly composites as acquisition-window composites.

## Scientific and operational recommendation

The central question—whether temporal mismatch explains current error—remains unresolved until same-year rasters exist. Retain the 2020–2024 multi-year production composite in the meantime. Do not change the model, folds, or training sample to manufacture a temporal comparison. After exporting, run `biomass temporal-prepare`; it will build the common sample and execute the fixed-fold experiment automatically.
