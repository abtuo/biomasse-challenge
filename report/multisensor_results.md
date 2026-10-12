# Multisensor biomass results

## Dataset and design

The matched analysis contains **490,158 GEDI observations** with explicit Sentinel-2, Sentinel-1, and PALSAR feature groups. All seven sensor stacks used the same persisted 10 km GroupKFold assignment. GEDI L4A is an estimated reference rather than field ground truth.

## Sensor-specific interpretation

**Sentinel-2** describes canopy reflectance, greenness, red-edge response, and SWIR-related moisture. Closed, dense canopies can show weak incremental optical response as leaf area and greenness saturate.

**Sentinel-1 C-band** has a shorter wavelength and responds to canopy-scale structure, moisture, and surface conditions. In dense forest it is expected to reach limited structural sensitivity earlier than L-band, although moisture effects can remain substantial.

**PALSAR L-band** has a longer wavelength and stronger interaction with larger woody elements and can penetrate farther into the canopy than C-band. This supplies a physical reason to test for extended high-biomass sensitivity, but does not guarantee it in these processed mosaics.

The empirical flattening diagnostics were {'S2': 161.97841649999998, 'S1': 137.00556188183594, 'PALSAR': 187.01602924999997}. PALSAR flattened later by this descriptive rule than S1 and the aggregated S2 diagnostic, but its incremental spatial-CV improvement was smaller than S1's. These are dataset-specific curve summaries, not universal saturation thresholds.

## Spatial validation and marginal sensor value

S2-only spatial CV achieved RMSE 98.91 Mg/ha and R² 0.252. Adding S1 alone to S2 produced RMSE 98.24 and R² 0.262; adding PALSAR produced RMSE 98.53 and R² 0.258. The selected stack was **S2+S1+PALSAR**, with RMSE 98.08, MAE 64.80, R² 0.265, and fold-RMSE SD 3.13. For the selected stack, random R² exceeded spatial R² by 0.010, while spatial RMSE exceeded random RMSE by 1.10 Mg/ha.

## High biomass

For AGBD >=370 Mg/ha, S2 mean residual was -274.04 Mg/ha and RMSE was 321.42. With S2+S1+PALSAR, these became -269.54 and 318.64. Persistent negative residuals indicate unresolved high-biomass underestimation even if radar improves global metrics.

## Uncertainty and domain of validity

S2 split-conformal coverage was 0.869; S2+S1+PALSAR coverage was 0.870, against a 0.90 nominal target. Above 370 Mg/ha, coverage was only 0.016 for S2 and 0.040 for S2+S1+PALSAR; top-5% coverage was 0.002 and 0.015. Constant-width split conformal therefore fails conditionally in the upper tail; this limitation is not presented as solved.

The S2+S1+PALSAR feature-space threshold was 2.366. Its distance/error Spearman correlation was -0.138. Large errors at small distance support an information-content or saturation limitation rather than pure feature-space extrapolation; saturation and domain validity are not interchangeable.

## Operational interpretation and limitations

The model predicts GEDI-referenced footprint biomass from temporally composited remote-sensing features. It is not direct field validation. Point/footprint support mismatch, geolocation uncertainty, allometric uncertainty, mosaic timing, conditional undercoverage, and residual spatial dependence all limit operational interpretation. On the 1,000-shot support subset, 50 m means reduced RMSE from 100.44 to 96.83 Mg/ha, suggesting that local landscape context and geolocation/support averaging matter. The final map uses a 25 m PALSAR-aligned grid and block-wise inference. Biomass is predicted at 25 m; kNN domain distance is evaluated on a 250 m lattice and interpolated to the same grid for operational tractability.
