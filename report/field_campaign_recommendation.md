# Recommended field campaign

1. **Campaign size.** The current archive already contains 252 valid plots. Acquire a next tranche of **80 additional plots**: approximately **60** for model updating and **20** kept as an untouched validation reserve. The 60-plot calibration allocation reflects the first stable learning-curve gain, not a universal threshold.
2. **Regional allocation.** Current counts are South 82, Centre 118, North 52. As a concrete starting quota, place about 40 new plots in North, 25 in environmentally underrepresented Centre settings, and 15 in South, emphasizing under-covered conditions rather than existing clusters. Apply the 75/25 calibration/validation split within each region.
3. **OOD and uncertainty.** Use domain distance and mapped uncertainty as acquisition-time covariates. The best tested selection strategy was **SPATIALLY_BALANCED**; operational selection should keep geographic quotas so OOD targeting does not collapse into one cluster.
4. **High biomass.** Deliberately oversample the upper national-prediction stratum, but define selection from mapped prediction/uncertainty rather than unknown field AGBD.
5. **Spatial layout.** Prefer dispersed plots across 50–100 km blocks, with a smaller number of local replicate clusters for measurement and support-variance estimation.
6. **Transfer.** Mean cross-region RMSE exceeded the within-region summary by 24.79 Mg/ha. Local corrections should not be exported without held-out regional evidence.
7. **Required metadata.** Record plot area and geometry, DBH, height, species and wood density, GPS accuracy, measurement date, allometric equation/version, disturbance history, protocol, and plot/measurement uncertainty. Preserve tree-level data and permanent plot identifiers where permitted.

Re-evaluate the learning curve after each acquisition tranche; stop or redirect sampling when spatially held-out gains plateau rather than committing to a fixed total in advance.
