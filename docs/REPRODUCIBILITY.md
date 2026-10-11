# Reproducibility conventions

- Capacity is reported in Ah; SOH is capacity divided by the declared nominal
  capacity, multiplied by 100. A percentage error and an Ah error are different units.
- Self-collected NMC nominal capacity is 2.5 Ah; LFP nominal capacity is 15.5 Ah.
- Data identity is the exact file hash plus sample index. Repeated test appearances
  across seeds or label budgets are not independent new cells.
- Labelled, unlabelled, validation and test roles remain distinct. Normalization
  and saved indices must travel with each checkpoint. Validation selects models;
  test labels evaluate predictions.
- Quantile outputs describe the trained prediction interval. Coverage must be
  evaluated empirically; a nominal quantile level is not proof of calibration.
- Fixed-model bootstrap results describe the saved cohort and sampling design.
  Monte Carlo intervals do not automatically cover retraining, a new population,
  or physical measurement error. Paired comparisons use identical sampled records.
- For the eight-cell series-pack simulation, usable capacity is the minimum measured
  cell capacity under the stated idealized rule. Resistance, thermal behavior, ageing,
  balancing and electrical safety are not modelled by this rule.
- The simulated targeted RPT uses measured labels only after selection. Duplicate
  bootstrap appearances represent repeated simulated records; they are not additional
  physical batteries and must not inherit another appearance's revealed label.

The detailed scientific definitions and run settings are described in each
module. Root `release_manifest.json` covers the delivered files; newly generated
`runs/` outputs are excluded from the frozen inventory by design.
