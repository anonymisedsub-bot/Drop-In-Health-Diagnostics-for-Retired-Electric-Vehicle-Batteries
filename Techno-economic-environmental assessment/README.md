# Targeted-RPT techno-economic and environmental assessment

This module compares the proposed diagnostic method with pulse, EIS, partial-charge and full-RPT protocols. The proposed method includes an additional full RPT for a fraction of its deployment pool. The package supplies the scientific model, parameter registry, analysis design, numerical reference summaries, density cache and Figure 8 panels.

Use the repository's Python 3.12 environment and `requirements.txt`; see [installation](../README.md#installation). TEA uses NumPy, pandas, SciPy, Matplotlib and Pillow. It runs independently of the other battery modules.

## Run from the repository root

```console
python "Techno-economic-environmental assessment/run.py" demo --output runs/tea_demo
python "Techno-economic-environmental assessment/run.py" demo --draws 10000 --output runs/tea_10000
python "Techno-economic-environmental assessment/run.py" full --output runs/tea_full
python "Techno-economic-environmental assessment/run.py" verify --output runs/tea_full
python "Techno-economic-environmental assessment/run.py" figures
```

The repository entry point also accepts `python run.py tea demo --output runs/tea_demo`. Relative paths are resolved from the caller's working directory.

- `demo` recomputes 2,000 paired triangular draws, a separate uniform-distribution repeat, all registered deterministic sensitivity grids, an independent numerical audit and a summary figure. The seed is 20260921. `--draws` changes the demo sample count; demo summaries are demonstration results.
- `full` uses 800,000 draws for each distribution, performs the independent audit and generates a density cache, 12 panels and six assembly assets. `--skip-panels` retains the numerical analysis and summary figure while omitting panel export. Allow at least 1 GB of free storage for numerical ledgers and figures.
- `verify` audits a generated run against the supplied baseline model and independent additive-RPT formulas. It regenerates the 15 shared input streams, verifies the independent q stream, checks exported method/paired/targeted records and deterministic sweeps, and recomputes primary and uniform summaries.
- `figures` (alias `reference-figures`) renders from the supplied 800,000-draw summary tables and compact density cache. It checks packaged artifacts against `reference_integrity.json`. The output records `reference_only: true` and `raw_draws_verified: false`; its verification scope is the supplied summaries and cache.

Without `--output`, reference figures go to `visualization/outputs/`, numerical runs go to `results/runs/<command>/`, and new-run figures go to `visualization/outputs/runs/<command>/`. Explicit `--output <root>` creates the same `results/` and `visualization/` structure beneath an external root, the module root or a root under package `runs/`. Entry points protect source, baseline and reference directories.

The summary figure shows medians and 2.5th-97.5th percentile simulation ranges. These are conditional simulation intervals, not experimental confidence intervals.

## Scientific scope and assumptions

The functional unit is one tonne of cells. Costs, purchased electricity and electricity-related CO2 are incremental diagnostic burdens after subtracting the common mandatory de-energisation baseline once. The model covers diagnostic burdens rather than profit, NPV or a full life-cycle assessment. Instrument auxiliary electricity, compute energy, a specialised EIS capital premium, discharge energy recovery and additional selection/handling overhead outside channel labour are outside its scope.

For equivalent cell count N, training-label fraction p and deployment retest fraction q, the extra RPT count is **q(1-p)N**. The central q is 4% of the deployment pool, with an independent triangular distribution of 3-5%. It is a deployment retest fraction. Each selected cell starts its additional full RPT at zero SOC: 1.5 h charge, 10 min rest and capacity-retention-scaled discharge. Extra purchased energy is E_cell/eta. The common discharge baseline is subtracted once. Conditioning-time approximations and comparator protocols are specified in [08_execution_protocol.md](08_execution_protocol.md).

Monte Carlo analysis uses 16 independent triangular inputs and repeats the analysis with uniform distributions over the same bounds. The 15 shared parameter streams use seed 20260921; q uses the separate SeedSequence tag 20260926. Sensitivity analysis fixes q=0.04. Costs, energy and emissions savings are computed within each paired draw before summarisation; negative savings are retained. The same draw's grid intensity multiplies electricity use, so relative CO2 and electricity reductions coincide.

This is a scenario analysis of targeted retesting. It does not execute a high-risk-cell selection model or establish improved prediction accuracy or reduced mis-sorting. The selected subgroup uses batch-representative capacity retention because subgroup-specific capacity data are outside the supplied model. In the reference primary summaries, electricity savings relative to pulse testing are negative in about 9.54% of scenarios. Electricity savings are therefore conditional on the scenario.

## Supplied files and outputs

| Path | Purpose |
|---|---|
| `tea_model.py`, `03_parameter_registry.json`, `04_analysis_design.json` | Scientific model, parameter bounds and analysis design |
| `run_analysis.py` | Paired Monte Carlo and deterministic analysis |
| `verify_analysis.py` | Independent model and additive-cycle audit |
| `baseline/` | Model, configuration and sampler used for q=0 checks |
| `results/baseline/` | Compact q=0 numerical summaries |
| `results/reference/` | 800,000-draw reference summaries and deterministic grids |
| `results/reference/figure_source/` | Density table, caption and cache metadata |
| `reference_integrity.json` | SHA-256 checks and validation scope for reference artifacts |
| `results/figure_data/` | Plotting source index and panel dimensions |
| `results/runs/<mode>/` | New results, draw ledgers, manifest, audit and density cache |
| `visualization/` | Plotting and export scripts |
| `visualization/outputs/` | Twelve panel PNGs, six assembly assets, overviews and checks |
| `visualization/outputs/runs/<mode>/` | Plots from new demo/full runs |

`results/figure_data/source_index.json` locates nine plotting inputs by path and SHA-256. Source tables and caches stay with numerical results. Paths are relative for same-drive exports and absolute for cross-drive exports; keep the module together when moving figures. New full runs store their cache and plotting index in `figure_source/` and `figure_data/` beneath the numerical output directory.

Runtime checks cover independent numerical auditing, output protection, reference-file integrity and panel export consistency. See [repository validation](../README.md#validation).

Run `python run.py tea figures`, then open `visualization/outputs/Figure_8_overview.png`. See the [visualization instructions](visualization/README.md) and [common Figure 7/8 layout](../docs/FIGURE_OUTPUTS.md).
