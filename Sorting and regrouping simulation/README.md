# Sorting and regrouping simulation

This module runs gated fusion + FiLM neural inference, five cell-sorting strategies and eight-cell series-pack simulation. Its LFP dataset contains 3,511 cells, with 609 test cells per seed. The supplied predictions cover 27 conditions: tail indices 46/50/54, 400/1,000/1,600 labelled training cells and seeds 0/1/2. The resulting tables support Figure 7 a-l. This module has its own LFP cohort and split; compare input hashes before combining its results with another workflow.

Use the Python environment in the repository's [installation instructions](../README.md#installation). Run these commands from the repository root. From another working directory, use an absolute script path; relative output paths are resolved from the caller's working directory.

```powershell
python "Sorting and regrouping simulation/run.py" --mode demo --output runs/regrouping_demo
python "Sorting and regrouping simulation/run.py" --mode model --output runs/regrouping_inference
python "Sorting and regrouping simulation/run.py" --mode full --output runs/regrouping_full
python "Sorting and regrouping simulation/run.py" --mode verify
python "Sorting and regrouping simulation/run.py" --mode figures
python "Sorting and regrouping simulation/model/train.py" --epochs 1 --output runs/regrouping_training
```

## Run modes

- `demo` executes the supplied `t46_i2_s0` checkpoint on all 609 test cells, compares predictions with saved diagnostic outputs, and evaluates 20 paired batches of 300 occurrences using the newly inferred predictions.
- `model` runs checkpoint inference and writes predictions and a comparison report.
- `full` uses all 27 prepared prediction conditions and the supplied paired 1,000-by-300 sampling indices. It rebuilds 135,000 strategy rows, 270,000 pairwise rows, nine fingerprinted result tables and Figure 7 a-l.
- `verify` checks the supplied results. Pass a completed full-run output root to check that run.
- `figures` redraws Figure 7 from the supplied result tables. It does not repeat training or simulation.
- `model/train.py` performs one training round for `t46_i2_s0`. One epoch is a short execution example; the configured training budget is 250 epochs. A fresh training run does not establish reproduction of the supplied checkpoint.

## Inputs and inference

The checkpoint and saved split/training state are in `results/model/`; model, data and training code are in `model/`. The processed MAT is `inputs/LFP_data_processed.mat`. Normalizers use the 400 training global IDs in the saved round workbook; test labels do not fit them. Missing upstream segments use the supplied `V_init.npy`. Mean aggregation uses native PyTorch `index_add_`.

`inputs/manifest.json` supplies the 27 prepared CSV paths and hashes, condition metadata, draw hashes and inventory hash. Each condition `t{tail}_i{iteration}_s{seed}` contains 609 saved predictions mapped by ordered global test ID. Runtime input resolution uses the bundled local files. The CSVs retain `cell_id`, `y_true`, `y_pred`, raw interval endpoints, sorted endpoints and `interval_reordered`.

Three crossed intervals among 16,443 records use sorted endpoints without measured-label information: cell 3463 in `t46_i4_s2`, and cells 838 and 989 in `t50_i2_s2`. Their point predictions lie inside the sorted endpoints. The supplied records contain no duplicate IDs within a condition, missing values or nonfinite values; measured labels agree with the bundled MAT within 5e-7 Ah.

The recorded CPU inference comparison differs from the saved predictions by at most 0.000422 Ah for point estimates and 0.000616 Ah for interval endpoints, below the empirical 0.001 Ah tolerance. The cause of these residuals is not established. Point >=14, point >14.3 and lower >=14 eligibility have zero disagreements for the representative test set. Quantile endpoints use a normal-approximation expansion; they are not newly calibrated prediction intervals. Full simulation uses the supplied saved predictions for numerical comparison.

## Sorting and statistical scope

| Strategy | Selection |
|---|---|
| S1 | Prediction >=14 Ah |
| S2 | Prediction >14.3 Ah |
| S3 | Lower endpoint >=14 Ah and prediction >=14 Ah |
| S4 | Reject low predicted capacity and the top ceil(h/5) widths within the high-capacity/high-uncertainty quadrant |
| S5 | Reveal measured labels only for S4's additional rejection candidates; retain selected records with measured capacity >=14 Ah |

High uncertainty means width greater than the full 300-occurrence batch median. Ties use source ID and occurrence position. S1-S4 sort by predicted capacity; S5 uses revealed capacity only for selected occurrences. Consecutive groups of eight form packs; leftovers are not rejected. Pack capacity is the minimum measured single-cell capacity proxy and must be at least 14 Ah to qualify; pack SD uses ddof=1.

Effective yield is qualified packs divided by floor(300/8)=37. Pass rate is qualified packs divided by assembled packs; summary tables report the mean of batch ratios and separately store the pooled pack ratio. Rejection precision is the measured-bad share among rejected occurrences, bad-cell recall is the rejected share of all measured-bad occurrences, and false rejection is the rejected share of all measured-good occurrences. Measured-bad means capacity below 14 Ah.

RPT simulates exact selected-label reveal with no added measurement noise. Duplicate bootstrap occurrences remain separate, and a reveal never propagates to an unselected copy. These outputs represent simulated RPT and pack assembly. Fixed-model Monte Carlo central 95% batch ranges describe random batch variation; conditional mean intervals reflect finite simulation error. Three-seed mixtures have no population mean confidence intervals. The ranges do not cover model retraining, population shifts or measurement noise.

## Results and verification

| Directory | Contents |
|---|---|
| `inputs/` | Processed input data, prepared predictions and paired draws in `draws/` |
| `results/reference/` | Supplied simulation tables and run metadata |
| `results/reference/figure_source/` | Nine Figure 7 comparison tables |
| `results/model/` | Checkpoint and saved split/training state |
| `results/figure_data/` | Generated plotting tables |
| `results/runs/<mode>/` | New simulation, inference or training outputs |
| `visualization/` | Plotting code, caption and plotting instructions |
| `visualization/outputs/` | Twelve panel PNGs, overview and visual checks |
| `visualization/outputs/runs/<mode>/` | Figures from new demo/full runs |
| `provenance/` | Numerical fingerprint used by the verifier |

New full runs generate `RESULTS_REPORT.md`, `RPT_workload.csv` and `RPT_comparison_overview.csv`. Filter merged tables by `n_train_total` to select a training budget. The `.csv.gz` files are compressed numerical tables readable directly by pandas.

Default figure rebuilds refresh `visualization/outputs/`; new numerical runs write beneath `results/runs/<mode>/`. Explicit `--output <root>` creates the same `results/` and `visualization/` structure below that root. The training entry point accepts module `results/runs/`, package `runs/` or an external directory and protects supplied inputs, source and checkpoints. The training configuration names its joint-training budget `num_finetune_epochs`.

Verification is read-only by default. It checks current Python source against `artifact_manifest.json`, input hashes, deterministic draws, numerical fingerprints, metric identities and Figure 7 sources/exports. A new full run writes its own source-bound manifest, English report and verification report. Figure-only rebuilds operate on supplied numerical tables.

See [Figure 7 instructions](visualization/README.md), the [common figure layout](../docs/FIGURE_OUTPUTS.md) and [repository validation commands](../README.md#validation).
