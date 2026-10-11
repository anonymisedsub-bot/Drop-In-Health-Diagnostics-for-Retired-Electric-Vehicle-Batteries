# Baseline comparisons

Each dataset's `model_construction/Baseline group N/` contains a `main.py`, `config.py`, models and training code. Run that main.py directly. The adjacent configuration controls the experiment.

## Fixed methods

| Directory | Methods | Input |
| --- | --- | --- |
| Baseline group 1 | XGB, RF, CGB, LGBM | Discharge tail |
| Baseline group 2 | MLP, CNN, LSTM, Transformer | Discharge tail |
| Baseline group 3 | DA-AL, DA-MMD, DA-CORAL, SSL-PL | Discharge tail |
| Baseline group 4 | MLP, CNN, CGB, LGBM | Tail plus padded upstream input |

Group 3 uses MLP predictors. DA-AL, DA-MMD and DA-CORAL use adversarial, MMD and CORAL alignment. SSL-PL generates weighted KNN pseudo-labels for unlabelled training examples from labelled training references, directly in normalized tail space.

## Packaged data contracts

| Dataset | MAT | Cells | Tail window | Groups 1–3 | Group 4 |
| --- | --- | ---: | --- | ---: | ---: |
| LFP | LFP_data_processed.mat | 3511 | 2.50–2.73 V | 47 | 107 |
| NMC_A | NMC_A_data_processed.mat | 353 | 3.00–3.30 V | 61 | 121 |
| NMC_B | NMC_B_data_processed.mat | 334 | 2.75–3.20 V | 91 | 151 |

MAT files are in each dataset's `dataset/`. NMC configurations display both presets and select one with `DATASET_NAME`. The voltage-removal window is zero. Paths, windows and derived dimensions are resolved in config.py, and validated by the data loader. Incompatible custom data/settings produce validation errors rather than silent corrections.

Group 4 adds 60 upstream positions (0.30 V). LFP uses its masking protocol for initially labelled, queried and held-out cells. NMC preserves measured available upstream observations in every role, including validation/test, pads unavailable positions with zero after normalization, and can further mask training prefixes. LFP targets are final processed full-discharge capacity Q[-1]; NMC targets are independent RPT capacities. Proposed resampled fragment dimensions are a separate interface.

## Configuration and execution

Install the base environment and optional baseline dependencies:

```console
python -m pip install -r requirements-baselines.txt
python run.py check --baselines
```

Edit the chosen group's `config.py`:

```python
RUN_MODE = "train"               # "check" performs no training
SELECTED_MODELS = ("SSL-PL",)    # example for group 3
random_seeds = (0,)
ROUND_NUMBERS = (1,)             # None means all nested budgets
THREADS = 2
device = "cpu"
# NMC only:
DATASET_NAME = "NMC_A"           # or "NMC_B"
```

The example values replace the corresponding settings in that file; place DATASET_NAME before its derived preset values. Keep OUTPUT_DIR at its documented runs/ location or choose an external directory. New campaigns create fresh timestamped children below OUTPUT_DIR, preserving previous outputs and bundled artifacts.

Run without extra arguments:

```console
python "LFP dataset/model_construction/Baseline group 3/main.py"
python "Self-collected datasets/model_construction/Baseline group 3/main.py"
```

For other groups, edit/run that group's own files. To run both NMC datasets, perform one configured run for NMC_A, then select NMC_B in the configuration and run again. Each framework validates the supported method list.

## Training budget and bounded validation

Neural training uses `num_epochs` in config.py, along with batch size, optimizer, early stopping and the applicable scheduler. Early stopping monitors q50 validation MSE; there is no configurable metric-name label. Trees use estimator parameters in ML_CONFIG. Group 1 runs on CPU and has no neural `device` setting; group 4's `device` applies only to MLP/CNN. Formal group 4 LGBM defaults are 600 trees, depth 10, learning rate 0.03, min_child_samples 20 and random_state 42. Group 1's LightGBM settings remain specific to that comparison group.

For a short execution check, explicitly set one seed, one round and one epoch. For trees, explicitly reduce n_estimators/iterations in the selected estimator's configuration. Record these edits and use a separate output location. Such results verify execution only; they are not paper accuracy results. No runner silently reduces the data roles, tree count or epoch count.

Full LFP training has eight budgets: 100, 400, 700, 1000, 1300, 1600, 1900 and 2200 labels. NMC has seven: 30, 60, 90, 120, 150, 180 and 210. Validation/test roles remain excluded from fitting and acquisition. Normalizers are fitted on the appropriate labelled training examples. Selecting a single LFP round does not promise identical neural weights to reaching that round after earlier rounds in the same random stream.

## Outputs and scientific conventions

Campaign outputs record configured methods, seeds, budgets, data identity, effective configuration, predictions, normalization, splits and saved models. Neural best checkpoints are independent copies. Point-capacity supervision uses MSE; retained outer quantiles use pinball loss. NMC predictions are not clipped in normalized space, and residual standard deviations use ddof=0. LFP retains its dataset-specific prediction protocol, including group 3 clipping.

Internal helpers isolate each model group's local module names without changing the configured experiment. Baseline checkpoint formats remain method-specific; use their associated configuration and normalization, not the Proposed inference loader.

Neural constructors require their complete model dictionary from config.py. Input dimensions, single-capacity output width and quantiles are validated against the data and shared settings. Conflicting settings raise validation errors. MLP initialization is fixed Xavier-uniform for linear weights with zero biases; it is not a selectable initialization-method interface.

Use the [validation commands](../README.md#validation) to check configured model inputs and file integrity.
