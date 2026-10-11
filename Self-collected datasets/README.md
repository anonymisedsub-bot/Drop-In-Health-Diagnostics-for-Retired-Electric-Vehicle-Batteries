# Self-collected NMC_A and NMC_B datasets

NMC_A and NMC_B data are for academic research only; commercial use is not permitted. See [the data terms](dataset/DATA_LICENSE.md). The software retains its existing MIT licence.

| Dataset | Processed cells | Tail voltage | Raw tail points | Proposed tail / fragment length |
| --- | --- | --- | --- | --- |
| NMC_A | 353 | 3.00–3.30 V | 61 | 64 / 48 |
| NMC_B | 334 | 2.75–3.20 V | 91 | 64 / 48 |

Both datasets use zero voltage removal and a 0.30 V upstream region divided into six fragments. Raw tail counts include both endpoints; model input lengths are resampled dimensions. Complete-tail screening precedes the seeded split, so cohort counts are not test-set sizes. Capacities are Ah.

## Run a model

Edit the corresponding `config.py`, then run its `main.py` without arguments. From the release root:

```console
python "Self-collected datasets/model_construction/Proposed framework/main.py"
python "Self-collected datasets/model_construction/Baseline group 1/main.py"
python "Self-collected datasets/model_construction/Baseline group 2/main.py"
python "Self-collected datasets/model_construction/Baseline group 3/main.py"
python "Self-collected datasets/model_construction/Baseline group 4/main.py"
```

Every entry defaults to training. Set `DATASET_NAME = "NMC_A"` or `"NMC_B"` in that model's config before execution. Proposed's two presets are visible in [model_construction/Proposed framework/config.py](model_construction/Proposed%20framework/config.py); changing the dataset selects its data path and voltage boundaries when the file loads. Other training settings remain directly editable. Training uses the current configuration.

For one bounded run, explicitly set `num_epochs = 1`, `random_seeds = (0,)`, `ROUND_NUMBERS = (1,)`, `device = "cpu"` and `THREADS = 2`. All other configured settings and sample membership remain in effect. These short runs verify execution, not paper accuracy. See [the Proposed guide](model_construction/Proposed%20framework/README.md) and [baseline guide](../docs/BASELINES.md).

Default Proposed training fits independent models at seven nested budgets from 30 to 210 labels, for seeds 0, 1 and 2, using 250 joint-training epochs per budget. It retains fixed validation/test splits, labelled-only normalization, gated fusion, FiLM, reconstruction, capacity and quantile losses, CORAL and dynamic loss balancing. Labelled and unlabelled samples draw fragment prefixes independently, bounded by each cell's measured availability. Unlabelled capacities are not exposed to training. Validation uses all actually observed complete fragments.

New Proposed campaigns are written below `runs/Proposed/<timestamp>/<dataset>_seed<seed>/iter_<round>/`. They include effective configuration, split and budget manifests, input protocol, normalizers, independent best weights, logs, metrics and prediction spreadsheets. Outputs within the release are restricted to this dataset's `runs/` or root `runs/`; external destinations are also supported.

## Saved-weight inference

Set `RUN_MODE = "infer"` and run the same `main.py`. With `CHECKPOINT = None`, `DATASET_NAME`, `random_seeds` and `ROUND_NUMBERS` automatically select bundled checkpoints; `ROUND_NUMBERS = None` selects all seven budgets. Run each dataset with all three default seeds to check all 42 bundled checkpoints. Set `CHECKPOINT` to a trained `iter_<round>/best.pt` to replay that run; dataset, seed and round then come from its saved manifests.

The selected snapshot is loaded into a separate immutable namespace, leaving current training settings unchanged. Inference preserves saved model geometry and dataset identity, checks the processed MAT hash, and compares predictions with that checkpoint's saved test table. The allowed difference is 0.0002 Ah. `device` and `THREADS` remain explicit runtime settings. CSV and validation JSON filenames include dataset, seed and round.

Set `RUN_MODE = "check"` to validate configured data and splits without training. Runtime paths resolve relative to the source files and work from an arbitrary working directory.

## Data identity, figures and baselines

Models read the supplied `dataset/NMC_A_data_processed.mat` and `dataset/NMC_B_data_processed.mat`. Raw two-sheet workbooks are under `dataset/raw/NMC_A/` and `dataset/raw/NMC_B/`; their public IDs restart at one in each dataset and equal the one-based processed MAT column. `dataset/raw/source_mapping.csv` links public IDs, column order and file hashes. Each workbook contains `DisC` and `DisCC2`. The [preprocessing guide](preprocessing/README.md) lists the eight arrays built from these sheets and the separate outputs `NMC_A_from_raw.mat` and `NMC_B_from_raw.mat`. Public IDs are not manufacturer serial numbers.

Keep each bundled checkpoint with its supplied configuration, manifests and matching processed MAT. The inference entry point selects these files automatically. Native PyTorch mean aggregation supports the trained parameter names without requiring compiled `torch_scatter`.

Run Figure 3 independently:

```console
python "Self-collected datasets/dataset_visualization/plot.py"
```

The figures use the exact 353/334-cell cohorts; the joint initial-voltage panel additionally reads the packaged LFP cohort. See [the visualization guide](dataset_visualization/README.md) for panel selection, formats and numeric definitions.

Baseline groups retain their separate scientific implementations: group 1 trees, group 2 neural regressors, group 3 domain adaptation/pseudo-labelling, and group 4 padded tail/upstream models. Their raw tail lengths are 61/91 and padded lengths 121/151. Each group's parameters live in its own config. Optional tree dependencies are listed in `../requirements-baselines.txt`.

See the [installation instructions](../README.md#installation), [validation checks](../README.md#validation) and [Proposed comparison](../docs/PROPOSED_FRAMEWORK_COMPARISON.md).
