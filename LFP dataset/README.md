# LFP dataset

This directory contains the CETL-HFUT LFP cohort: 4,000 raw records in `dataset/CETL-data.mat` and 3,511 retained records in `dataset/LFP_data_processed.mat`. These are two forms of the same cohort. The processed voltage/capacity matrices have shape 201 × 3,511; voltage is V and capacity is Ah. The target is the final processed discharge capacity, not an independent RPT measurement. Nominal capacity for descriptive SOH is 15.5 Ah.

## Run a model

Install the dependencies using the root [installation guide](../README.md#installation). Edit the corresponding `config.py`, then execute that model's `main.py` directly. Paths are resolved from the source files, so execution does not depend on the working directory. From the repository root:

```console
python "LFP dataset/model_construction/Proposed framework/main.py"
python "LFP dataset/model_construction/Baseline group 1/main.py"
python "LFP dataset/model_construction/Baseline group 2/main.py"
python "LFP dataset/model_construction/Baseline group 3/main.py"
python "LFP dataset/model_construction/Baseline group 4/main.py"
```

Every model entry defaults to training and accepts no command-line arguments. Proposed settings are documented in [model_construction/Proposed framework/README.md](model_construction/Proposed%20framework/README.md); baseline settings are in each group's `config.py` and [the baseline guide](../docs/BASELINES.md).

For a bounded Proposed run, set these values explicitly in its `config.py`:

```python
RUN_MODE = "train"
num_epochs = 1
random_seeds = (0,)
ROUND_NUMBERS = (1,)
device = "cpu"
THREADS = 2
```

All other configuration settings remain in effect, including the full dataset and split sizes. A one-epoch run checks software execution; it does not establish paper accuracy. New campaigns use a unique timestamp under `OUTPUT_DIR`, defaulting to `runs/Proposed/`, and save effective configuration, checkpoints, splits, normalizers, initial voltages and predictions. Within the repository, output is permitted under this dataset's `runs/` or the repository-root `runs/`; external output directories are also supported.

## Proposed protocol and inference

Default training uses 250 joint-training epochs at each of eight nested budgets (100, 400, …, 2,200 labels), for seeds 0, 1 and 2. The 3,511-cell split contains 100 initial labels, 2,100 pool cells, 702 validation cells and 609 test cells. Each budget fits an independent model. Normalizers use only the currently labelled source cells.

The tail spans 2.50–2.73 V and is resampled to 64 points. Upstream 2.735–3.03 V is divided into six disjoint windows, each resampled to 48 points. Features are Q and dQ/dV. The model uses gated fusion, FiLM, a learned zero-fragment vector, reconstruction, capacity MSE, quantile losses and CORAL with dynamic loss weighting. Disabling a loss through `loss_weights.enabled` keeps its weight zero throughout training. Initial-voltage simulation restricts pool and test availability. Source training samples fragment counts through the `missing_sim` curriculum; validation sweeps all fragment counts. Source and validation inputs are not truncated by simulated initial voltage. The normal draw rejects voltages at or below the tail boundary and has no upper cutoff.

Set `RUN_MODE = "check"` to inspect data without training. Set `RUN_MODE = "infer"` to use `CHECKPOINT`, which defaults to the bundled `model_construction/Proposed framework/demo/best.pt`. `INFERENCE_INDICES` selects processed zero-based columns in the requested order; `INFERENCE_K` selects a prefix length, or `None` retains saved simulated availability. The checkpoint's geometry and model settings are read into a separate immutable namespace. Inference never changes the editable training configuration and checks the MAT hash before predicting.

The bundled demo checkpoint is an inference example trained for one epoch on 32 source, 16 pool, 16 validation and 16 test records. It does not represent the full 250-epoch protocol or establish paper accuracy.

Inference exports processed zero-based and MATLAB one-based indices, reference capacities, predicted capacities and quantiles. Raw one-based record IDs are included only when the MAT hash matches the packaged data. Quantiles have no guaranteed calibrated coverage; training exports retain Gaussian-derived 95% prediction intervals estimated from quantile widths, which are not validated coverage estimates.

## Data preparation and figures

Run figures independently of model training:

```console
python "LFP dataset/dataset_visualization/plot.py"
python "LFP dataset/dataset_visualization/plot.py" --only capacity correlation
```

The nine visualization notebooks cover capacity/SOH, raw and processed curves, IC, voltage segments, SOC, t-SNE and correlation. See [the visualization guide](dataset_visualization/README.md) for PNG/PDF/SVG exports and numeric source tables.

Run `dataset/data_processing_code_V1.m` in MATLAB to prepare the data. The preprocessing retains 3,511 of 4,000 raw cells after 487 first-stage removals and two second-stage removals; `cell_index_mapping.csv` links processed columns to raw records in order. See the [data guide](dataset/README.md) for units and files, and the root [validation summary](../README.md#validation) for validation details.

The model uses native PyTorch mean aggregation. See the root [licensing](../LICENSE) and [data-availability](../AVAILABILITY.md) documents; the code licence does not grant unrestricted rights to the data.
