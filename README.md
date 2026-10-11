# Drop-In Health Diagnostics for Retired Electric Vehicle Batteries

This package provides LFP and NMC datasets, diagnostic models, four baseline groups, sorting and regrouping simulation, and techno-economic-environmental assessment (TEA).

[Code guide](docs/CODE_GUIDE.md) · [Data dictionary](docs/DATA_DICTIONARY.md) · [Baseline methods](docs/BASELINES.md) · [Data and licences](AVAILABILITY.md) · [Citation](CITATION.md)

## Contents

| Directory | Included content |
| --- | --- |
| `LFP dataset/` | 4000 raw records, 3511 processed cells, MATLAB preprocessing, dataset figures, Proposed model and four baseline groups |
| `Self-collected datasets/` | NMC_A (353 cells), NMC_B (334 cells), 687 two-sheet workbooks, processed MATs, sample maps, preprocessing, figures, Proposed model and four baseline groups |
| `Sorting and regrouping simulation/` | LFP model inference, five sorting policies, paired bootstrap, eight-cell series regrouping, reference results and Figure 7 |
| `Techno-economic-environmental assessment/` | Targeted-RPT parameters, Monte Carlo and sensitivity analysis, reference results and Figure 8 |
| `docs/`, `tools/` | Scientific definitions, configuration guides and environment/integrity checks |

Each dataset directory contains `dataset/`, `dataset_visualization/` and `model_construction/`. Each `model_construction/` contains five peer directories: `Proposed framework/` and `Baseline group 1/` through `Baseline group 4/`. Each model directory has its own `main.py`, `config.py` and README.

## Installation

Use Python 3.12. Create and activate a virtual environment, then install the CPU dependencies:

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in Linux/macOS.

```console
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-baselines.txt
python run.py check --baselines
```

On macOS, install PyTorch with `python -m pip install torch==2.6.0`. For CUDA, choose the appropriate PyTorch 2.6.0 build using the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). Model `device` settings select CPU or CUDA. MATLAB is needed to regenerate the LFP processed MAT; the Python workflows can use the supplied processed files directly.

## Configure and run models

Edit the selected model directory's `config.py`, then run its `main.py` without command-line options. `RUN_MODE="train"` trains the model; `RUN_MODE="check"` checks the configured inputs. Proposed also supports `RUN_MODE="infer"` for saved checkpoints.

```console
python "LFP dataset/model_construction/Proposed framework/main.py"
python "Self-collected datasets/model_construction/Proposed framework/main.py"
python "LFP dataset/model_construction/Baseline group 1/main.py"
python "Self-collected datasets/model_construction/Baseline group 3/main.py"
```

`DATASET_NAME` selects `NMC_A` or `NMC_B` in NMC configurations. `random_seeds`, `ROUND_NUMBERS`, `OUTPUT_DIR` and `THREADS` control the campaign. Neural training uses `num_epochs` and `device`; tree estimators use `ML_CONFIG`. Baseline `SELECTED_MODELS` selects methods within the chosen group. `ROUND_NUMBERS=None` runs all label-budget rounds.

Training reads the adjacent configuration. Saved-checkpoint inference reads the model settings, splits and normalization saved with that checkpoint. See the [code guide](docs/CODE_GUIDE.md) and each model README for details.

## Baseline groups

| Directory | Methods | Input |
| --- | --- | --- |
| Baseline group 1 | XGB, RF, CGB, LGBM | Discharge tail |
| Baseline group 2 | MLP, CNN, LSTM, Transformer | Discharge tail |
| Baseline group 3 | DA-AL, DA-MMD, DA-CORAL, SSL-PL | Discharge tail |
| Baseline group 4 | MLP, CNN, CGB, LGBM | Tail plus padded upstream input |

SSL-PL uses weighted KNN pseudo-labels in normalized tail space. Groups 1–3 use 47 tail points for LFP, 61 for NMC_A and 91 for NMC_B. Group 4 adds 60 upstream positions. Proposed uses 64 resampled tail points and 48 points per upstream fragment.

## Datasets

| Dataset | Processed cells | Model tail | Upstream range |
| --- | ---: | --- | --- |
| LFP | 3511 | 2.50–2.73 V | 2.735–3.030 V |
| NMC_A | 353 | 3.00–3.30 V | 3.30–3.60 V |
| NMC_B | 334 | 2.75–3.20 V | 3.20–3.50 V |

LFP data are attributed to CETL; see [citation information](CITATION.md). NMC workbooks contain `DisC` and `DisCC2`, and public sample IDs follow processed-MAT column order. The preprocessing script reconstructs eight specified arrays from these workbooks. Model scripts use the supplied processed MAT files. See the [data dictionary](docs/DATA_DICTIONARY.md) for schemas, units and sample indexing.

LFP includes a one-epoch demonstration checkpoint. NMC includes saved Proposed checkpoints for the three seeds and seven label budgets. Regrouping uses its supplied LFP inputs and model; TEA evaluates the stated economic and environmental scenarios.

## Simulation and figures

```console
python run.py regrouping --mode figures
python run.py tea figures
```

Dataset figures use each dataset's `dataset_visualization/plot.py`. See the [figure guide](docs/FIGURE_OUTPUTS.md), [regrouping README](Sorting%20and%20regrouping%20simulation/README.md) and [TEA README](Techno-economic-environmental%20assessment/README.md) for simulations and output locations. Quote paths containing spaces.

## Validation

```console
python run.py verify
```

The integrity command checks the supplied file inventory. To inspect configured model inputs, set `RUN_MODE="check"` in the selected model configuration and run its `main.py`. Proposed `RUN_MODE="infer"` evaluates saved weights; NMC inference also compares predictions with each checkpoint's supplied prediction table. Simulation modules validate their numerical outputs during execution. The verified environment is Windows 11 with Python 3.12.7 and CPU execution; this scope does not establish full manuscript-result reproduction or execution on other platforms.

## Reuse

Software uses the MIT licence. LFP and NMC data are for academic research, with commercial use prohibited. Dataset terms and attribution are described in [AVAILABILITY.md](AVAILABILITY.md).
