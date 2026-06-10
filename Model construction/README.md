# Model Construction

This folder contains four model variants for battery health diagnostics:

- `Proposed (Gated fusion+FiLM)`
- `Ab1 (FiLM_only)`
- `Ab2 (tail_only)`
- `Ab3 (CA_only)`

Each subfolder is self-contained and includes:

- `config.py`: experiment settings and data path
- `data.py`: dataset loading and preprocessing
- `model.py`: model architecture
- `losses.py`: loss functions and regression metrics
- `train_model.py`: training and evaluation routines
- `main.py`: experiment entry point

## Python Environment

Required package versions:

| Package | Version |
|---|---|
| `torch` | `2.6.0+cu126` |
| `torch_scatter` | `2.1.2+pt26cu124` |
| `numpy` | `2.1.3` |
| `scipy` | `1.15.2` |
| `pandas` | `2.2.3` |

The scripts also write Excel files through `pandas.ExcelWriter(..., engine="openpyxl")`, so `openpyxl` must be installed for full result export.

## Data Path

Each model's `config.py` resolves `DATA_MAT_PATH` from the repository structure. The default is:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MAT_PATH = Path(
    os.environ.get(
        "CETL_DATA_MAT_PATH",
        PROJECT_ROOT / "Dataset visualization" / "data" / "CETL_data_processed.mat",
    )
)
```

This points to the project-relative data file:

```text
Dataset visualization/data/CETL_data_processed.mat
```

The expected `.mat` file is:

```text
CETL_data_processed.mat
```

## Shared Experiment Setting

All model variants currently use:

```python
VSEG_LOW = 2.73
```

With the current voltage-grid settings, this gives:

```text
TAIL_LEN_ORIGIN = 46
```

## Running Experiments

Run `main.py` from the desired model subfolder, for example:

```powershell
cd "Model construction\Proposed (Gated fusion+FiLM)"
python main.py
```

The configured random seeds are:

```python
random_seeds = [0, 1, 2]
```

## Output Folders

Each `main.py` saves results relative to its own model subfolder, independent of the shell working directory.

For the current `VSEG_LOW = 2.73` setting, outputs are written under:

```text
Results_Vtail_46_seed_0
Results_Vtail_46_seed_1
Results_Vtail_46_seed_2
```

The model-specific result subfolders are:

```text
Ab1_random_seed{seed}
Ab2_random_seed{seed}
Ab3_random_seed{seed}
Proposed_random_seed{seed}
```

Each result subfolder may contain:

```text
best.pt
log.jsonl
summary_performance.xlsx
pool_indices.npy
metrics.json
iter_*/al_results.xlsx
iter_*/summary_iter.json
al_summary_all_iters.json
splits_init.json
V_init.npy
```

## Code Availability

The model construction, training, evaluation, ablation-study, loss-function, and result-export code is provided in this folder. The released version includes the proposed model, three ablation variants, configuration files, training scripts, and environment requirements. A permanent public repository and DOI will be provided upon publication.

## Data Availability

The raw dataset used in this study is publicly available at the GitHub repository "Imbalanced retired battery data" provided by CETL-HFUT (https://github.com/CETL-HFUT/Imbalanced-retired-battery-data). The processed data, data splits and intermediate features generated in this study will be made available upon publication in a public repository. The dataset is provided for academic use only, in accordance with the declaration of the original data provider.
