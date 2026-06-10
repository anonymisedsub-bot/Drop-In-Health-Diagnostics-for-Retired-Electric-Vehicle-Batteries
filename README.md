# Drop-In Health Diagnostics for Retired Electric Vehicle Batteries

This repository contains the data preprocessing, dataset visualization, and model construction workflows for retired electric vehicle battery health diagnosis.

The project is organized into three main folders:

| Folder | Purpose |
| --- | --- |
| `Data preprocessing` | MATLAB workflow for converting the raw CETL battery dataset into processed voltage-capacity matrices. |
| `Dataset visualization` | Jupyter notebooks for reproducing dataset and feature-distribution figures. |
| `Model construction` | PyTorch implementations of the proposed model and ablation variants. |

## Repository Structure

```text
.
|-- Data preprocessing
|   |-- CETL-data.mat
|   |-- CETL_data_processed.mat
|   |-- data_processing_code_V1.m
|   |-- README.md
|   `-- LICENSE
|-- Dataset visualization
|   |-- notebook_setup.py
|   |-- README.md
|   `-- LICENSE
`-- Model construction
    |-- Proposed (Gated fusion+FiLM)
    |-- Ab1 (FiLM_only)
    |-- Ab2 (tail_only)
    |-- Ab3 (CA_only)
    |-- README.md
    `-- LICENSE
```

## Workflow

1. Run the preprocessing script in `Data preprocessing` to generate `CETL_data_processed.mat`.
2. Use the notebooks in `Dataset visualization` to inspect the dataset and regenerate figures.
3. Run the model scripts in `Model construction` to train and evaluate the proposed model and ablations.

## Data

The raw and processed CETL data files are handled by the preprocessing workflow:

```text
Data preprocessing/CETL-data.mat
Data preprocessing/CETL_data_processed.mat
```

The model scripts expect the processed file:

```text
CETL_data_processed.mat
```

Each model folder resolves `DATA_MAT_PATH` from the repository structure. The default is:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MAT_PATH = Path(
    os.environ.get(
        "CETL_DATA_MAT_PATH",
        PROJECT_ROOT / "Dataset visualization" / "data" / "CETL_data_processed.mat",
    )
)
```

This is equivalent to using the project-relative data location:

```text
Dataset visualization/data/CETL_data_processed.mat
```

For visualization notebooks, either place the data files in:

```text
Dataset visualization/data/
```

or set the environment variable `BATTERY_DATA_DIR` to the folder containing the `.mat` files.

## Software Requirements

### Data Preprocessing

The preprocessing workflow is implemented in MATLAB.

Recommended:

```text
MATLAB R2021a or later
```

### Dataset Visualization

The visualization notebooks were checked with:

```text
Python 3.12.7
ipykernel 6.29.5
matplotlib 3.10.1
numpy 2.1.3
scipy 1.15.2
scikit-learn 1.6.1
```

### Model Construction

The model scripts require:

| Package | Version |
| --- | --- |
| `torch` | `2.6.0+cu126` |
| `torch_scatter` | `2.1.2+pt26cu124` |
| `numpy` | `2.1.3` |
| `scipy` | `1.15.2` |
| `pandas` | `2.2.3` |

`openpyxl` is also required for Excel result export through `pandas.ExcelWriter(..., engine="openpyxl")`.

## Running Preprocessing

Open MATLAB in the `Data preprocessing` folder and run:

```matlab
run('data_processing_code_V1.m')
```

This generates the processed matrices used by the downstream visualization and modeling code.

## Running Model Experiments

Run `main.py` from the target model folder, for example:

```powershell
cd "Model construction\Proposed (Gated fusion+FiLM)"
python main.py
```

Available model variants:

```text
Model construction/Proposed (Gated fusion+FiLM)
Model construction/Ab1 (FiLM_only)
Model construction/Ab2 (tail_only)
Model construction/Ab3 (CA_only)
```

The current shared model setting is:

```python
VSEG_LOW = 2.73
```

This gives:

```text
TAIL_LEN_ORIGIN = 46
```

Model outputs are saved inside each model subfolder under:

```text
Results_Vtail_46_seed_0
Results_Vtail_46_seed_1
Results_Vtail_46_seed_2
```

with model-specific subfolders such as:

```text
Proposed_random_seed0
Ab1_random_seed0
Ab2_random_seed0
Ab3_random_seed0
```

## Reproducibility Notes

- Model scripts set Python, NumPy, PyTorch, CUDA, cuDNN, and DataLoader random seeds.
- DataLoader shuffling uses explicit `torch.Generator` objects.
- CUDA kernels involving `torch_scatter` may still introduce small GPU-level numerical differences on some systems.
- For the strictest bit-level repeatability, run on CPU or verify deterministic behavior on the target GPU and driver stack.

## Module Documentation

See the module-level README files for more details:

- `Data preprocessing/README.md`
- `Dataset visualization/README.md`
- `Model construction/README.md`

## Data Availability

The raw dataset used in this study is publicly available at the GitHub repository "Imbalanced retired battery data" provided by CETL-HFUT (https://github.com/CETL-HFUT/Imbalanced-retired-battery-data). The processed data, data splits and intermediate features generated in this study will be made available upon publication in a public repository. The dataset is provided for academic use only, in accordance with the declaration of the original data provider.

## Code Availability

The code used for data preprocessing, dataset visualization, model construction, training, evaluation, and result export is provided in this repository. The repository includes MATLAB preprocessing code, Jupyter notebooks for figure generation, PyTorch model implementations, ablation-study scripts, documentation, and environment requirement files. A permanent public repository and DOI will be provided upon publication.

## License

The code and documentation in the three main folders are released under the MIT License. See the `LICENSE` file in each folder for details.

Dataset files may be subject to separate data-release terms if distributed independently.
