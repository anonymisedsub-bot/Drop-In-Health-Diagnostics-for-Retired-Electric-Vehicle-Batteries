"""Edit this file, then run main.py for NMC Baseline group 3."""
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_MODE = "train"  # "check" validates data and inputs without fitting models.
SELECTED_MODELS = ('DA-AL', 'DA-MMD', 'DA-CORAL', 'SSL-PL')
random_seeds = (0, 1, 2)
ROUND_NUMBERS = None  # None: every available round; or e.g. (1, 3, 7).
OUTPUT_DIR = PROJECT_ROOT / "runs" / "baseline_group_3"
THREADS = 2

# Both public MAT presets live here. Selecting a name derives all dimensions.
DATASET_NAME = "NMC_A"  # "NMC_A" or "NMC_B"
DATASET_PRESETS = {
    "NMC_A": {"SOURCE_BATCH": 4, "MODEL_BATCH": 4, "V_MIN": 3.0,
              "VSEG_LOW": 3.30, "VOLTAGE_REMOVAL_WINDOW": 0.0},
    "NMC_B": {"SOURCE_BATCH": 6, "MODEL_BATCH": 3, "V_MIN": 2.75,
              "VSEG_LOW": 3.20, "VOLTAGE_REMOVAL_WINDOW": 0.0},
}
if DATASET_NAME not in DATASET_PRESETS:
    raise ValueError("DATASET_NAME must be NMC_A or NMC_B")
_preset = DATASET_PRESETS[DATASET_NAME]
DATA_MAT_PATH = PROJECT_ROOT / "dataset" / f"{DATASET_NAME}_data_processed.mat"
SOURCE_BATCH = _preset["SOURCE_BATCH"]
MODEL_BATCH = _preset["MODEL_BATCH"]
V_MIN = _preset["V_MIN"]
V_MAX = 4.20
DV = 0.005
VSEG_LOW = _preset["VSEG_LOW"]
VSEG_UPP = VSEG_LOW + 0.30
VOLTAGE_REMOVAL_WINDOW = _preset["VOLTAGE_REMOVAL_WINDOW"]
TAIL_LEN = int(round((VSEG_LOW - V_MIN) / DV)) + 1

# Seeded within-batch split and 30-sample random acquisition.
POOL_TARGET_RATIO = 0.60
POOL_DIVISOR = 30
INITIAL_LABELLED_COUNT = 30
ACQUISITION_SIZE = 30
normalization_mode = "per_feature"

# Training budgets are used exactly as written, including short verification runs.
batch_size = 100
num_epochs = 250
device = "cuda" if torch.cuda.is_available() else "cpu"
es_enable = True
# Early stopping always monitors q50 validation MSE.
es_mode = "min"
es_patience = 30
es_min_delta = 1e-4
lr = 5e-3
lr_min = 1e-10
weight_decay = 1e-4
scheduler = dict(policy="plateau", factor=0.95, patience=5, min_lr=lr_min,
                 epoch_step=10, step_gamma=0.95)

# q50 is the MSE point output; q10/q90 retain their existing loss behavior.
QUANTILES = [0.1, 0.5, 0.9]

QUANTILE_LOSS_WEIGHTS = (0.0, 1.0, 0.0)
# SSL-PL: temperature-weighted KNN directly on normalized tail features.
knn_k = 10
knn_temperature = 0.1
knn_metric = "euclidean"
knn_pseudo_weight = 1.0
DA_ADV_WEIGHT = 0.1
DA_DISCRIMINATOR_HIDDEN_DIM = 64
DA_MMD_WEIGHT = 0.2
DA_MMD_BANDWIDTHS = (0.5, 1.0, 2.0)
DA_CORAL_WEIGHT = 0.2

dropout_rate = 0.1

# MLP linear layers use Xavier-uniform weights and zero biases.
MLP_CONFIG = {
    'hidden_dims': [512, 128, 32],
    'activation': 'relu',
    'dropout': dropout_rate,
    'use_batchnorm': False,
    'quantiles': QUANTILES,
    'in_ch': 1,
    'seq_len': TAIL_LEN,
    'output_dim': 1,
}
