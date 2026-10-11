"""Edit this file, then run main.py for NMC Baseline group 4."""
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_MODE = "train"  # "check" validates data and inputs without fitting models.
SELECTED_MODELS = ('MLP', 'CNN', 'CGB', 'LGBM')
random_seeds = (0, 1, 2)
ROUND_NUMBERS = None  # None: every available round; or e.g. (1, 3, 7).
OUTPUT_DIR = PROJECT_ROOT / "runs" / "baseline_group_4"
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

UPSTREAM_LEN = int(round((VSEG_UPP - VSEG_LOW) / DV))
segment_length = TAIL_LEN + UPSTREAM_LEN
UPSTREAM_START_IDX = TAIL_LEN
MAX_SEGS = 6
SUBSEG_LEN = UPSTREAM_LEN // MAX_SEGS

missing_sim_curriculum = dict(
    p_drop_all=0.1,
    p_keep_one=0.9,
    curriculum_epochs=[0, 50, 100, 150, 200],
    curriculum_keep=[1, 2, 3, 4, MAX_SEGS],
)

missing_sim_random = dict(
    p_drop_all=0.1,
    p_keep_one=0.9,
    curriculum_keep=MAX_SEGS,
)

# Training budgets are used exactly as written, including short verification runs.
batch_size = 100
num_epochs = 250
device = "cuda" if torch.cuda.is_available() else "cpu"  # Neural models only; trees use CPU.
es_enable = True
# Early stopping always monitors validation MSE of the q50 point-output slot.
es_mode = "min"
es_patience = 250
es_min_delta = 1e-4
lr = 5e-3
lr_min = 1e-10
weight_decay = 1e-4
scheduler = dict(policy="plateau", factor=0.95, patience=5, min_lr=lr_min,
                 epoch_step=10, step_gamma=0.95)

# q50 is the MSE point output; q10/q90 retain their existing loss behavior.
QUANTILES = [0.1, 0.5, 0.9]

ML_CONFIG = {'lgbm': {'n_estimators': 600,
          'max_depth': 10,
          'num_leaves': 31,
          'min_child_samples': 20,
          'min_split_gain': 0.0,
          'learning_rate': 0.03,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'n_jobs': THREADS,
          'random_state': 42,
          'objective': 'regression_l2'},
 'cgb': {'iterations': 1200,
         'learning_rate': 0.03,
         'depth': 6,
         'verbose': False,
         'random_state': 42,
         'thread_count': THREADS,
         'allow_writing_files': False},
 'quantile': {'quantiles': QUANTILES, 'enabled_models': ['cgb', 'lgbm']}}

dropout_rate = 0.2

MLP_CONFIG = {
    # Network architecture
    'hidden_dims': [512, 128, 32],  # Hidden dimension of each layer
    'activation': 'relu',                   # Activation function
    'dropout': dropout_rate,                         # Dropout rate
    'use_batchnorm': False,                 # Enable BatchNorm (off by default)
    'quantiles': QUANTILES,

    # MLP linear layers use Xavier-uniform weights and zero biases.
    
    # Input settings
    'in_ch': 1,                             # Input channels
    'seq_len': segment_length,              # Sequence length
    # Each quantile head produces one value, so output_dim remains 1.
    # Total output dimension = len(quantiles) * output_dim = 3 * 1 = 3.
    'output_dim': 1,                        # Output dimension per head
}

CNN1D_CONFIG = {
    # Convolutional layers
    'conv_channels': [32, 64, 128],  # Output channels of each convolutional layer
    'kernel_sizes': [7, 5, 3],       # Kernel size of each convolutional layer
    'paddings': [3, 2, 1],           # Padding of each convolutional layer
    'use_batchnorm': False,          # Enable BatchNorm (off by default)
    'quantiles': QUANTILES,
    'activation': 'relu',            # Activation function
    'use_global_pool': False,        # Enable global average pooling (off by default)
    
    # Fully connected layers
    'fc_dims': [256, 64, 8],         # Fully connected layer dimensions
    'use_dropout': True,             # Enable dropout
    'dropout_rate': dropout_rate,             # Dropout rate
    
    # Input and output settings
    'in_ch': 1,                      # Input channels
    'seq_len': segment_length,       # Sequence length
    # Each quantile head produces one value, so output_dim remains 1.
    # Total output dimension = len(quantiles) * output_dim = 3 * 1 = 3.
    'output_dim': 1,                 # Output dimension per head
}
