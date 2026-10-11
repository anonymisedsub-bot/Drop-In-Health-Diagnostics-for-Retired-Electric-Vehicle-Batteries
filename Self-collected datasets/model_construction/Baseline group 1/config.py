"""Edit this file, then run main.py for NMC Baseline group 1."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_MODE = "train"  # "check" validates data and inputs without fitting models.
SELECTED_MODELS = ('XGB', 'RF', 'CGB', 'LGBM')
random_seeds = (0, 1, 2)
ROUND_NUMBERS = None  # None: every available round; or e.g. (1, 3, 7).
OUTPUT_DIR = PROJECT_ROOT / "runs" / "baseline_group_1"
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

# Tree estimators run on CPU; their thread counts are set in ML_CONFIG below.

# q50 is the MSE point output; q10/q90 retain their existing loss behavior.
QUANTILES = [0.1, 0.5, 0.9]

ML_CONFIG = {'rf': {'n_estimators': 600,
        'max_depth': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'n_jobs': THREADS,
        'random_state': 42},
 'xgb': {'n_estimators': 600,
         'max_depth': 5,
         'learning_rate': 0.03,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'min_child_weight': 1.0,
         'reg_lambda': 1.0,
         'tree_method': 'hist',
         'random_state': 42,
         'n_jobs': THREADS},
 'lgbm': {'n_estimators': 400,
          'max_depth': -1,
          'num_leaves': 31,
          'min_child_samples': 10,
          'min_split_gain': 0.0,
          'learning_rate': 0.05,
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
 'quantile': {'quantiles': QUANTILES, 'enabled_models': ['cgb', 'lgbm', 'xgb']}}
