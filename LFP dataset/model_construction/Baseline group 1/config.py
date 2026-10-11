"""Editable LFP Baseline group 1 settings; packaged 2.50–2.73 V tail inputs."""

from pathlib import Path

# Run this group's main.py after editing these settings.
RUN_MODE = 'train'  # 'check' validates inputs without training.
SELECTED_MODELS = ('XGB', 'RF', 'CGB', 'LGBM')
ROUND_NUMBERS = None  # All nested budgets, or a tuple such as (1, 2).
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'baseline_group_1'
THREADS = 2


TAIL_LEN = 47

num_tr_initial = 100

num_tr_per_iter = 300

al_pool_ratio = 0.6

al_val_ratio = 0.2

normalization_mode = 'per_feature'

random_seeds = (0, 1, 2)

QUANTILES = [0.1, 0.5, 0.9]

DATA_MAT_PATH = Path(__file__).resolve().parents[2] / 'dataset' / 'LFP_data_processed.mat'

# q50 uses squared-error regression; outer heads retain pinball losses.
ML_CONFIG = {
    'rf': {'n_estimators': 600, 'max_depth': 100, 'min_samples_split': 2,
           'min_samples_leaf': 1, 'n_jobs': THREADS, 'random_state': 42},
    'xgb': {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.03,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1.0,
            'reg_lambda': 1.0, 'tree_method': 'hist', 'random_state': 42, 'n_jobs': THREADS},
    'lgbm': {'n_estimators': 400, 'max_depth': -1, 'num_leaves': 31,
             'min_child_samples': 10, 'min_split_gain': 0.0, 'learning_rate': 0.05,
             'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': THREADS,
             'random_state': 42, 'objective': 'regression_l2'},
    'cgb': {'iterations': 1200, 'learning_rate': 0.03, 'depth': 6,
            'verbose': False, 'random_state': 42, 'thread_count': THREADS,
            'allow_writing_files': False},
    'quantile': {'quantiles': QUANTILES, 'enabled_models': ['cgb', 'lgbm', 'xgb']},
}
