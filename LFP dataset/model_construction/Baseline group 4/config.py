"""Editable LFP Baseline group 4 settings; 2.50–2.73 V tail plus upstream inputs."""

from pathlib import Path

import torch

# Run this group's main.py after editing these settings.
RUN_MODE = 'train'  # 'check' validates inputs without training.
SELECTED_MODELS = ('MLP', 'CNN', 'CGB', 'LGBM')
ROUND_NUMBERS = None  # All nested budgets, or a tuple such as (1, 2).
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'baseline_group_4'
THREADS = 2


V_MIN, V_MAX, DV = (2.5, 3.5, 0.005)

VSEG_LOW = 2.73

VSEG_UPP = VSEG_LOW + 0.3

TAIL_LEN = int(round((VSEG_LOW - V_MIN) / DV)) + 1

UPSTREAM_LEN = int(round((VSEG_UPP - VSEG_LOW) / DV))

segment_length = TAIL_LEN + UPSTREAM_LEN

UPSTREAM_START_IDX = TAIL_LEN

MAX_SEGS = 6

SUBSEG_LEN = UPSTREAM_LEN // MAX_SEGS

num_tr_initial = 100

num_tr_per_iter = 300

al_pool_ratio = 0.6

al_val_ratio = 0.2

missing_sim_curriculum = dict(p_drop_all=0.1, p_keep_one=0.9, curriculum_epochs=[0, 50, 100, 150, 200], curriculum_keep=[1, 2, 3, 4, MAX_SEGS])

missing_sim_random = dict(p_drop_all=0.1, p_keep_one=0.9, curriculum_keep=MAX_SEGS)

normalization_mode = 'per_feature'

random_seeds = (0, 1, 2)

batch_size = 100

num_epochs = 250

device = 'cuda' if torch.cuda.is_available() else 'cpu'

es_enable = True

es_mode = 'min'

es_patience = 250

es_min_delta = 0.0001

lr = 0.005

lr_min = 1e-10

weight_decay = 0.0001

scheduler = dict(policy='plateau', factor=0.95, patience=5, min_lr=lr_min, epoch_step=10, step_gamma=0.95)

QUANTILES = [0.1, 0.5, 0.9]

ML_CONFIG = {'lgbm': {'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.03, 'min_child_samples': 20, 'random_state': 42, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': THREADS}, 'cgb': {'iterations': 1200, 'learning_rate': 0.03, 'depth': 6, 'verbose': False, 'thread_count': THREADS, 'allow_writing_files': False}, 'quantile': {'quantiles': QUANTILES, 'enabled_models': ['cgb', 'lgbm']}}

dropout_rate = 0.2

MLP_CONFIG = {'hidden_dims': [512, 128, 32], 'activation': 'relu', 'dropout': dropout_rate, 'use_batchnorm': False, 'quantiles': QUANTILES, 'in_ch': 1, 'seq_len': segment_length, 'output_dim': 1}

CNN1D_CONFIG = {'conv_channels': [32, 64, 128], 'kernel_sizes': [7, 5, 3], 'paddings': [3, 2, 1], 'use_batchnorm': False, 'quantiles': QUANTILES, 'activation': 'relu', 'use_global_pool': False, 'fc_dims': [256, 64, 8], 'use_dropout': True, 'dropout_rate': dropout_rate, 'in_ch': 1, 'seq_len': segment_length, 'output_dim': 1}

DATA_MAT_PATH = Path(__file__).resolve().parents[2] / 'dataset' / 'LFP_data_processed.mat'
