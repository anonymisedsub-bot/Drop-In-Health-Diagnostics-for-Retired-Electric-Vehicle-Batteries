"""Editable LFP Baseline group 3 settings; packaged 2.50–2.73 V tail inputs."""

from __future__ import annotations

from pathlib import Path

import torch

# Run this group's main.py after editing these settings.
RUN_MODE = 'train'  # 'check' validates inputs without training.
SELECTED_MODELS = ('DA-AL', 'DA-MMD', 'DA-CORAL', 'SSL-PL')
ROUND_NUMBERS = None  # All nested budgets, or a tuple such as (1, 2).
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'baseline_group_3'
THREADS = 2


TAIL_LEN = 47

num_tr_initial = 100

num_tr_per_iter = 300

al_pool_ratio = 0.6

al_val_ratio = 0.2

random_seeds = (0, 1, 2)

normalization_mode = 'per_feature'

batch_size = 100

num_epochs = 250

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.005

lr_min = 1e-10

weight_decay = 0.0001

es_enable = True

es_mode = 'min'

es_patience = 30

es_min_delta = 0.0001

scheduler = {'policy': 'plateau', 'factor': 0.95, 'patience': 5, 'min_lr': lr_min, 'epoch_step': 10, 'step_gamma': 0.95}

QUANTILES = [0.1, 0.5, 0.9]

QUANTILE_LOSS_WEIGHTS = (0.0, 1.0, 0.0)

DA_ADV_WEIGHT = 0.1

DA_DISCRIMINATOR_HIDDEN_DIM = 64

DA_MMD_WEIGHT = 0.2

DA_MMD_BANDWIDTHS = (0.5, 1.0, 2.0)

DA_CORAL_WEIGHT = 0.2

knn_k = 10

knn_temperature = 0.1

MLP_CONFIG = {'hidden_dims': [512, 128, 32], 'activation': 'relu', 'dropout': 0.1, 'use_batchnorm': False, 'quantiles': QUANTILES, 'in_ch': 1, 'seq_len': TAIL_LEN, 'output_dim': 1}

DATA_MAT_PATH = Path(__file__).resolve().parents[2] / 'dataset' / 'LFP_data_processed.mat'
