"""Editable LFP Baseline group 2 settings; packaged 2.50–2.73 V tail inputs."""

from pathlib import Path

import torch

# Run this group's main.py after editing these settings.
RUN_MODE = 'train'  # 'check' validates inputs without training.
SELECTED_MODELS = ('MLP', 'CNN', 'LSTM', 'Transformer')
ROUND_NUMBERS = None  # All nested budgets, or a tuple such as (1, 2).
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'baseline_group_2'
THREADS = 2


TAIL_LEN = 47

num_tr_initial = 100

num_tr_per_iter = 300

al_pool_ratio = 0.6

al_val_ratio = 0.2

normalization_mode = 'per_feature'

random_seeds = (0, 1, 2)

batch_size = 100

num_epochs = 250

device = 'cuda' if torch.cuda.is_available() else 'cpu'

es_enable = True

es_mode = 'min'

es_patience = 30

es_min_delta = 0.0001

lr = 0.005

lr_min = 1e-10

weight_decay = 0.0001

scheduler = dict(policy='plateau', factor=0.95, patience=5, min_lr=lr_min, epoch_step=10, step_gamma=0.95)

QUANTILES = [0.1, 0.5, 0.9]

dropout_rate = 0.1

MLP_CONFIG = {'hidden_dims': [512, 128, 32], 'activation': 'relu', 'dropout': dropout_rate, 'use_batchnorm': False, 'quantiles': QUANTILES, 'in_ch': 1, 'seq_len': TAIL_LEN, 'output_dim': 1}

CNN1D_CONFIG = {'conv_channels': [32, 64, 128], 'kernel_sizes': [7, 5, 3], 'paddings': [3, 2, 1], 'use_batchnorm': False, 'quantiles': QUANTILES, 'activation': 'relu', 'use_global_pool': False, 'fc_dims': [256, 64, 8], 'use_dropout': True, 'dropout_rate': dropout_rate, 'in_ch': 1, 'seq_len': TAIL_LEN, 'output_dim': 1}

LSTM_CONFIG = {'use_tcn': True, 'tcn_channels': [32, 64], 'tcn_kernel_size': 3, 'tcn_dropout': dropout_rate, 'tcn_activation': 'relu', 'hidden_dims': [512, 128], 'num_layers': 2, 'bidirectional': False, 'lstm_dropout': dropout_rate, 'use_batchnorm': False, 'lstm_activation': 'relu', 'quantiles': QUANTILES, 'fc_dims': [64, 16], 'fc_activation': 'relu', 'fc_dropout': dropout_rate, 'use_fc_dropout': True, 'in_ch': 1, 'output_dim': 1}

TRANSFORMER_CONFIG = {'use_tcn': True, 'tcn_channels': [32, 64], 'tcn_kernel_size': 3, 'tcn_dropout': dropout_rate, 'tcn_activation': 'relu', 'd_model': 64, 'nhead': 8, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'max_seq_len': TAIL_LEN, 'attention_dropout': dropout_rate, 'use_pos_encoding': True, 'pos_encoding_type': 'sinusoidal', 'pooling_type': 'mean', 'output_proj_layers': [128, 64], 'output_activation': 'relu', 'output_dropout': dropout_rate, 'in_ch': 1, 'output_dim': 1, 'quantiles': QUANTILES}

DATA_MAT_PATH = Path(__file__).resolve().parents[2] / 'dataset' / 'LFP_data_processed.mat'
