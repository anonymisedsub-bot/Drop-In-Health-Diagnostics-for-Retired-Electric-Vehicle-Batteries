"""Internal helper for the experiment pipeline."""
import os
from pathlib import Path

import torch


# Portable path to the processed CETL battery dataset used by all experiments.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MAT_PATH = Path(
    os.environ.get(
        "CETL_DATA_MAT_PATH",
        PROJECT_ROOT / "Dataset visualization" / "data" / "CETL_data_processed.mat",
    )
)

# Voltage grid definition: lower bound, upper bound, and sampling interval.
V_MIN, V_MAX, DV = 2.5, 3.5, 0.005
# Upper voltage boundary of discharge tail segment.
VSEG_LOW = 2.73
# Upper voltage boundary of upstream region.
VSEG_UPP = VSEG_LOW + 0.3
# Raw tail length
TAIL_LEN_ORIGIN = round((VSEG_LOW - V_MIN) / DV)
# Model input lengths: resampled tail length, resampled upstream segment length, and total number of upstream segments.
TAIL_LEN, SEG_LEN, MAX_SEGS = 64, 48, 6

# Initial labelled samples and randomly labelled samples per budget step.
num_tr_initial = 100
num_tr_per_iter = 300
# Fractions of non-initial samples reserved for labelling candidates and validation.
al_pool_ratio = 0.60
al_val_ratio = 0.20
# Labelling strategies to execute.
AL_STRATS = ["random"]

# Repeated random seeds.
random_seeds = [0, 1, 2]
# Mini-batch size for supervised and target-domain loaders.
batch_size = 100
# Normalization granularity for tail and segment inputs.
normalization_mode = "per_channel"

# Epoch budgets for optional unsupervised pretraining and final joint training.
num_pretrain_epochs = 0
num_finetune_epochs = 250
# Early-stopping patience measured in validation epochs.
patience_epochs = 250
# Runtime device; uses GPU when CUDA is available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Validation protocol for missing-segment robustness.
val_mode_pretrain = "sweep_k"
val_mode_finetune = "sweep_k"
# Extra validation weight assigned to the zero-segment case.
val_tail_weight = 1.0

# Optimizer learning rate, lower LR bound, and AdamW regularization.
lr = 5e-3
lr_min = 1e-10
weight_decay = 1e-4
# Legacy cosine schedule horizon; kept for configs using cosine_epoch.
cosine_T_max = 100

# Learning-rate scheduler parameters.
scheduler = dict(
    policy="plateau",
    factor=0.95,
    patience=5,
    min_lr=lr_min,
    epoch_step=10,
    step_gamma=0.95,
)

# Dynamic multi-loss weighting for reconstruction, capacity, quantile, and DA.
loss_weights = dict(
    mode="dynamic",
    da_mode="Coral",
    enabled=["recon", "cap_mse", "cap_pin", "da"],
    dynamic=dict(
        strategy="ewma_balance",
        ema_beta=0.9,
        floor=0.05,
        ceil=10.0,
        normalise=True,
        base=dict(recon=1.0, cap_mse=1.0, cap_pin=1.0, da=0.2),
        eps=1e-8,
    ),
)

# FiLM-only ablation architecture and feature switches.
model = dict(
    d_tail=[16, 64, 128],
    d_seg=[16, 64, 128],
    hidden_dim=128,
    d_meta=0,
    tail_only=False,
    use_gating=False,
    use_film=True,
    use_learned_zero=True,
    quantiles=[0.1, 0.5, 0.9],
)

# Missing upstream-segment simulation used during curriculum training.
missing_sim = dict(
    p_drop_all=0.1,
    p_keep_one=0.9,
    curriculum_epochs=[0, 50, 100, 150, 200],
    curriculum_keep=[1, 2, 3, 4, MAX_SEGS],
)
