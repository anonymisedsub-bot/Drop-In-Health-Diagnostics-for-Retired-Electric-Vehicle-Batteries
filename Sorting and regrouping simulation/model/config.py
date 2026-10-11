# config.py
"""
Configuration file for Battery Health Diagnosis Framework.
Designed for Active Learning and Domain Adaptation on retired battery datasets.
"""
from pathlib import Path
import torch

# ==============================================================================
# 1. Dataset & Paths
# ==============================================================================
# Resolve the bundled historical dataset independently of the working directory.
DATA_MAT_PATH = str(Path(__file__).resolve().parents[1] / 'inputs/LFP_data_processed.mat')

# Voltage threshold settings
V_MIN, V_MAX, DV = 2.5, 3.5, 0.005
VSEG_LOW = 2.73  # Bundled historical t46_i2_s0 checkpoint, not current R1 protocol.
VSEG_UPP = VSEG_LOW + 0.3
TAIL_LEN_ORIGIN = round((VSEG_LOW - V_MIN) / DV)
TAIL_LEN, SEG_LEN, MAX_SEGS = 64, 48, 6

# ==============================================================================
# 2. Active Learning Settings
# ==============================================================================
num_tr_initial = 100          # Initial number of labeled cells
num_tr_per_iter = 300         # Number of queried cells per active learning iteration

al_pool_ratio = 0.60          # ~60% cells for unlabeled pool
al_val_ratio  = 0.20          # ~20% cells for validation

# The historical campaign uses random active-learning queries.

# ==============================================================================
# 3. Training & Optimization Settings
# ==============================================================================
random_seeds = [0, 1, 2]
batch_size   = 100
normalization_mode = "per_channel"

num_finetune_epochs = 250     # Historical name for the joint-training epoch budget.
patience_epochs     = 250     # Early stopping patience
device = "cuda" if torch.cuda.is_available() else "cpu"

val_mode_finetune = 'sweep_k'  # 'sweep_k': all k; 'random_k': k=0 plus a sampled available k.
val_tail_weight   = 1.0        # Relative weight of k=0 (no upstream segments) in sweep evaluation.

# Optimizer & Scheduler
lr           = 5e-3
lr_min       = 1e-10
weight_decay = 1e-4

scheduler = dict(
    policy     = "plateau",     # "plateau" | "cosine_epoch"; other values disable scheduling.
    factor     = 0.95,          # Reduce factor for plateau
    patience   = 5,             # Patience for plateau
    min_lr     = lr_min,
    epoch_step = 10,            # Step decay interval
    step_gamma = 0.95,          # Step decay factor
)

# ==============================================================================
# 4. Loss Weights & Dynamic Controller
# ==============================================================================
# Keeps only: Tail Reconstruction, Capacity MSE, Capacity Quantile, and Domain Adaptation
loss_weights = dict(
    mode="dynamic",                 # "fixed" or "dynamic"
    da_mode='Coral',                # Domain Adaptation metric: 'Coral' or 'MMD'
    enabled=["recon", "cap_mse", "cap_pin", "da"],  
    dynamic=dict(
        # Loss balancing uses an exponentially weighted moving average.
        ema_beta=0.9,               
        floor=0.05,                 
        ceil=10.0,                  
        normalise=True,             
        base=dict(
            recon=1.0,
            cap_mse=1.0,
            cap_pin=1.0,
            da=0.2,                 
        ),
        eps=1e-8,                   
    ),
)

# ==============================================================================
# 5. Model Architecture Settings
# ==============================================================================
model = dict(
    # ====================================================
    # 1. Feature extraction
    # ====================================================
    d_tail=[16, 64, 128],      # 1D-CNN channels for the low-voltage tail.
    d_seg=[16, 64, 128],       # 1D-CNN channels for upstream voltage segments.
    
    # ====================================================
    # 2. Latent space and fusion dimensions
    # ====================================================
    hidden_dim=128,            # Shared latent dimension for gating, FiLM and DA alignment.
    d_meta=0,                  # Metadata dimension; zero when no manufacturing features are supplied.
    
    # ====================================================
    # 3. Feature fusion switches
    # ====================================================
    tail_only=False,           # Use only the tail; disable upstream fusion, gating and FiLM when true.
    use_gating=True,           # Enable gating when tail_only=False and use_film=True.
    use_film=True,             # Enable FiLM modulation when tail_only=False.
    use_learned_zero=True,     # Learn an embedding for k=0, when all upstream segments are absent.
    
    # ====================================================
    # 4. Regression outputs
    # ====================================================
    quantiles=[0.1, 0.5, 0.9], # Nominal 80% quantiles; legacy 95% endpoints use normal-approximation expansion.
)

# ==============================================================================
# 6. Missing Simulation Settings (Curriculum)
# ==============================================================================
missing_sim = dict(
    p_drop_all=0.1,                 # Probability to drop all upstream segments
    p_keep_one=0.9,                 # Probability to keep at least one segment
    curriculum_epochs=[0],  
    curriculum_keep=[MAX_SEGS],  
)
