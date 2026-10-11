"""All editable settings for Proposed on one selected self-collected dataset."""
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "NMC_A"  # NMC_A or NMC_B; one model campaign per dataset
# Published processed-data geometry; raw tail points include both endpoints.
DATASET_PRESETS = {
    "NMC_A": dict(SOURCE_BATCH=4, MODEL_BATCH=4, V_MIN=3.0, VSEG_LOW=3.3),
    "NMC_B": dict(SOURCE_BATCH=6, MODEL_BATCH=3, V_MIN=2.75, VSEG_LOW=3.2),
}
if DATASET_NAME not in DATASET_PRESETS:
    raise ValueError("DATASET_NAME must be NMC_A or NMC_B")
SOURCE_BATCH = DATASET_PRESETS[DATASET_NAME]["SOURCE_BATCH"]
MODEL_BATCH = DATASET_PRESETS[DATASET_NAME]["MODEL_BATCH"]
DATA_MAT_PATH = PROJECT_ROOT / "dataset" / f"{DATASET_NAME}_data_processed.mat"
V_MIN = DATASET_PRESETS[DATASET_NAME]["V_MIN"]
VSEG_LOW = DATASET_PRESETS[DATASET_NAME]["VSEG_LOW"]
V_MAX, DV = 4.20, 0.005
VSEG_UPP = VSEG_LOW + 0.30
VOLTAGE_REMOVAL_WINDOW = 0.0
TAIL_LEN_ORIGIN = round((VSEG_LOW - V_MIN) / DV)  # 60 or 90 intervals
RAW_TAIL_POINTS = TAIL_LEN_ORIGIN + 1  # NMC_A: 61; NMC_B: 91
# Network dimensions are independent of the raw grid and match the saved models.
TAIL_LEN, SEG_LEN, MAX_SEGS = 64, 48, 6
POOL_TARGET_RATIO = 0.60
POOL_DIVISOR = 30
INITIAL_LABELLED_COUNT = 30
ACQUISITION_SIZE = 30

# Training, runtime, and output settings. Edit this file before running main.py.
RUN_MODE = "train"  # train, infer, or check
random_seeds = (0, 1, 2)
ROUND_NUMBERS = None  # None: every budget round; otherwise e.g. (1, 3)
OUTPUT_DIR = PROJECT_ROOT / "runs" / "Proposed"
THREADS = 2
num_epochs = 250
patience_epochs = 250
batch_size = 30
normalization_mode = "per_channel"
device = "cuda" if torch.cuda.is_available() else "cpu"
# None uses the bundled checkpoint(s); an explicit best.pt loads a new run.
CHECKPOINT = None

lr = 5e-3
weight_decay = 1e-4
# Plateau reduction plus the configured epoch staircase. Other supported policies:
# cosine_epoch (horizon=num_epochs), none (constant learning rate).
scheduler = dict(
    policy="plateau", factor=0.95, patience=5, min_lr=1e-10,
    epoch_step=10, step_gamma=0.95,
)

# Dynamic multi-loss weighting for reconstruction, capacity, quantile, and DA.
loss_weights = dict(
    mode="dynamic",
    da_mode="Coral",
    enabled=["recon", "cap_mse", "cap_pin", "da"],
    dynamic=dict(
        ema_beta=0.9,
        floor=0.05,
        ceil=10.0,
        normalise=True,
        base=dict(recon=1.0, cap_mse=1.0, cap_pin=1.0, da=0.2),
        eps=1e-8,
    ),
)

# Proposed model architecture and ablation switches.
model = dict(
    d_tail=[16, 64, 128],
    d_seg=[16, 64, 128],
    hidden_dim=128,
    d_meta=0,
    tail_only=False,
    use_gating=True,
    use_film=True,
    use_learned_zero=True,
    quantiles=[0.1, 0.5, 0.9],
)

# Independent prefix sampling for both labelled and unlabelled training inputs.
# Each sample is bounded by its actual complete fragments and the curriculum cap.
missing_sim = dict(
    p_drop_all=0.1,
    p_keep_one=0.9,
    curriculum_epochs=[0],
    curriculum_keep=[MAX_SEGS],
)
