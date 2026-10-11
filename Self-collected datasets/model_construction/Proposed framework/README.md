# Proposed: gated fusion and FiLM

Edit `config.py`, then run `python main.py` from this directory. The script accepts no command-line arguments. From elsewhere, pass the path to this same `main.py`.

| Setting | Meaning |
| --- | --- |
| `RUN_MODE` | `train` (default), `infer`, or `check` |
| `num_epochs`, `patience_epochs` | Joint-training epoch budget and early-stopping patience; both default to 250 |
| `random_seeds` | Repeated seeds, default `(0, 1, 2)` |
| `ROUND_NUMBERS` | `None` for all nested budgets, or a tuple such as `(1, 3)` |
| `DATA_MAT_PATH` | Processed data file |
| `device`, `THREADS` | Runtime device and PyTorch CPU thread count |
| `OUTPUT_DIR` | Parent for fresh timestamped campaigns, default dataset `runs/Proposed` |
| `CHECKPOINT` | Explicit `best.pt` for inference, or `None` for bundled weights |
| `lr`, `weight_decay`, `scheduler` | AdamW and learning-rate schedule; policies: `plateau`, `cosine_epoch`, `none` |
| `loss_weights` | Loss mode, enabled terms, base weights and EMA settings |
| `model`, `missing_sim` | Architecture/ablation switches and fragment sampling curriculum |

A short execution uses `num_epochs = 1`, `random_seeds = (0,)`, `ROUND_NUMBERS = (1,)` and `device = "cpu"`. These are ordinary explicit settings; all remaining configuration stays in effect. There is one joint-training stage, with no pretraining parameter or hidden dataset reduction.

Training saves the effective configuration and complete checkpoint input context. Best model tensors are cloned independently. Disabled loss terms retain zero weight after EMA updates. Inference reads checkpoint settings into an immutable local namespace and leaves the training configuration unchanged. Every training or inference invocation creates a fresh output campaign.

`DATASET_NAME` selects `NMC_A` or `NMC_B` and the corresponding supplied `../../dataset/NMC_A_data_processed.mat` or `../../dataset/NMC_B_data_processed.mat`. Both presets are visible in config: minimum/tail voltage 3.00/3.30 V and 2.75/3.20 V; raw tails 61 and 91 points, zero voltage removal. Both networks use 64 tail points and 48 points per fragment. Training uses the current configuration.

With no explicit `CHECKPOINT`, inference automatically selects bundled checkpoints for the configured dataset, seeds and rounds. Keep each checkpoint with its supplied configuration, manifests and matching processed MAT. With an explicit weight path, saved manifests identify its dataset, seed and round, and its MAT hash is checked. Validation compares capacity predictions using each cell's observed fragments. Bundled-checkpoint inference must agree with the saved table within 0.0002 Ah.

See [the dataset README](../../README.md) for scientific protocol, input identity and saved-artifact details.
