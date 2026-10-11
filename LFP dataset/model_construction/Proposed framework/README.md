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

A short execution uses `num_epochs = 1`, `random_seeds = (0,)`, `ROUND_NUMBERS = (1,)` and `device = "cpu"`. All remaining configuration stays in effect, including the full dataset and split sizes. Training uses one joint-training stage.

Training saves the effective configuration and complete checkpoint input context. Best model tensors are cloned independently. Disabled loss terms retain zero weight after EMA updates. Inference reads checkpoint settings into an immutable local namespace and leaves the training configuration unchanged. Every training or inference invocation creates a fresh output campaign.

LFP uses eight label budgets from 100 to 2,200. Validation sweeps fragment counts 0 through 6. `val_tail_weight` controls the zero-fragment case. For inference, `INFERENCE_INDICES` optionally selects zero-based MAT columns and `INFERENCE_K` optionally selects a fragment prefix. Without an explicit checkpoint, inference uses the bundled one-epoch demo weight as an execution example. This checkpoint does not establish paper accuracy.

See [the dataset README](../../README.md) for the scientific protocol, input identity and saved files, and the root [installation guide](../../../README.md#installation) for dependencies.
