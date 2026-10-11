# NMC Baseline group 4

Edit this folder's `config.py`, then run its `main.py` from any working directory:

```powershell
python "Self-collected datasets/model_construction/Baseline group 4/main.py"
```

The command above is relative to the release root. From this folder, use `python main.py`.
There are no command-line tuning options. Every model/data/training choice is in `config.py`.

- `RUN_MODE = "train"` runs training; use `"check"` to validate the MAT, split and model inputs without fitting.
- `SELECTED_MODELS` selects a nonempty tuple from **MLP, CNN, CGB, LGBM**. The default selects all four.
- `random_seeds` defaults to `(0, 1, 2)`; `ROUND_NUMBERS = None` selects all available acquisition rounds. Use `(1,)` for the first round.
- `THREADS = 2` sets CPU threads per worker; model/seed workers run sequentially in separate processes. Estimator thread settings derive from it when the configuration is imported.
- `OUTPUT_DIR` defaults to `Self-collected datasets/runs/baseline_group_4`. Each invocation creates a fresh timestamped campaign beneath it, with independent model/seed/round checkpoints and a copy of the configuration source.

## Dataset selection

Set `DATASET_NAME` to `"NMC_A"` or `"NMC_B"`. Both presets are written out in this configuration file and select the supplied `../../dataset/NMC_A_data_processed.mat` or `../../dataset/NMC_B_data_processed.mat`.

| Preset | V_MIN | Tail endpoint | Tail points |
| --- | ---: | ---: | ---: |
| NMC_A | 3.00 V | 3.30 V | 61 |
| NMC_B | 2.75 V | 3.20 V | 91 |

Both use `DV = 0.005` V and `VOLTAGE_REMOVAL_WINDOW = 0.0` V. Input dimensions are calculated when `config.py` is imported. The runner validates physical grid and input shapes and never substitutes another preset. `DATA_MAT_PATH` may point to a schema-compatible MAT with the selected dataset's physical grid.

## Training and records

Neural training uses `num_epochs`, `batch_size`, `lr`, `scheduler`, early-stopping settings and the model dictionaries exactly as configured. For a short verification, select one model/seed/round and set `num_epochs = 1`; keep the complete data split.

Inputs contain the tail plus 60 upstream points (121 points for A, 151 for B). Missing positions are zero-padded after fitting normalization on observed labelled values. Curriculum and random masks never reveal unavailable observations. The LightGBM default is 600 estimators, maximum depth 10, learning rate 0.03, minimum child samples 20 and random state 42.

The split uses seeded random acquisition in 30-cell increments. Normalization is fitted on labelled samples only; unlabelled capacities remain hidden. The q50 output uses MSE, predictions remain unclipped, and residual standard deviation uses `ddof=0`. Each case records the MAT hash, split membership, input protocol, configuration and normalization. Save outputs beneath the configured runs directory or to an external destination.
