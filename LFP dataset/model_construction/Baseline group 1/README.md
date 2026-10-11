# LFP Baseline group 1

Methods: XGB, RF, CGB and LGBM.

Edit the adjacent `config.py`, then run `python main.py` from this directory (or invoke this file by its full path). The default `RUN_MODE = 'train'` trains the configured methods. Set `RUN_MODE = 'check'` to validate the data, dependencies, windows and nested labelling budgets without training. This entry accepts no command-line settings.

- `SELECTED_MODELS`: a nonempty tuple containing this group's canonical model names.
- `random_seeds`: the seed tuple, initially `(0, 1, 2)`.
- `ROUND_NUMBERS`: `None` for every nested budget, or a tuple of one-based round numbers such as `(1,)`.
- `OUTPUT_DIR`: parent of fresh timestamped campaigns, initially `LFP dataset/runs/baseline_group_1`. Existing campaigns are never overwritten; bundled data and model files are protected.
- `THREADS`, `DATA_MAT_PATH`, split ratios, labelling sizes and `TAIL_LEN` control execution and inputs. The supplied tree presets run on CPU; their thread counts derive from `THREADS`. Group 1 has no neural `device` setting.
- `ML_CONFIG` controls estimator settings. Group 1 contains no neural-training settings.

For a short verification run, select one model and seed, set `ROUND_NUMBERS = (1,)`, and reduce its tree count in `config.py`. All other settings remain in effect. This checks software execution; it does not establish paper accuracy.

A campaign contains `run_manifest.json`, `config_snapshot.py` and `<model>/seed_<seed>/round_<number>/` with checkpoints, predictions, splits, fitted normalizers and metadata. Each model/seed pair uses a fresh worker process. Metadata records effective settings and SHA-256 hashes of configuration, dataset and checkpoint. Normalizers are fitted on labelled training cells. Runs use fixed splits and nested acquisitions, with each model's configured training and checkpoint rules.

See the dataset [README](../../README.md) for data preparation and the root [installation guide](../../../README.md#installation) for dependencies.
