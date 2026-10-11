# LFP Baseline group 4

Methods: MLP, CNN, CGB and LGBM with padded inputs.

`device` selects the neural models' runtime device. The supplied CGB and LGBM presets run on CPU, with estimator settings in `ML_CONFIG`. Neural early stopping monitors q50 validation MSE; `es_enable`, `es_mode`, `es_patience`, and `es_min_delta` configure the stopping rule, while the metric is fixed.

MLP/CNN constructors require their complete architecture dictionary from `config.py`. Builders validate `seq_len` against the derived `segment_length`, quantiles against `QUANTILES`, and one input channel/output per quantile. Inconsistent settings raise an error during model construction.

Edit the adjacent `config.py`, then run `python main.py` from this directory (or invoke this file by its full path). The default `RUN_MODE = 'train'` trains the configured methods. Set `RUN_MODE = 'check'` to validate the data, dependencies, windows and nested labelling budgets without training. This entry accepts no command-line settings.

- `SELECTED_MODELS`: a nonempty tuple containing this group's canonical model names.
- `random_seeds`: the seed tuple, initially `(0, 1, 2)`.
- `ROUND_NUMBERS`: `None` for every nested budget, or a tuple of one-based round numbers such as `(1,)`.
- `OUTPUT_DIR`: parent of fresh timestamped campaigns, initially `LFP dataset/runs/baseline_group_4`. Existing campaigns are never overwritten; bundled data and model files are protected.
- `THREADS`, `device`, `DATA_MAT_PATH`, split ratios, labelling sizes and voltage-window settings control execution and inputs. Edit the existing assignments in `config.py`; tree thread counts derive from `THREADS` there.
- Neural models use `num_epochs`, `batch_size`, learning-rate/scheduler, early-stopping and architecture settings. Tree methods use `ML_CONFIG` for their estimator settings.

For a short verification run, select one model and seed, set `ROUND_NUMBERS = (1,)`, and reduce its `num_epochs` or tree count in `config.py`. All other settings remain in effect. This checks software execution; it does not establish paper accuracy.

A campaign contains `run_manifest.json`, `config_snapshot.py` and `<model>/seed_<seed>/round_<number>/` with checkpoints, predictions, splits, fitted normalizers and metadata. Each model/seed pair uses a fresh worker process. Metadata records effective settings and SHA-256 hashes of configuration, dataset and checkpoint. Normalizers are fitted on labelled training cells. Runs use fixed splits and nested acquisitions, with each model's configured masking, loss and checkpoint rules.

The LGBM preset uses 600 trees, maximum depth 10, learning rate 0.03, minimum child samples 20 and random state 42.

See the dataset [README](../../README.md) for data preparation and the root [installation guide](../../../README.md#installation) for dependencies.
