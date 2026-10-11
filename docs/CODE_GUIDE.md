# Code and configuration guide

## Where to start

| Task | Guide | Entry point |
| --- | --- | --- |
| Install and validate the environment | [Installation](../README.md#installation) | Root `run.py check`, `verify` |
| Understand samples, units and MAT fields | [Data dictionary](DATA_DICTIONARY.md) | Dataset README files |
| Train or evaluate Proposed | [Protocol comparison](PROPOSED_FRAMEWORK_COMPARISON.md) | Each dataset's `model_construction/Proposed framework/main.py` |
| Run a baseline | [Baseline guide](BASELINES.md) | Each `Baseline group N/main.py` |
| Rebuild data or dataset figures | [LFP](../LFP%20dataset/README.md), [NMC](../Self-collected%20datasets/README.md) | Local preprocessing/visualization scripts |
| Simulate sorting and regrouping | [Regrouping](../Sorting%20and%20regrouping%20simulation/README.md) | Root `run.py regrouping` |
| Run economic/environmental scenarios | [TEA](../Techno-economic-environmental%20assessment/README.md) | Root `run.py tea` |
| Locate generated tables and figures | [Output layout](FIGURE_OUTPUTS.md) | Module-local `results/` and `visualization/` |
| Check what was actually validated | [Validation summary](../README.md#validation) | Tested environment, outcomes and limitations |
| Check reuse and citation terms | [Availability](../AVAILABILITY.md), [Citation](../CITATION.md) | Code and data have separate terms |

## Model code responsibilities

For LFP and NMC, edit the chosen model directory's `config.py` and run its `main.py` without command-line arguments.

- `config.py`: editable experiment settings, physical data presets and derived dimensions.
- `main.py`: model entry point and experiment orchestration.
- `data.py` or the shared dataset adapter: MAT loading, sample roles, feature construction and normalization.
- `model.py` or `models/`: neural architecture and prediction heads.
- `train_model.py` or `train/`: optimization, validation and estimator fitting.
- Proposed `runtime.py` / `inference.py`: configured execution, output protection and checkpoint-context restoration.
- Dataset `baseline_support/`: internal process isolation and data adapters shared by the four baseline groups. These helpers are not alternative public configuration interfaces.

`RUN_MODE="train"` is the distributed model default. Use `"check"` before a non-training input check; Proposed additionally supports `"infer"`. LFP baseline checks return their report without creating a campaign. NMC baseline checks save configuration/check records in a fresh campaign without fitting weights.

`random_seeds` and `ROUND_NUMBERS` select experiments; `num_epochs` controls neural budgets. Tree counts and depth are in `ML_CONFIG`. `SELECTED_MODELS` selects supported methods within a baseline group. `THREADS` controls CPU threads per worker, not the number of simultaneous experiments. Neural `device` selects CPU or CUDA; group 1 tree estimators run on CPU. In mixed group 4, `device` applies to the neural methods.

New campaigns are created under `OUTPUT_DIR`. Keep each checkpoint with its effective settings, split indices, normalization and dataset identity. New training reads the editable configuration; saved-checkpoint inference restores its own independent context.

## Configuration and saved settings

| Setting or record | Purpose |
| --- | --- |
| Model `config.py` | Settings for new training |
| NMC `DATASET_PRESETS` | Dataset paths and physical windows selected by NMC_A or NMC_B |
| Derived tail/fragment lengths | Input dimensions computed from the selected windows |
| Model dictionaries | Architecture and estimator parameters |
| Method-specific settings | Parameters used by the selected method, such as MMD bandwidths for DA-MMD |
| Checkpoint configuration, splits and normalizers | Context required to evaluate that saved model |

Baseline groups 1–3 use tails of 47, 61 and 91 points for LFP, NMC_A and NMC_B. Group 4 adds 60 upstream positions, giving 107, 121 and 151. Proposed encoders resample to 64 tail points and 48 points per fragment; those dimensions describe a different stage of processing.

Group 4 LightGBM uses 600 trees, depth 10, learning rate 0.03, minimum child samples 20 and random state 42. Group 1 has its own LightGBM configuration; its different values are intentional. Group 3 contains DA-AL, DA-MMD, DA-CORAL and SSL-PL; SSL-PL uses weighted KNN in normalized tail space, without KPCA.

The regrouping module uses its included LFP inputs, model and simulation settings. See its [README](../Sorting%20and%20regrouping%20simulation/README.md) for the five policies, sampling design and pack-capacity definition.

## Data and outputs

Sample maps identify data columns; input manifests provide file paths and hashes. Keep checkpoints with their saved configuration, split indices and normalization. Simulation reference tables and plotting caches are in each module's `results/`; new runs use the output paths described in the [figure guide](FIGURE_OUTPUTS.md).

See [reuse terms](../AVAILABILITY.md), [citation information](../CITATION.md) and the [validation commands](../README.md#validation) for the supplied package.
