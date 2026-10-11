# LFP / NMC Proposed models and training protocols

The three datasets use the same core network and losses, with different data protocols, missing-fragment handling and validation procedures. This describes the current public configuration, not equality of trained results.

## Shared architecture and default hyperparameters

Both frameworks use TailEncoder, SegmentEncoder, SegmentSetAggregator, TailDecoder and ConditionalDecoder. Native mean aggregation groups fragments by sample. Each saved checkpoint is evaluated with its own normalization and configuration.

| Setting | LFP, NMC_A and NMC_B |
| --- | --- |
| Input | Two channels: Q and dQ/dV; 64 resampled tail points; 48 points per upstream fragment; at most six fragments |
| Encoders | Tail/segment channels [16, 64, 128]; hidden dimension 128 |
| Fusion | Gated fusion + FiLM; learned zero-fragment embedding enabled; tail_only=False |
| Capacity output | Quantiles [0.1, 0.5, 0.9]; middle output for capacity-error evaluation; no additional metadata input |
| Losses | Tail reconstruction, capacity MSE, quantile pinball and CORAL; the two losses.py files are identical |
| Dynamic weighting | EWMA balancing; beta 0.9; base reconstruction/MSE/pinball/DA weights 1/1/1/0.2 |
| Optimizer | AdamW; learning rate 0.005; weight decay 0.0001 |
| Scheduler | ReduceLROnPlateau: factor 0.95, patience 5; additional factor 0.95 every 10 epochs; minimum learning rate 1e-10 |
| Training | From scratch at each budget; up to 250 joint-training epochs; no pretraining; early-stopping patience 250 |
| Normalization | Fit on currently labelled training samples only; per-channel input min-max and capacity min-max |
| Seeds | 0, 1 and 2, selected by random_seeds |

These are the distributed training defaults. For a short check, explicitly select one seed, one budget and a small num_epochs in config.py. All remaining configuration settings stay in effect.

## Dataset-specific protocols

| Setting | LFP | NMC_A | NMC_B |
| --- | --- | --- | --- |
| Input MAT | LFP_data_processed.mat | NMC_A_data_processed.mat | NMC_B_data_processed.mat |
| Sample columns | 3511 | 353 | 334 |
| Data grid | 2.50–3.50 V, 201 points | 3.00–4.20 V, 241 points | 2.75–4.20 V, 291 points |
| Tail boundaries | 2.50–2.73 V | 3.00–3.30 V | 2.75–3.20 V |
| Actual upstream grid | 2.735–3.030 V | 3.305–3.600 V | 3.205–3.500 V |
| Training batch size | 100 | 30 | 30 |
| Initial labelled / acquisition pool | 100 / 2100 | 30 / 180 | 30 / 180 |
| Validation / test | 702 / 609 | 71 / 72 | 62 / 62 |
| Label budgets | 100, 400, 700, 1000, 1300, 1600, 1900, 2200 | 30, 60, 90, 120, 150, 180, 210 | Same as NMC_A |
| Extra voltage-removal window | NMC-specific setting not used | 0 V | 0 V |
| Validation | Average capacity MSE across k=0…6; k=0 weight 1 | All complete fragments actually available for each validation sample | Same as NMC_A |

NMC checks partial-tail completeness before splitting. All 353 NMC_A and 334 NMC_B samples pass under the distributed configuration. Recorded splits for seeds 0/1/2 are disjoint and cover available samples.

1. **Feature construction.** LFP resamples Q/V and computes dQ/dV with np.gradient; the target is the final processed Q value. NMC resamples partial Q and IC separately, aligning IC to the right endpoints of voltage intervals; the target is Qmax. NMC tail features begin 0.005 V above the lower boundary.
2. **Missing fragments.** LFP simulates initial voltage for the acquisition pool and test samples; labelled and validation roles retain the complete upstream range. All four NMC roles use fragments observed in partial-discharge measurements.
3. **Training-time random k.** LFP resamples k for the labelled stream only; its unlabelled stream uses all fragments after simulated truncation. NMC independently resamples k for both streams. Both use p_drop_all=0.1 and p_keep_one=0.9, applied to different streams.
4. **Unlabelled targets.** The LFP container carries targets, but its unlabelled training branch discards them. NMC returns None for unlabelled targets and checks they are not exposed to capacity supervision. Both fit normalizers only on labelled samples.
5. **Model selection.** Both save the best validation-MSE epoch. LFP aggregates across multiple k values; NMC uses observed fragments. This also affects scheduling.
6. **Splitting.** LFP rounds 60% of the entire cohort down to a multiple of 300 for its pool. After reserving the initial 30 labels, NMC selects the multiple of 30 nearest 60% of the cohort for its pool and splits the remainder into validation/test. Random generators and draw order differ.

## Configuration and direct execution

Run each model_construction/Proposed framework/main.py; the adjacent config.py controls it. RUN_MODE defaults to "train"; use "check" without training or "infer" for saved checkpoints.

NMC DATASET_NAME selects NMC_A or NMC_B. Physical presets, paths, windows and derived dimensions are computed in that configuration. Loaders validate data/configuration consistency.

random_seeds selects the seed tuple; ROUND_NUMBERS selects one-based budget rounds, with None meaning all rounds; num_epochs limits epochs per budget. OUTPUT_DIR, device and THREADS are configured locally. There is one joint-training stage.

From the release root:

```console
python "LFP dataset/model_construction/Proposed framework/main.py"
python "Self-collected datasets/model_construction/Proposed framework/main.py"
```

Running the same script in an IDE has the same meaning. Default paths resolve relative to the package. Follow the local README for custom paths. LFP and NMC loaders implement different contracts; changing a MAT filename does not make them interchangeable.

## Saved-checkpoint inference

In RUN_MODE="infer", architecture, normalization and sample identity must match the checkpoint. Saved settings are read into an independent configuration object without modifying new-training configuration. See each model README for CHECKPOINT, seed/budget selection and outputs. An explicitly selected NMC checkpoint is identified from its associated dataset/split metadata.

The bundled LFP demo/best.pt is a one-epoch execution demonstration, not a final-paper checkpoint. NMC defaults to its bundled saved checkpoints. Keep each new checkpoint's split, configuration and normalization together; replacing best.pt alone is insufficient.

## Short execution checks

Edit the existing settings in config.py:

```python
RUN_MODE = "train"
random_seeds = (0,)
ROUND_NUMBERS = (1,)
num_epochs = 1
device = "cpu"
THREADS = 2
```

Choose a separate OUTPUT_DIR, run the adjacent main.py and retain the edited configuration. This uses the configured data-role protocol and cohort. Restore the required epoch budget, seeds and rounds before a full campaign. Short-check metrics are not paper performance results.

Run the supplied [validation commands](../README.md#validation) to check model behavior and file integrity. Record the actual configuration and outputs for each training campaign.
