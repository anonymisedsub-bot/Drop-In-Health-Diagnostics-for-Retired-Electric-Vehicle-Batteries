# Data dictionary and sample identity

Array columns are samples and rows are voltage-grid positions unless specified otherwise. Processed voltage is in V, capacity in Ah and incremental capacity in Ah/V.

| Artifact | Included content |
| --- | --- |
| LFP raw (`CETL-data.mat`) | `unbalanced_data`, a 1x1 MATLAB struct holding 4000 records; raw discharge Vd/Qd use mV/mAh. |
| LFP processed (`LFP_data_processed.mat`) | `processed_Qd_mat`, `processed_Vd_mat`: 201x3511; `processed_qIC_mat`, `processed_vIC_mat`: 200x3511. |
| NMC_A (`NMC_A_data_processed.mat`) | 353 columns; capacity/voltage matrices have 241 rows and IC matrices 240 rows; voltage grid 3.00–4.20 V. |
| NMC_B (`NMC_B_data_processed.mat`) | 334 columns; capacity/voltage matrices have 291 rows and IC matrices 290 rows; voltage grid 2.75–4.20 V. |

## NMC processed fields

Both NMC MAT files contain `processed_Qd_partial_mat`, `processed_Vd_partial_mat`, `processed_IC_partial_mat`, `processed_IC_full_mat`, `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, `processed_RPT_IC_full_mat`, `processed_Qmax_mat` and `voltage_grid_mat`.

`processed_Qmax_mat` is the 1xN capacity target. `voltage_grid_mat` is 241x1 for NMC_A and 291x1 for NMC_B. Partial matrices contain non-finite positions where observations are unavailable; loaders identify complete observed fragments. Preserve these missing-value masks. Capacity labels reserved for validation/test evaluate predictions and are excluded from training inputs.

NMC_A capacity labels range from approximately 0.246665 to 2.498397 Ah; NMC_B from 1.980402 to 2.570060 Ah. NMC descriptive SOH uses nominal capacity 2.5 Ah and LFP descriptive SOH uses 15.5 Ah. Values above 100% are possible. LFP model targets use the final processed-Q row.

## Sample indexing

LFP `dataset/cell_index_mapping.csv` links zero-based processed columns to one-based raw records and applies to the supplied processed MAT. NMC `dataset/raw/source_mapping.csv` defines the workbook paths, hashes and public sample order used by preprocessing.

For NMC, `new_id` equals `mat_column_1based`: 1–353 for NMC_A and 1–334 for NMC_B. Exported zero-based indices are explicitly labelled. Keep the dataset name with each sample ID. Regrouping has its own input MAT and model-associated sample identities, specified in its `inputs/` and saved model records.

## NMC workbook preprocessing

Each numbered workbook contains `DisC` and `DisCC2`. The public preprocessing command reconstructs these eight arrays:

| Input | Reconstructed arrays |
| --- | --- |
| `DisC` | `processed_Qd_partial_mat`, `processed_Vd_partial_mat`, `processed_IC_partial_mat` |
| `DisCC2` | `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, `processed_RPT_IC_full_mat`, `processed_Qmax_mat` |
| Dataset voltage grid | `voltage_grid_mat` |

The command verifies workbook identities and compares all eight reconstructed arrays with the corresponding supplied MAT values. Outputs are named `NMC_A_from_raw.mat` and `NMC_B_from_raw.mat` and are written to a new directory. Training uses the supplied `NMC_A_data_processed.mat` or `NMC_B_data_processed.mat` selected in `config.py`. See the [preprocessing guide](../Self-collected%20datasets/preprocessing/README.md) for commands.
