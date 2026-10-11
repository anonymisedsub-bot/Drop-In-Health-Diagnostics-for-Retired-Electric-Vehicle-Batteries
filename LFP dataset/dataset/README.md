# LFP dataset files

The CETL-HFUT cohort contains 4,000 raw records in `CETL-data.mat` and 3,511 retained records in `LFP_data_processed.mat`. These files describe the same cohort. Raw voltage/capacity are in mV/mAh; processed voltage/capacity are in V/Ah.

The processed MAT contains `processed_Vd_mat`, `processed_Qd_mat`, `processed_vIC_mat` and `processed_qIC_mat`. The voltage/capacity matrices have 201 rows and 3,511 columns; each column identifies one retained cell. The capacity target is the final row of `processed_Qd_mat`, in Ah. Descriptive SOH uses a nominal capacity of 15.5 Ah.

`cell_index_mapping.csv` links zero-based processed columns to one-based raw MATLAB records in processed-column order. Preprocessing removes 487 records in its first filtering stage and two in its second stage.

Run `data_processing_code_V1.m` in MATLAB to execute preprocessing. It resolves inputs and output relative to its own directory and writes `LFP_data_processed.mat`. See the root [installation guide](../../README.md#installation) and [validation summary](../../README.md#validation).

See the [dataset overview](../README.md) for the model protocol and the root [data-availability document](../../AVAILABILITY.md) for data-use conditions. The supplied `LICENSE` covers code; it does not override the dataset's academic-use terms.
