# Self-collected NMC datasets

These raw and processed NMC data are for academic research only, with no
commercial use permitted. See [DATA_LICENSE.md](DATA_LICENSE.md).

| Dataset | Supplied model input | Cells | Raw workbooks |
| --- | --- | ---: | --- |
| NMC_A | `NMC_A_data_processed.mat` | 353 | `raw/NMC_A/1.xlsx`–`353.xlsx` |
| NMC_B | `NMC_B_data_processed.mat` | 334 | `raw/NMC_B/1.xlsx`–`334.xlsx` |

Every workbook contains two sheets, **DisC** and **DisCC2**. Public IDs restart
at 1 in each dataset. The filename number equals the one-based column in the
corresponding processed MAT; use `(dataset, new_id)` to identify a cell.
`raw/source_mapping.csv` records the public IDs, MAT column order, raw-file
paths, byte sizes, worksheet hashes and processed-MAT hashes used for verification.

Models use the two supplied processed MAT files listed above. The preprocessing
command builds eight arrays from the two-sheet workbooks into separately named
files, `NMC_A_from_raw.mat` and `NMC_B_from_raw.mat`:

| Input | Output arrays |
| --- | --- |
| DisC | `processed_Qd_partial_mat`, `processed_Vd_partial_mat`, `processed_IC_partial_mat` |
| DisCC2 | `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, `processed_RPT_IC_full_mat`, `processed_Qmax_mat` |
| Dataset voltage grid | `voltage_grid_mat` |

Run from the release root after following the [installation instructions](../../README.md#installation):

```console
python "Self-collected datasets/preprocessing/preprocess_raw.py" --dataset-root "Self-collected datasets/dataset" --output-dir runs/nmc_raw_rebuild --workers 4
```

The output directory must not exist. The command verifies workbook hashes and
compares all eight arrays with the corresponding supplied MAT arrays, including
shapes, numerical values and NaN masks. It writes `reconstruction_comparison.json`
alongside the two output MATs and fails if a comparison differs. The supplied model
inputs remain in `dataset/`. MAT container timestamps can differ while numerical
arrays match exactly.

The included `raw/extraction_verification.json` and
`raw/reconstruction_comparison.json` record workbook and array checks. See the
[preprocessing guide](../preprocessing/README.md) for algorithm details and
[data dictionary](../../docs/DATA_DICTIONARY.md) for array definitions and units.
Data availability and terms are described in [AVAILABILITY.md](../../AVAILABILITY.md).
