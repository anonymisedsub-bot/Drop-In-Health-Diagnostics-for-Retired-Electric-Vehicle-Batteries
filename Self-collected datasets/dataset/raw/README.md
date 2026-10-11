# Two-sheet raw measurements

The supplied raw data contain exactly **DisC** and **DisCC2** for each of the
353 NMC_A and 334 NMC_B cells in the processed model datasets.

| Dataset | Public filenames | Cells | Workbook bytes |
| --- | --- | ---: | ---: |
| NMC_A | `NMC_A/1.xlsx`–`353.xlsx` | 353 | 51,882,224 |
| NMC_B | `NMC_B/1.xlsx`–`334.xlsx` | 334 | 53,218,987 |

Combined size: **105,101,211 bytes** in 687 two-sheet workbooks.

Within each dataset, **public ID = filename stem = one-based MAT column**.
IDs restart at 1 in the other dataset; use `(dataset, new_id)` as the cell key.
The supplied models read `../NMC_A_data_processed.mat` and
`../NMC_B_data_processed.mat`.

`source_mapping.csv` provides the public ID and MAT column order. Entries
describing the supplied files include:

- `raw_relative_path`, `sha256`, `byte_size`: the two-sheet XLSX path and integrity checks.
- `retained_sheets`, `DisC_xml_sha256`, `DisCC2_xml_sha256`: the sheet names and worksheet hashes.
- Processed MAT filename and SHA-256: the corresponding processed dataset.

The [preprocessing command](../../preprocessing/README.md) writes
`NMC_A_from_raw.mat`, `NMC_B_from_raw.mat` and `reconstruction_comparison.json`
to a new directory. It builds these eight arrays in public-ID order:

| Input | Output arrays |
| --- | --- |
| DisC | `processed_Qd_partial_mat`, `processed_Vd_partial_mat`, `processed_IC_partial_mat` |
| DisCC2 | `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, `processed_RPT_IC_full_mat`, `processed_Qmax_mat` |
| Dataset voltage grid | `voltage_grid_mat` |

The command verifies raw workbook and processed-MAT hashes before comparing the
eight output arrays with the supplied datasets, including shapes, numerical values
and NaN masks. Models continue to use the supplied processed MAT files.
`extraction_verification.json` and `reconstruction_comparison.json` contain
the included workbook and array verification results.

These measurements are for academic research only; commercial use is not
permitted. See [DATA_LICENSE.md](../DATA_LICENSE.md).
