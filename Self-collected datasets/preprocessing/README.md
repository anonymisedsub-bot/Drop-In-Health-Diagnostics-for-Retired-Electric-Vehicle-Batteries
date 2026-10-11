# Two-sheet raw data preprocessing

Requires Python 3.10 or newer, NumPy, and SciPy. Verification used Python 3.12,
NumPy 2.1.3, and SciPy 1.15.2. Workbooks are read directly as XLSX XML, so neither
Excel nor openpyxl is needed. See the [installation instructions](../../README.md#installation).

Run from this directory, supplying the dataset directory that contains
`NMC_A_data_processed.mat`, `NMC_B_data_processed.mat`, and `raw/`:

```shell
python preprocess_raw.py --dataset-root ../dataset --output-dir ../runs/nmc_raw_rebuild --workers 4
```

The output directory must not exist. Each workbook contains two sheets,
`DisC` and `DisCC2`. The command builds these **eight arrays** in the same
public-ID/MAT-column order:

| Input | Output arrays |
| --- | --- |
| DisC | `processed_Qd_partial_mat`, `processed_Vd_partial_mat`, `processed_IC_partial_mat` |
| DisCC2 | `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, `processed_RPT_IC_full_mat`, `processed_Qmax_mat` |
| Dataset voltage grid | `voltage_grid_mat` |

It writes `NMC_A_from_raw.mat`, `NMC_B_from_raw.mat`, and
`reconstruction_comparison.json` into the new output directory. Models use the
supplied `../dataset/NMC_A_data_processed.mat` and
`../dataset/NMC_B_data_processed.mat`; preprocessing leaves those inputs in place.

The command verifies workbook hashes and byte sizes against `raw/source_mapping.csv`,
requires both sheets with no extra sheet, and compares each of the eight arrays
with its corresponding supplied MAT array. It exits unsuccessfully for changed
values, NaN masks, shapes, schema, or reference MAT hashes. The report records
`all_available_fields_exact` and `raw_sheets` under schema
`nmc-two-sheet-reconstruction-v2`. `--reference-dir` can point to a separate
reference MAT directory; it defaults to the dataset root. SciPy's generated MAT
container timestamp may differ; all eight numerical arrays must match exactly.

## Processing definitions

The modules in `rebatt_preprocessing/` supply the parsing, interpolation and IC
helpers used by `two_sheet_processing.py`. Raw voltage and capacity are not
smoothed. They are split into monotone runs and mapped to the grid with PCHIP,
preserving local support, low-end extrapolation and at most one upper ceiling
row. Capacity is remaining capacity on an ascending voltage grid, zero at the
cutoff. IC is calculated by differencing Q and applying one MATLAB-compatible
SG20 smoothing pass within finite segments.

NMC_A uses 3.00–4.20 V and NMC_B uses 2.75–4.20 V, both at 5 mV spacing.
Qmax is the DisCC2 cumulative capacity at 3.00 V for NMC_A and the maximum
DisCC2 cumulative capacity for NMC_B. Capacity is in Ah, voltage in V and IC
in Ah/V. See the [data dictionary](../../docs/DATA_DICTIONARY.md).

The mapping contains 353 NMC_A and 334 NMC_B cells. Public ID equals one-based
MAT column, with numbering restarting at 1 in each dataset. The workflow needs
the included two-sheet workbooks, `raw/source_mapping.csv`, and the two supplied
processed MATs. See the [raw-data guide](../dataset/raw/README.md).

## Reconstruction checks

The preprocessing command checks workbook identities, worksheet structure and
sample order, then compares all eight reconstructed arrays against the supplied
processed MATs, including missing-value masks. It writes the comparison report
to the selected output directory. The complete rebuild covers all 687 cells.
See the [package validation commands](../../README.md#validation).
