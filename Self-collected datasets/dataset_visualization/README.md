# Figure 3: NMC_A and NMC_B dataset panels

This directory contains the statistical panels for the two self-collected NMC
datasets in Figure 3. The plots read the same packaged MAT files used for model
training and do not require model weights.

## Panel mapping

| Figure 3 | Content | Output PNG |
|---|---|---|
| d | NMC_A SOH distribution and within-bin standard deviation, 353 cells | `panel_d_NMC_A.png` |
| e | NMC_B SOH distribution and within-bin standard deviation, 334 cells | `panel_e_NMC_B.png` |
| g | NMC_A full RPT V–Q and IC curves with a shared capacity color bar | `panel_g_NMC_A.png` |
| h | NMC_B full RPT V–Q and IC curves with a shared capacity color bar | `panel_h_NMC_B.png` |
| j | Joint violin plot of three simulated LFP initial-voltage groups and measured NMC_A/NMC_B initial voltages | `panel_j_initial_voltage.png` |
| NMC_A portion of k | NMC_A partial-discharge curves | `panel_k_NMC_A.png` |
| NMC_B portion of k | NMC_B partial-discharge curves | `panel_k_NMC_B.png` |
| NMC_A portion of l | NMC_A cell counts for each available upstream fragment | `panel_l_NMC_A.png` |
| NMC_B portion of l | NMC_B cell counts for each available upstream fragment | `panel_l_NMC_B.png` |

There are **nine separate panels**, each saved as a 600 dpi PNG, and a
`Figure_3_NMC_overview.png` showing their mapping. The overview is not the final
layout of the complete Figure 3. Individual panels use their configured sizes
and omit embedded panel letters; filenames and the overview identify panels and
datasets. The experimental workflow schematic in Figure 3b and the other
LFP-only panels are outside the scope of these statistical plots. Public names
`NMC_A` and `NMC_B` correspond to NMC-A and NMC-B in the manuscript.

```text
dataset_visualization/
  plot.py                       Plotting entry point
  figure3_panels.py              d/e/g/h and the NMC portions of k/l
  figure3_initial_voltage.py     Cross-dataset comparison in j
  outputs/                      Nine panels and their overview
  results/
    panel_*/                    Panel CSV/NPZ files and metadata.json
    NMC_A_cell_identity.csv      Identity and data mapping for all 353 cells
    NMC_B_cell_identity.csv      Identity and data mapping for all 334 cells
    panel_manifest.json         Panel, code, data, and image hashes
  README.md
  LICENSE
```

## Run

From `Self-collected datasets`, use a Python environment prepared with the
[installation instructions](../../README.md#installation):

```console
python dataset_visualization/plot.py
python dataset_visualization/plot.py --dataset NMC_A --only soh rpt partial availability
python dataset_visualization/plot.py --only initial-voltage
python dataset_visualization/plot.py --only soh --formats png,pdf,svg
```

The first command generates all Figure 3 panels. Images default to `outputs/`
and numerical results to `results/` in this directory. `--output <directory>`
sets the image destination and defaults the numerical destination to its
`results/` subdirectory; `--results <directory>` sets a separate numerical
destination. `--only` accepts `soh`, `rpt`, `partial`, `availability`, and
`initial-voltage`. `--dataset` selects the cohort for the individual NMC panels;
panel j always includes both NMC datasets and all three LFP groups. PNG is the
default format. Add vector versions with `--formats png,pdf,svg`; SVG retains
text, and PDF embeds TrueType fonts.

The scripts can also be called by their full paths from another working
directory. Only panel j requires the neighboring packaged
`LFP dataset/dataset/LFP_data_processed.mat`; the other panels use this module's
`../dataset` files.

## Data and statistical definitions

- **Same training cohorts:** NMC_A uses all 353 columns of
  `../dataset/NMC_A_data_processed.mat`, and NMC_B uses all 334 columns of
  `../dataset/NMC_B_data_processed.mat`. Loading checks the expected SHA-256 hashes
  of the supplied datasets. Plots use each complete cohort, not a model test
  subset. `mat_column_1based` equals the public raw Excel ID. The complete mapping
  comes from `../dataset/raw/source_mapping.csv` and preserves the MAT column order.
- **d/e:** SOH = `processed_Qmax_mat / 2.5 Ah × 100%`. NMC_A uses
  Freedman–Diaconis bins; NMC_B uses Scott bins. The secondary axis shows the
  within-bin sample standard deviation of SOH (ddof=1, in percentage points),
  not prediction error or a confidence interval. Bins include every sample,
  including cells above nominal capacity.
- **g/h:** Use `processed_RPT_Q_full_mat`, `processed_RPT_V_full_mat`, and
  `processed_RPT_IC_full_mat`. IC voltages are the midpoints of adjacent RPT
  voltage points. These are the full-discharge RPT data at approximately
  0.5C. Both subplots use the same cell maximum capacity for coloring, with no
  additional smoothing.
- **NMC portions of k:** Use `processed_Qd_partial_mat` and
  `processed_Vd_partial_mat`, measured during partial discharge at approximately
  1C. NaN marks unobserved regions; missing curves are not imputed.
- **j:** NMC initial voltage is the highest valid voltage in each column of
  `processed_Vd_partial_mat`. It is the start of the processed measured curve on
  the 0.005 V grid, not a simulated voltage. Each of the three LFP groups reuses
  3,511 samples with scalar rejection sampling: mean 2.85 V,
  standard deviation 0.10 V, seed reset to 999 for each group, and acceptance
  only above the corresponding 2.73/2.75/2.77 V threshold, without an upper bound.
  The internal violin lines show Q1, the median, and Q3. The LFP groups are not
  three independently measured cohorts.
- **NMC portions of l:** Use cumulative fragment availability.
  `None` means no complete first fragment; bar k counts
  cells that fully cover fragment k. A cell can contribute to multiple bars, so
  the sum of bar heights does not equal the number of cells. Complete-fragment
  endpoint comparisons use a 1e-9 V tolerance, with no additional removal of an
  initial-voltage interval.

| Dataset | Tail | Six upstream fragments | Bar heights for None, 1, 2, 3, 4, 5, 6 |
|---|---|---|---|
| NMC_A | 3.00–3.30 V | 3.30–3.60 V, 0.05 V each | 6, 347, 340, 334, 325, 311, 285 |
| NMC_B | 2.75–3.20 V | 3.20–3.50 V, 0.05 V each | 3, 331, 323, 287, 206, 153, 100 |

## Model checks and file inventory

Select model checks or inference through `RUN_MODE` in
`../model_construction/Proposed framework/config.py`, then run `main.py` in that directory. Model
outputs are saved separately from these Figure 3 dataset statistics. See the
[package validation checks](../../README.md#validation).

`results/panel_manifest.json` records the plot exports, and
`../../release_manifest.json` provides the package file inventory. The software
retains its MIT licence; see [the data terms](../dataset/DATA_LICENSE.md) for
academic research use and attribution.
