# LFP dataset visualization

This directory contains **eight visualization categories and nine Python notebooks**. The notebooks read `../dataset/LFP_data_processed.mat` and `../dataset/CETL-data.mat` directly. The figures describe the dataset, tail, and upstream fragments; they do not invoke trained models.

## Contents and default figures

| Notebook | Command category | Content | PNG count |
|---|---|---|---:|
| `Capacity distribution/plot.ipynb` | `capacity` | Capacity histogram with within-bin standard deviation; SOH distribution | 2 |
| `Raw charge-discharge curves/plot.ipynb` | `raw-curves` | Complete raw charge/discharge curves | 1 |
| `Complete and segmented discharge curves/plot.ipynb` | `complete-curves` | Complete V–Q and IC curves | 1 |
| `Complete and segmented discharge curves/combined_2_plus_2_plot.ipynb` | `segmented-curves` | Complete/truncated curves, simulated initial voltage, and cell counts by fragment availability | 1 |
| `Tail+segment plot/plot.ipynb` | `tail-segments` | Tail, six upstream fragments, and a separate color bar | 8 |
| `Tail+upstream fragments/plot.ipynb` | `tail-soc` | Tail/SOC distributions and six upstream-fragment composites | 6 |
| `Input tsne distribution/plot.ipynb` | `tsne` | t-SNE embeddings of the tail and upstream fragments | 3 |
| `Correlation analysis/plot.ipynb` | `correlation` | Pearson correlation between capacity at each voltage sample and maximum capacity | 3 |
| `Voltage and SOC ranges of different tail settings/plot.ipynb` | `tail-ranges` | Tail voltage-span fractions and SOC fractions at cutoff voltages | 3 |

The default output is **28 PNG figures**. Plots that sweep tail upper boundaries use 2.73, 2.75, and 2.77 V. The standalone tail and four-panel curve plots use 2.73 V.

```text
dataset_visualization/
  <category folders>/*.ipynb           plotting source
  notebook_setup.py                   shared data discovery and export
  plot.py                             batch entry point
  outputs/<category>/*.png             figures
  results/<category>/*.csv             numeric source tables
  results/<category>/manifest.json
  README.md
  LICENSE
```

## Run

Install the scientific Python environment using the root [installation guide](../../README.md#installation). Run these commands from **LFP dataset**:

```console
python dataset_visualization/plot.py
python dataset_visualization/plot.py --list
python dataset_visualization/plot.py --only capacity correlation
python dataset_visualization/plot.py --only complete-curves --formats png,pdf,svg
```

The first command generates every figure. t-SNE performs 21 embedding fits and takes longer than the other categories. The default CPU thread count is two; change it with `--threads`. The command-line entry executes notebook Python cells in order and does not require Jupyter. Alternatively, open a notebook in Jupyter or VS Code and run all cells. Its working directory may be the notebook directory, the LFP dataset directory, or the repository root.

Figures default to this directory's `outputs`, and numeric tables default to its `results`. With `--output <directory>`, figures go to that directory and numeric tables default to its `results` subdirectory. Use `--results <directory>` to choose a separate numeric-output location. Default PNG exports use 300 dpi; the tail-range figures use 600 dpi. `--formats png,pdf,svg` also exports vector formats: PDF embeds TrueType fonts and SVG preserves text. Vector files containing thousands of curves can be large. Generated images are saved separately from the notebooks.

Notebook environment variables: `BATTERY_DATA_DIR` selects a directory containing both MAT files; `LFP_PROCESSED_MAT` overrides only the processed MAT. `LFP_FIGURE_OUTPUT`, `LFP_FIGURE_RESULTS`, and `LFP_FIGURE_FORMATS` control the figure directory, numeric directory, and formats. Packaged data are used by default; custom data paths and hashes are recorded in each result manifest. The CLI `--data <processed.mat>` option overrides the processed MAT; raw curves still use `BATTERY_DATA_DIR` or the packaged raw MAT.

## Numeric definitions and output records

- The raw cohort contains 4,000 cell records, of which 3,511 remain after processing. Raw curves use the raw cohort; `plotted_raw_cells.csv` lists the raw record IDs actually plotted successfully. Other analyses use the processed cohort. Raw record IDs are one-based; `cell_index_zero_based` is the zero-based column in the processed MAT.
- Maximum capacity is the final row of `processed_Qd_mat`, in Ah. SOH = capacity / 15.5 Ah × 100%. The capacity histogram's secondary axis shows within-bin sample standard deviation (`ddof=1`), not confidence intervals. Bins with fewer than two samples have NaN standard deviations. The SOH plot uses 3% bins and a 3D pie, including a `>100%` group for five cells whose capacities slightly exceed nominal capacity. Percentages use all 3,511 cells as the denominator.
- IC plots read `processed_qIC_mat` and `processed_vIC_mat` directly from the MAT. Model training recomputes dQ/dV from resampled Q–V curves; these are distinct input-preparation procedures.
- Initial voltages are simulated with mean 2.85 V, standard deviation 0.10 V, and seed 999. Only draws at or below the tail upper boundary are rejected; there is no additional upper cutoff. The normal curve depicts the specified sampling distribution, not a fit to measured initial voltages. Fragment-availability counts are cumulative, rather than mutually exclusive groups of cells.
- SOC is remaining discharge capacity interpolated at a voltage, divided by maximum capacity. Interpolation outside a cell's voltage range returns NaN; those values are excluded from histograms but kept in per-cell CSV files. Voltage-span fractions use the 2.50–3.65 V reference interval.
- t-SNE uses raw fragment Q features, `random_state=42`, PCA initialization, automatic learning rate, perplexity 30, and at most 1,000 iterations. Each tail or fragment is fitted separately; embedding axes are not comparable physical quantities. No model predictions enter these plots. CSV files contain all cell IDs, capacities, and embedding coordinates; the MAT row indices are also saved.
- Each correlation scatter point corresponds to one voltage-sample row, with Pearson r computed across cells. Violin/box plots describe the distribution of r across rows, not confidence intervals from independent cell replicates. Visualization fragment selections include some overlapping endpoints; model windows are disjoint.
- Each `manifest.json` records the actual MAT SHA-256, notebook SHA-256, figure and numeric-table inventories, and software versions. Curve coordinates come directly from the referenced MAT without re-exporting the complete dataset. See `../dataset/cell_index_mapping.csv` for the complete raw-to-processed mapping.

For model inference, set `RUN_MODE = "infer"` in `model_construction/Proposed framework/config.py` and run the adjacent `main.py`. See the [model guide](../model_construction/Proposed%20framework/README.md) for configuration and the example checkpoint.

The supplied code is covered by its MIT licence. Data-use conditions are explained in the root [data-availability document](../../AVAILABILITY.md); the code licence does not override data terms.
