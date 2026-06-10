# Dataset Visualization Notebooks

This repository contains the notebooks used to generate the dataset
visualization figures for the retired battery diagnosis experiments.

The folder is organized as a reproducible figure-generation package for code
and data availability. All notebooks resolve dataset paths through
`notebook_setup.py`, so the code does not depend on local absolute paths.

## Environment

The notebooks were prepared and checked with the following Python environment:

- Python 3.12.7
- ipykernel 6.29.5
- matplotlib 3.10.1
- numpy 2.1.3
- scipy 1.15.2
- scikit-learn 1.6.1

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The `jupyter` metapackage is not required by the code itself. If notebook
execution from the command line is needed, install Jupyter separately in the
same environment.

## Data Location

The notebooks load MATLAB files through `notebook_setup.py`. To reproduce the
figures, either place the data files in `data/`:

- `data/CETL_data_processed.mat`
- `data/CETL-data.mat`

or set `BATTERY_DATA_DIR` to the directory containing those files.

Example:

```bash
set BATTERY_DATA_DIR=D:\path\to\battery\data
```

On Linux or macOS:

```bash
export BATTERY_DATA_DIR=/path/to/battery/data
```

## Notebooks and Outputs

Run each notebook from its own folder or from an editor that uses this project
folder as the working directory. The helper module searches upward for
`notebook_setup.py`, so both workflows are supported.

| Notebook | Main outputs |
| --- | --- |
| `Capacity distribution/capacity_distribution.ipynb` | `cap_distribution.png`, `soh_distribution.png` |
| `Raw charge-discharge curves/Combined_chg_dis_plot.ipynb` | `Charge_discharge_full_cycle_Nature_Marine.png` |
| `Complete and segmented discharge curves/VQ_IC_plot.ipynb` | `VQ_and_IC_curves_shared_colorbar_Nature_style.png` |
| `Complete and segmented discharge curves/combined_2_plus_2_plot.ipynb` | `combined_2x2_tail_boundary_swapped_Vtail_*.png` |
| `Tail+segment plot/plot.ipynb` | `fig1_tail_curve_2.50_2.73.png`, `fig2_upstream_segment_*.png`, `fig3_standalone_colorbar.png` |
| `Tail+upstream fragments/Tail_Upstream_Fragment_SOCdist.ipynb` | `fig1_tail_curve_and_SOC_Vtail_*.png`, `fig2_upstream_segment_curves_2x3_Vtail_*.png` |
| `Input tsne distribution/tsne_distribution.ipynb` | `combined_tail_and_segments_tsne_Vtail_*.png` |
| `Correlation analysis/tail_segement_corr.ipynb` | `tail_segment_subbins_correlation_Vtail_*.png` |
| `Voltage and SOC ranges of different tail settings/plot.ipynb` | `Tail_voltage_SOC_ratio_barplots/Tail_voltage_SOC_cutoff_ratio_Vtail_*.png` |

The notebooks save figures using explicit filenames and 300 dpi resolution
unless otherwise noted in the notebook.

## Reproducibility Notes

- Data paths are relative to the project or controlled by `BATTERY_DATA_DIR`.
- Randomized analyses use fixed random seeds where applicable.
- The t-SNE visualization uses `random_state=42`.
- Figure styling is defined inside each notebook for transparent reproduction.
- Existing output PNG files are included only as reference outputs; the
  notebooks are the source used to regenerate them.

## Code Availability

The code used to generate the dataset visualization figures is available in
this repository. The released version includes the Jupyter notebooks,
environment requirements, shared dataset-path helper, and representative output
figures. A permanent public repository and DOI will be provided upon
publication.

## Data Availability

The raw dataset used in this study is publicly available at the GitHub
repository "Imbalanced retired battery data" provided by CETL-HFUT
(https://github.com/CETL-HFUT/Imbalanced-retired-battery-data). The processed
data, data splits and intermediate features generated in this study will be
made available upon publication in a public repository. The dataset is provided
for academic use only, in accordance with the declaration of the original data
provider.

## License

The source code in this repository is released under the MIT License. See
`LICENSE` for the full license text.

The MIT License applies to the code and documentation in this folder. It does
not automatically apply to the accompanying dataset files unless the data
release explicitly states that the data are distributed under the same license.
