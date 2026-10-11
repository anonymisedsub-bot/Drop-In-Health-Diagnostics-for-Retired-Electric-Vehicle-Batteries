# Code and data availability

## Code

The package includes model source code, configurations, processed and raw datasets, saved checkpoints, numerical results and plotting scripts. Software use is governed by the MIT [licence](LICENSE). Dataset terms are specified separately below.

## LFP

The LFP data are provided by [CETL-HFUT](https://github.com/CETL-HFUT/Imbalanced-retired-battery-data). The package includes `CETL-data.mat`, `LFP_data_processed.mat`, MATLAB preprocessing and a raw-to-processed sample map. These represent raw and processed forms of the same dataset.

The provider specifies academic research use and prohibits commercial use of the data. Cite the dataset and associated article as listed in [CITATION.md](CITATION.md). The MIT software licence does not change the dataset terms.

## NMC_A and NMC_B

`Self-collected datasets/` includes NMC_A (353 cells) and NMC_B (334 cells), their processed MAT files, and 687 workbooks containing `DisC` and `DisCC2`. Public sample IDs run from 1 to 353 and 1 to 334, respectively, and correspond to one-based MAT columns. Keep the dataset name with each sample ID.

The sample mapping supplies workbook paths, identifiers and integrity hashes used by preprocessing. The public preprocessing script reconstructs eight specified arrays; see its [README](Self-collected%20datasets/preprocessing/README.md).

NMC raw and processed data are available for academic research only; commercial use is prohibited. See [NMC data terms](Self-collected%20datasets/dataset/DATA_LICENSE.md).

## Sorting and regrouping simulation

The module includes its LFP input MAT, saved model, sampling records, five sorting policies and reference results. Simulated targeted RPT reveals the measured capacity of selected records. Pack capacity follows the stated minimum-cell-capacity rule for eight-cell series packs.

## Techno-economic-environmental assessment

The TEA module includes its parameter registry, baseline model, Monte Carlo and sensitivity-analysis code, reference summaries, plotting tables and density caches. Its uncertainty intervals describe the specified economic and environmental scenarios. See the [module README](Techno-economic-environmental%20assessment/README.md) for execution and interpretation.
