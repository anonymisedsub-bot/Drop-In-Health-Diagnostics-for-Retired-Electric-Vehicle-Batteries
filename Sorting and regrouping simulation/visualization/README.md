# Figure 7: sorting and regrouping

This folder contains the portable Python renderers for Figure 7 sorting
and regrouping panels a–l.
The scientific evidence is the supplied LFP simulation, with 27 model conditions,
1,000 paired batches per model and 300 sampled records per batch.

From the package root:

```powershell
python run.py regrouping --mode figures
```

The default result is `Sorting and regrouping simulation/visualization/outputs/`:

- `panel_a.png` through `panel_l.png`: 12 independently drawn, native-size,
  600 dpi PNG panels. Outer panel letters and titles are omitted for manuscript assembly.
- `Figure_7_overview.png`: labelled navigation preview of all panels.
- `../results/figure_data/`: generated numerical plotting tables, linked from `panel_manifest.json`.
- `FIGURE_7_CAPTION.md`, `panel_manifest.json`, `qa/`: caption, traceability and checks.

An explicit `--output runs/figure7` writes the same structure under `runs/figure7/visualization/outputs/`.
No training or Monte Carlo simulation is repeated by the figure-only command.
The `full` command reconstructs the simulation first, then renders the same panels.

| Panels | Evidence |
|---|---|
| a | Schematic of the five strategies and eight-cell regrouping |
| b–e | Representative 300-record batch: selection, RPT flow, cell and pack capacities |
| f–i | Sorting quality, residual risk, pass rate, yield, capacity and consistency at the representative setting |
| j–l | Paired S5–S4 yield/pass changes and RPT workload over nine settings |

The representative setting is 1,000 training cells, a 2.5–2.75 V tail segment,
and three model trials with 1,000 batches each. The b–e example is trial index 1,
batch 0. RPT is simulated exact revelation of selected measured labels.
Central 95% batch ranges are not population mean confidence intervals.

`../results/reference/figure_source/` contains nine manuscript reference tables. Rendering
checks all nine comparisons (including the combined h/i distribution table) and
nine result-file content hashes. Gzip container timestamps are excluded
from those input checks. These references do not replace the raw simulation
tables: each panel is recomputed from the packaged outputs before comparison.

The compressed simulation tables and sampling indices supply the numerical
evidence for reproduction. The panel manifest records source references and
export checks for all twelve PNGs.
