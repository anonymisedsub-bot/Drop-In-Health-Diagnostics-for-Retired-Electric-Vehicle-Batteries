# Figure 8 visualization

Run `python run.py tea figures` from the package root. The 12 panels and 6 assembly assets are written directly to `outputs/` here. `Figure_8_overview.png` and `shared_overview.png` are browsing sheets.

- `tea_figure.py`: scientific plotting primitives and source validation.
- `export_tea_panels.py`: panel export, assembly metadata and raster checks.
- `build_density_cache.py`: full-run cost-density cache generation.
- `plot_summary.py`: demo/full numerical summary plots.
- `export_sankey_schematic.py`: optional text-free schematic after panel export.

Supplied source tables and density cache: `../results/reference/`. Generated plotting index and dimensions: `../results/figure_data/`. Newly computed numerical results and cache: `../results/runs/<mode>/`.

The `figures` command validates the supplied summaries and cache, recording `reference_only: true` and `raw_draws_verified: false` in its output metadata. The `full` command generates and audits new numerical draws before exporting their figures. Keep the whole module together to preserve relative data references.
