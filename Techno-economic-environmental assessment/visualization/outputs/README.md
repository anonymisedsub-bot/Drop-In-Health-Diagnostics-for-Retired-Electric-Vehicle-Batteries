# Targeted-RPT TEA panels

Each panel is exported without its panel title or a-l identifier. The j-l method names are retained.
The protocol includes full RPT from zero SOC for q=4% of the deployment pool after diagnosis. Method colours are consistent across panels. All panel and shared-asset text is 10.5 pt.
Current response values range from 29.551718% to 90.328392%. The common colour scale is 25-95% (default: 30-95%; expanded in 5-point steps when necessary). No source values are clipped.
Panels b-f use 16 independent triangular inputs, including q ~ Tri(0.03, 0.04, 0.05). Panels g-l keep q=0.04 fixed; Proposed-versus-RPT curves and surfaces use the current data.

PNG-only export at 300 dpi. Each asset has 2-mm white padding. Use the physical dimensions below when assembling; 100% placement preserves 10.5-pt type. Scaling below 76.19% makes the text smaller than 8 pt.

## File index

| Asset | Width x height (mm) | Files |
|---|---|---|
| a | 232.00 x 61.91 | [PNG](panel_a_workflow.png) |
| b | 109.29 x 48.08 | [PNG](panel_b_cost_distribution.png) |
| c | 100.14 x 48.08 | [PNG](panel_c_relative_reductions.png) |
| d | 80.99 x 43.40 | [PNG](panel_d_cost_savings.png) |
| e | 53.25 x 43.40 | [PNG](panel_e_electricity_savings.png) |
| f | 50.21 x 43.40 | [PNG](panel_f_co2_savings.png) |
| g | 63.65 x 41.82 | [PNG](panel_g_capacity_sensitivity.png) |
| h | 56.57 x 39.82 | [PNG](panel_h_initial_soc_sensitivity.png) |
| i | 56.57 x 39.82 | [PNG](panel_i_label_fraction_sensitivity.png) |
| j | 56.52 x 65.50 | [PNG](panel_j_fraction_initial_soc.png) |
| k | 59.50 x 65.50 | [PNG](panel_k_capacity_initial_soc.png) |
| l | 56.52 x 65.50 | [PNG](panel_l_fraction_capacity.png) |
| legend_methods | 156.21 x 10.35 | [PNG](legend_methods_panels_d_i.png) |
| colorbar_reductions | 19.85 x 45.50 | [PNG](colourbar_panel_c.png) |
| colorbar_response | 17.39 x 53.39 | [PNG](colourbar_panels_j_l.png) |
| monte_carlo_note | 116.77 x 11.31 | [PNG](monte_carlo_note_panels_b_f.png) |
| legend_cost_methods | 145.73 x 15.59 | [PNG](legend_methods_panel_b.png) |
| legend_mc_statistics | 83.62 x 10.35 | [PNG](legend_statistics_panels_b_f.png) |

## Assembly notes

- All panel, legend, colour-bar and note PNGs are in this outputs directory. See PANEL_ASSETS.md for the panel-to-accessory map.
- Panel b has six cost scenarios, including Proposed (no targeted RPTs) and Proposed+targeted RPTs; panels d-i share the four-comparator marker legend.
- Panel b keeps its 109.289306-mm output width with a shorter x axis; panel c matches its total PNG height, including labels and padding.
- Method order throughout: Proposed (where applicable), Pulse-based, EIS-based, Partial-charge-based, RPT-based. Colours and marker shapes consistently identify each method.
- Panels b-f have a separate Median / 95% simulation interval legend; intervals are central simulation ranges, not confidence intervals.
- Panels e/f use a shared y-axis layout; their method order is the same as d.
- Savings intervals can cross zero: negative electricity/CO2 savings indicate an increase in burden. Negative intervals have visible negative-axis ticks and a dashed zero reference; no interval is clipped.
- Panels g-i use the separate methods legend. It also identifies methods in d-f.
- Panel c includes its borderless colour bar labelled Median reduction (%) (0-100%). A standalone copy is also supplied.
- Panels d-f explicitly label absolute cost, electricity and CO2 emissions savings, with units on a second line to preserve plot widths and type size.
- The optional text-free Sankey schematic can be generated with export_sankey_schematic.py (command below). It routes RPT-based only to model deployment, never dataset construction. Ribbon widths are illustrative, not quantitative results.
- Panels j-l use the borderless colour bar labelled Cost reduction (%) (25-95%). Their Partial-charge-based heading wraps onto two lines.
- Each j-l panel contains four maps: Pulse-based / EIS-based above Partial-charge-based / RPT-based, read left to right within each row.
- Heatmap gaps are automatically fitted to rendered titles and ticks, with 0.8-mm clearance, 17-mm maps and 10.5-pt fonts.
- The response colour bar is 3 mm wide and automatically matches the complete two-row heatmap axes height, excluding titles and tick labels.
- The Monte Carlo note is exported separately and can be placed with panels b-f.
- Figure_8_overview.png and shared_overview.png are labelled browsing sheets, not final figure panels.
- The source_index path in panel_manifest.json locates the numerical tables and cache by path and SHA-256. Paths are relative on the same drive; cross-drive exports use absolute paths. The targeted-RPT caption accompanies the files verbatim.

From the repository root, rebuild numerical sources and figures: `python "Techno-economic-environmental assessment/run.py" full`
Render the packaged compact references only: `python "Techno-economic-environmental assessment/run.py" figures`
Optional schematic after panel export: `python "Techno-economic-environmental assessment/visualization/export_sankey_schematic.py"`
See panel_manifest.json for reference_only and export_verification.json for raw_draws_verified.
