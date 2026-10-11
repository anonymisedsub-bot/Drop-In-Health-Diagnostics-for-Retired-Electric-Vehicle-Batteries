# Panel assembly assets

All PNGs use 10.5-pt type at their recorded physical dimensions and 300 dpi.
See [physical dimensions](README.md) before resizing.

| Panel | Panel PNG | Standalone legend / colour bar |
|---|---|---|
| a | [PNG](panel_a_workflow.png) | Not needed |
| b | [PNG](panel_b_cost_distribution.png) | [legend_methods_panel_b](legend_methods_panel_b.png); [legend_statistics_panels_b_f](legend_statistics_panels_b_f.png) |
| c | [PNG](panel_c_relative_reductions.png) | [colourbar_panel_c](colourbar_panel_c.png) |
| d | [PNG](panel_d_cost_savings.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png); [legend_statistics_panels_b_f](legend_statistics_panels_b_f.png) |
| e | [PNG](panel_e_electricity_savings.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png); [legend_statistics_panels_b_f](legend_statistics_panels_b_f.png) |
| f | [PNG](panel_f_co2_savings.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png); [legend_statistics_panels_b_f](legend_statistics_panels_b_f.png) |
| g | [PNG](panel_g_capacity_sensitivity.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png) |
| h | [PNG](panel_h_initial_soc_sensitivity.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png) |
| i | [PNG](panel_i_label_fraction_sensitivity.png) | [legend_methods_panels_d_i](legend_methods_panels_d_i.png) |
| j | [PNG](panel_j_fraction_initial_soc.png) | [colourbar_panels_j_l](colourbar_panels_j_l.png) |
| k | [PNG](panel_k_capacity_initial_soc.png) | [colourbar_panels_j_l](colourbar_panels_j_l.png) |
| l | [PNG](panel_l_fraction_capacity.png) | [colourbar_panels_j_l](colourbar_panels_j_l.png) |

Method order: Proposed (where applicable), Pulse-based, EIS-based, Partial-charge-based, RPT-based.
The 2-by-2 maps in j-l follow this order row by row, starting at the top left (excluding Proposed).
Panel c already includes its labelled colour bar; its standalone copy is optional.
Panels j-l share the same labelled colour bar and need only one copy in a combined figure.
Its coloured strip matches the two-row map extent (currently 46 mm high and 3 mm wide). Align the strip endpoints to the heatmap edges, not the padded PNG edges.
The statistics legend describes the median and the 2.5th-97.5th percentile simulation range, not a confidence interval.
For panels d-f, the method-specific markers indicate the median; the generic black point identifies that statistic.
The optional export_sankey_schematic.py command creates a separate text-free schematic; it is not part of the standard Figure 8 panels.
The Monte Carlo note is available separately as [PNG](monte_carlo_note_panels_b_f.png).
