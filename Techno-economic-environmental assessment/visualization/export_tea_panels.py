"""Export targeted-RPT PNG panels and shared assets from verified variant sources."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from matplotlib.text import Text
from matplotlib.transforms import Bbox

from tea_figure import (
    ROOT, FIGURE_SOURCE, GROUP_SPECS, METHODS, COST_METHODS, COMPARATORS, LABELS, METHOD_NAMES, SAVINGS_LABELS, STYLE, FONT_PT, matplotlib, plt,
    np, load_core, load_verified_density, source_grid, _hash, validate_destination,
    source_paths, input_hashes, render_current, configure_response_norm,
)


DEFAULT_OUTPUT = ROOT / "visualization/outputs"
PADDING_MM = 2
PNG_DPI = 300
# Preserve the previously delivered panel-b PNG width while accommodating longer labels.
PANEL_B_WIDTH_MM = 109.28930555555559
PANEL_FILES = dict(zip("abcdefghijkl", (
    "panel_a_workflow", "panel_b_cost_distribution", "panel_c_relative_reductions",
    "panel_d_cost_savings", "panel_e_electricity_savings", "panel_f_co2_savings",
    "panel_g_capacity_sensitivity", "panel_h_initial_soc_sensitivity",
    "panel_i_label_fraction_sensitivity", "panel_j_fraction_initial_soc",
    "panel_k_capacity_initial_soc", "panel_l_fraction_capacity",
)))
ASSET_FILES = {key: name for key, name in {
    **PANEL_FILES,
    "legend_methods": "legend_methods_panels_d_i",
    "colorbar_reductions": "colourbar_panel_c",
    "colorbar_response": "colourbar_panels_j_l",
    "monte_carlo_note": "monte_carlo_note_panels_b_f",
    "legend_cost_methods": "legend_methods_panel_b",
    "legend_mc_statistics": "legend_statistics_panels_b_f",
}.items()}
PANEL_ACCESSORIES = {
    "a": (), "b": ("legend_cost_methods", "legend_mc_statistics"), "c": ("colorbar_reductions",),
    **{key: ("legend_methods", "legend_mc_statistics") for key in "def"},
    **{key: ("legend_methods",) for key in "ghi"},
    **{key: ("colorbar_response",) for key in "jkl"},
}


def prepare_figure(core, density):
    fig = render_current(core, density)
    for ax in fig.axes:
        if (ax.get_gid() or "").startswith("panel:"):
            for loc in ("left", "center", "right"):
                ax.set_title("", loc=loc)
        for text in list(ax.texts):
            if (text.get_gid() or "").startswith("panel-letter:"):
                text.remove()
    for text in list(fig.texts):
        if (text.get_gid() or "").startswith(("panel-letter:", "panel-title:")):
            text.remove()
    fit_cost_panels(fig)
    return fig


def fit_cost_panels(fig):
    """Fit b without changing type size; match c's total exported height to b."""
    assets = build_assets(fig)
    b, c = assets['b'], assets['c']
    width_mm, height_mm = fig.get_size_inches() * 25.4
    for _ in range(3):
        with isolate(fig, b):
            _, report = measure_asset(fig, b)
        delta = PANEL_B_WIDTH_MM - report['size_mm'][0]
        ax = b['axes'][0]
        box = ax.get_position()
        new_width = box.width + delta / width_mm
        if new_width * width_mm < 25:
            raise ValueError('Panel-b labels leave insufficient plotting width')
        ax.set_position((box.x0, box.y0, new_width, box.height))
    with isolate(fig, b):
        _, target = measure_asset(fig, b)
    for _ in range(3):
        with isolate(fig, c):
            _, report = measure_asset(fig, c)
        delta = target['size_mm'][1] - report['size_mm'][1]
        for ax in c['axes']:
            box = ax.get_position()
            ax.set_position((box.x0, box.y0, box.width, box.height + delta / height_mm))


def build_assets(fig):
    axes = {ax.get_gid(): ax for ax in fig.axes}
    legends = {legend.get_gid(): legend for legend in fig.legends}
    assets = {letter: dict(axes=[axes[f"panel:{letter}"]], texts=[], legends=[])
              for letter in "abcdefghi"}
    assets["c"]["axes"].append(axes["colorbar:reductions"])
    for spec in GROUP_SPECS:
        group = [axes[f"response:{spec['grid_id']}:{method}"] for method in COMPARATORS]
        left = group[0].get_position().x0
        center = (left + group[1].get_position().x1) / 2
        labels = []
        for name, rotation, target_x in ((spec["xlabel"], 0, center), (spec["ylabel"], 90, left)):
            candidates = [t for t in fig.texts if t.get_text() == name and t.get_rotation() == rotation]
            labels.append(min(candidates, key=lambda t: abs(t.get_position()[0] - target_x)))
        assets[spec["letter"]] = dict(axes=group, texts=labels, legends=[])
    assets["legend_methods"] = dict(axes=[], texts=[], legends=[legends["legend:methods"]])
    assets["colorbar_reductions"] = dict(axes=[axes["colorbar:reductions"]], texts=[], legends=[])
    assets["colorbar_response"] = dict(
        axes=[axes["colorbar:response"]],
        texts=[next(t for t in fig.texts if t.get_text() == "Cost reduction (%)")], legends=[])
    assets["monte_carlo_note"] = dict(
        axes=[], texts=[next(t for t in fig.texts if t.get_text().startswith("Monte Carlo:"))], legends=[])
    assets["legend_cost_methods"] = dict(axes=[], texts=[], legends=[legends["legend:cost_methods"]])
    assets["legend_mc_statistics"] = dict(axes=[], texts=[], legends=[legends["legend:mc_statistics"]])
    return assets


@contextmanager
def isolate(fig, asset):
    states = [(artist, artist.get_visible()) for artist in (*fig.axes, *fig.texts, *fig.legends)]
    selected = {*asset["axes"], *asset["texts"], *asset["legends"]}
    try:
        for artist, _ in states:
            artist.set_visible(artist in selected)
        yield
    finally:
        for artist, visible in states:
            artist.set_visible(visible)


def asset_texts(asset):
    texts = list(asset["texts"])
    for ax in asset["axes"]:
        texts.extend(child for child in ax.get_children() if isinstance(child, Text))
        for axis in (ax.xaxis, ax.yaxis):
            texts.extend((axis.label, axis.get_offset_text()))
            low, high = sorted(axis.get_view_interval())
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if low - 1e-9 <= tick.get_loc() <= high + 1e-9:
                    texts.extend((tick.label1, tick.label2))
    for legend in asset["legends"]:
        texts.extend(legend.get_texts())
    return [t for t in dict.fromkeys(texts) if t.get_visible() and t.get_text().strip()]


def measure_asset(fig, asset):
    # Keep measurement and export settings identical.
    with matplotlib.rc_context(STYLE):
        return _measure_asset(fig, asset)


def _measure_asset(fig, asset):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in asset["axes"]]
    boxes.extend(a.get_window_extent(renderer) for a in (*asset["texts"], *asset["legends"]))
    texts = asset_texts(asset)
    # Include full label extents, which axes-level tight boxes can shorten.
    boxes.extend(text.get_window_extent(renderer) for text in texts)
    crop_px = Bbox.union(boxes).padded(PADDING_MM / 25.4 * fig.dpi)
    outside, overlaps = [], []
    for index, text in enumerate(texts):
        box = text.get_window_extent(renderer)
        if not (crop_px.x0 <= box.x0 <= box.x1 <= crop_px.x1 and
                crop_px.y0 <= box.y0 <= box.y1 <= crop_px.y1):
            outside.append(text.get_text())
        for other in texts[index + 1:]:
            second = other.get_window_extent(renderer)
            dx = min(box.x1, second.x1) - max(box.x0, second.x0)
            dy = min(box.y1, second.y1) - max(box.y0, second.y0)
            if dx > 1 and dy > 1:
                overlaps.append([text.get_text(), other.get_text()])
    crop = crop_px.transformed(fig.dpi_scale_trans.inverted())
    minimum = min(t.get_fontsize() for t in texts)
    report = dict(size_mm=(crop.size * 25.4).tolist(), minimum_font_pt=minimum,
                  crop_padding_mm=PADDING_MM, text_outside_crop=outside, text_overlaps=overlaps,
                  passed=not outside and not overlaps and minimum >= FONT_PT)
    return crop, report


def export_asset(fig, asset, output, base):
    with matplotlib.rc_context(STYLE), isolate(fig, asset):
        crop, report = measure_asset(fig, asset)
        if not report["passed"]:
            raise ValueError(f"Panel crop failed: {base}: {report}")
        fig.savefig(output / f"{base}.png", bbox_inches=crop, dpi=PNG_DPI)
        report["text"] = [t.get_text() for t in asset_texts(asset)]
        report["font_sizes_pt"] = sorted({t.get_fontsize() for t in asset_texts(asset)})
        report["axes_size_mm"] = [(ax.get_position().size * fig.get_size_inches() * 25.4).tolist()
                                  for ax in asset["axes"]]
        if len(asset["axes"]) == 4 and all((ax.get_gid() or "").startswith("response:") for ax in asset["axes"]):
            boxes = [ax.get_position() for ax in asset["axes"]]
            width, height = fig.get_size_inches() * 25.4
            report["heatmap_gaps_mm"] = dict(horizontal=(boxes[1].x0 - boxes[0].x1) * width,
                                             vertical=(boxes[0].y0 - boxes[2].y1) * height)
    report["base"] = base
    report["exports"] = {"png": _hash(output / f"{base}.png")}
    return report


def write_index(output, assets, core, source_data=None):
    source_data=Path(source_data) if source_data is not None else ROOT/"results/figure_data"
    source_data.mkdir(parents=True,exist_ok=True)
    bounds = configure_response_norm(core)
    lines = ["# Targeted-RPT TEA panels", "",
             "Each panel is exported without its panel title or a-l identifier. The j-l method names are retained.",
             "The protocol includes full RPT from zero SOC for q=4% of the deployment pool after diagnosis. "
             f"Method colours are consistent across panels. All panel and shared-asset text is {FONT_PT:g} pt.",
             f"Current response values range from {bounds['minimum_pct']:.6f}% to {bounds['maximum_pct']:.6f}%. "
             f"The common colour scale is {bounds['vmin']:g}-{bounds['vmax']:g}% "
             "(default: 30-95%; expanded in 5-point steps when necessary). No source values are clipped.",
             "Panels b-f use 16 independent triangular inputs, including q ~ Tri(0.03, 0.04, 0.05). "
             "Panels g-l keep q=0.04 fixed; Proposed-versus-RPT curves and surfaces use the current data.",
             "", "PNG-only export at 300 dpi. "
             "Each asset has 2-mm white padding. Use the physical dimensions below when assembling; "
             f"100% placement preserves {FONT_PT:g}-pt type. Scaling below {800 / FONT_PT:.2f}% makes the text smaller than 8 pt.",
             "", "## File index", "", "| Asset | Width x height (mm) | Files |",
             "|---|---|---|"]
    for key, asset in assets.items():
        w, h = asset["size_mm"]
        links = f"[PNG]({asset['base']}.png)"
        lines.append(f"| {key} | {w:.2f} x {h:.2f} | {links} |")
    lines.extend(["", "## Assembly notes", "",
                  "- All panel, legend, colour-bar and note PNGs are in this outputs directory. See PANEL_ASSETS.md for the panel-to-accessory map.",
                  "- Panel b has six cost scenarios, including Proposed (no targeted RPTs) and Proposed+targeted RPTs; panels d-i share the four-comparator marker legend.",
                  f"- Panel b keeps its {PANEL_B_WIDTH_MM:.6f}-mm output width with a shorter x axis; panel c matches its total PNG height, including labels and padding.",
                  "- Method order throughout: Proposed (where applicable), Pulse-based, EIS-based, Partial-charge-based, RPT-based. Colours and marker shapes consistently identify each method.",
                  "- Panels b-f have a separate Median / 95% simulation interval legend; intervals are central simulation ranges, not confidence intervals.",
                  "- Panels e/f use a shared y-axis layout; their method order is the same as d.",
                  "- Savings intervals can cross zero: negative electricity/CO2 savings indicate an increase in burden. Negative intervals have visible negative-axis ticks and a dashed zero reference; no interval is clipped.",
                  "- Panels g-i use the separate methods legend. It also identifies methods in d-f.",
                  "- Panel c includes its borderless colour bar labelled Median reduction (%) (0-100%). A standalone copy is also supplied.",
                  "- Panels d-f explicitly label absolute cost, electricity and CO2 emissions savings, with units on a second line to preserve plot widths and type size.",
                  "- The optional text-free Sankey schematic can be generated with export_sankey_schematic.py (command below). It routes RPT-based only to model deployment, never dataset construction. Ribbon widths are illustrative, not quantitative results.",
                  f"- Panels j-l use the borderless colour bar labelled Cost reduction (%) ({bounds['vmin']:g}-{bounds['vmax']:g}%). Their Partial-charge-based heading wraps onto two lines.",
                  "- Each j-l panel contains four maps: Pulse-based / EIS-based above Partial-charge-based / RPT-based, read left to right within each row.",
                  "- Heatmap gaps are automatically fitted to rendered titles and ticks, with 0.8-mm clearance, 17-mm maps and 10.5-pt fonts.",
                  "- The response colour bar is 3 mm wide and automatically matches the complete two-row heatmap axes height, excluding titles and tick labels.",
                  "- The Monte Carlo note is exported separately and can be placed with panels b-f.",
                  "- Figure_8_overview.png and shared_overview.png are labelled browsing sheets, not final figure panels.",
                  "- The source_index path in panel_manifest.json locates the numerical tables and cache by path and SHA-256. Paths are relative on the same drive; cross-drive exports use absolute paths. The targeted-RPT caption accompanies the files verbatim.",
                  "", 'From the repository root, rebuild numerical sources and figures: `python "Techno-economic-environmental assessment/run.py" full`',
                  'Render the packaged compact references only: `python "Techno-economic-environmental assessment/run.py" figures`',
                  'Optional schematic after panel export: `python "Techno-economic-environmental assessment/visualization/export_sankey_schematic.py"`',
                  "See panel_manifest.json for reference_only and export_verification.json for raw_draws_verified.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    with (source_data / "panel_dimensions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["asset", "width_mm", "height_mm", "font_pt", "base"])
        for key, asset in assets.items():
            writer.writerow([key, *asset["size_mm"], asset["minimum_font_pt"], asset["base"]])
    if set(ASSET_FILES).issubset(assets):
        panel_index = ["# Panel assembly assets", "",
                       "All PNGs use 10.5-pt type at their recorded physical dimensions and 300 dpi.",
                       "See [physical dimensions](README.md) before resizing.", "",
                       "| Panel | Panel PNG | Standalone legend / colour bar |", "|---|---|---|"]
        for key, accessory_keys in PANEL_ACCESSORIES.items():
            panel_file = Path(assets[key]["base"]).name + ".png"
            links = [f"[{Path(assets[item]['base']).name}]({Path(assets[item]['base']).name}.png)"
                     for item in accessory_keys]
            panel_index.append(f"| {key} | [PNG]({panel_file}) | {'; '.join(links) or 'Not needed'} |")
        bar_width, bar_height = assets["colorbar_response"]["axes_size_mm"][0]
        panel_index.extend(["", "Method order: Proposed (where applicable), Pulse-based, EIS-based, Partial-charge-based, RPT-based.",
                            "The 2-by-2 maps in j-l follow this order row by row, starting at the top left (excluding Proposed).",
                            "Panel c already includes its labelled colour bar; its standalone copy is optional.",
                            "Panels j-l share the same labelled colour bar and need only one copy in a combined figure.",
                            f"Its coloured strip matches the two-row map extent (currently {bar_height:g} mm high and {bar_width:g} mm wide). Align the strip endpoints to the heatmap edges, not the padded PNG edges.",
                            "The statistics legend describes the median and the 2.5th-97.5th percentile simulation range, not a confidence interval.",
                            "For panels d-f, the method-specific markers indicate the median; the generic black point identifies that statistic.",
                            "The optional export_sankey_schematic.py command creates a separate text-free schematic; it is not part of the standard Figure 8 panels.",
                            "The Monte Carlo note is available separately as [PNG](monte_carlo_note_panels_b_f.png).", ""])
        (output / "PANEL_ASSETS.md").write_text("\n".join(panel_index), encoding="utf-8")


def make_overviews(output, assets):
    from PIL import Image

    with matplotlib.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 13), dpi=180)
        grid = fig.add_gridspec(5, 3, height_ratios=[.7, 1, .8, 1, 1.35],
                               left=.035, right=.985, top=.98, bottom=.03, wspace=.18, hspace=.3)
        slots = {"a": grid[0, :], "b": grid[1, :2], "c": grid[1, 2]}
        for row, letters in enumerate(("def", "ghi", "jkl"), start=2):
            slots.update({letter: grid[row, column] for column, letter in enumerate(letters)})
        for key, slot in slots.items():
            ax = fig.add_subplot(slot)
            with Image.open(output / f"{assets[key]['base']}.png") as image:
                ax.imshow(np.asarray(image))
            ax.set_axis_off()
            ax.set_title(f"Panel {key}", fontsize=12, pad=7, loc="left")
        fig.savefig(output / "Figure_8_overview.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 1, 2]})
        keys = ("legend_cost_methods", "legend_methods", "legend_mc_statistics", "monte_carlo_note",
                "colorbar_reductions", "colorbar_response")
        for ax, key in zip(axes.flat, keys):
            with Image.open(output / f"{assets[key]['base']}.png") as image:
                ax.imshow(np.asarray(image))
            ax.set_axis_off()
            ax.set_title(key, fontsize=12, pad=7)
        fig.tight_layout()
        fig.savefig(output / "shared_overview.png", dpi=180)
        plt.close(fig)


def verify_raster(path, size_mm, dpi):
    from PIL import Image

    with Image.open(path) as image:
        np.testing.assert_allclose([float(value) for value in image.info["dpi"]], [dpi, dpi], atol=.1)
        np.testing.assert_allclose(image.size, np.asarray(size_mm) / 25.4 * dpi, atol=1.1)
        pixels = np.asarray(image.convert("RGB"))
        assert np.any(pixels < 245), (path, "blank")
        assert min(pixels[0].min(), pixels[-1].min(), pixels[:, 0].min(), pixels[:, -1].min()) >= 250


def verify_exports(output, input_dir=None, cache_dir=None):
    output = Path(output)
    input_dir = Path(input_dir) if input_dir is not None else ROOT / "results/runs/full"
    cache_dir = Path(cache_dir) if cache_dir is not None else ROOT / "results/runs/full/figure_source"
    validate_destination(output, input_dir)
    validate_destination(output, cache_dir)
    manifest = json.loads((output / "panel_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_format"] == "png"
    assert manifest["method_order"] == list(METHODS)
    assert manifest['panel_b_cost_order'] == list(COST_METHODS)
    np.testing.assert_allclose(manifest['assets']['b']['size_mm'][0], PANEL_B_WIDTH_MM, atol=1e-7)
    np.testing.assert_allclose(manifest['assets']['b']['size_mm'][1], manifest['assets']['c']['size_mm'][1], atol=1e-7)
    unwanted = [str(p.relative_to(output)) for p in output.rglob("*")
                if p.is_file() and p.suffix.lower() in (".pdf", ".svg", ".tif", ".tiff")]
    assert not unwanted, f"Non-PNG figure files remain: {unwanted}"
    assert manifest["input_hashes"] == input_hashes(input_dir, cache_dir, include_draws=not manifest.get("reference_only", False)), "Analysis inputs changed"
    assert manifest["n_free_inputs"] == 16 and manifest["fixed_sensitivity_q"] == .04
    assert _hash(output / "source_figure_caption.md") == _hash(cache_dir / "figure_caption.md")
    scale = manifest["response_scale"]
    assert scale["vmin"] <= scale["minimum_pct"] <= scale["maximum_pct"] <= scale["vmax"]
    assert set(manifest["assets"]) == set(ASSET_FILES)
    assert manifest["panel_accessories"] == {key: list(value) for key, value in PANEL_ACCESSORIES.items()}
    index_path=(output/manifest["source_index"]).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sources = source_paths(input_dir, cache_dir)
    assert set(index) == set(manifest["source_hashes"]) == set(sources)
    for name, expected in manifest["source_hashes"].items():
        source = (index_path.parent / index[name]["path"]).resolve()
        assert source == sources[name].resolve(), f"Unexpected source path: {name}"
        assert index[name]["sha256"] == _hash(source) == expected
    verified = {}
    forbidden = {"Capacity retention", "Initial SOC", "Labelled set fraction"}
    forbidden.update(spec["title"] for spec in GROUP_SPECS)
    forbidden.update("abcdefghijkl")
    for key, asset in manifest["assets"].items():
        assert set(asset["exports"]) == {"png"}
        assert asset["base"] == ASSET_FILES[key]
        path = output / f"{asset['base']}.png"
        assert _hash(path) == asset["exports"]["png"]
        assert asset["passed"] and not asset["text_outside_crop"] and not asset["text_overlaps"]
        assert asset["font_sizes_pt"] == [FONT_PT], (key, asset["font_sizes_pt"])
        runs = [line.strip() for text in asset["text"] for line in text.splitlines() if line.strip()]
        assert forbidden.isdisjoint(runs), (key, forbidden.intersection(runs))
        if key in ("c", "colorbar_reductions"):
            assert "Median reduction (%)" in asset["text"]
        if key == "colorbar_response":
            assert "Cost reduction (%)" in asset["text"]
        if key in "def":
            assert list(SAVINGS_LABELS.values())["def".index(key)] in asset["text"]
        if key in "jkl":
            method_labels = [text.replace("\n", "") for text in asset["text"]]
            assert [text for text in method_labels if text in METHOD_NAMES.values()] == list(METHOD_NAMES.values()), key
            assert "Partial-charge-\nbased" in asset["text"], key
        if key in ("b", "c", "d", "legend_cost_methods", "legend_methods"):
            expected = [LABELS[method] for method in (COST_METHODS if key in ("b", "legend_cost_methods") else COMPARATORS)]
            assert [text for text in asset["text"] if text in LABELS.values()] == expected, key
        verify_raster(path, asset["size_mm"], PNG_DPI)
        verified[key] = dict(size_mm=asset["size_mm"], font_sizes_pt=asset["font_sizes_pt"], passed=True)
    make_overviews(output, manifest["assets"])
    report = dict(passed=True, raw_draws_verified=not manifest.get("reference_only", False), panel_count=12, shared_asset_count=6, assets=verified,
                  accessories_in_panels=True, colour_bar_labels_verified=True,
                  method_order_verified=True, method_order=[LABELS[method] for method in METHODS],
                  export_format="png", original_physical_scale=True, png_dpi=PNG_DPI,
                  source_data_unchanged=True, historical_dependencies=False,
                  text_check_basis="Source Matplotlib artists, not OCR of PNG pixels",
                  raster_checks="PNG hashes, DPI, physical size, nonblank content and white edge pixels",
                  agent_visual_review_performed=False)
    (output / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified 12 PNG panels and 6 PNG assembly assets in visualization/outputs at 300 dpi; analysis sources unchanged", flush=True)


def indexed_source_path(source, index_directory):
    """Use portable relative paths, or absolute paths for cross-drive exports."""
    try:
        return Path(os.path.relpath(source, index_directory)).as_posix()
    except ValueError:
        return Path(source).resolve().as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Render packaged compact references; omitted raw draws are not reverified.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/runs/full")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/runs/full/figure_source")
    parser.add_argument("--source-data",type=Path,default=ROOT/"results/figure_data")
    args = parser.parse_args(argv)
    if args.verify_only:
        verify_exports(args.output_dir, args.input_dir, args.cache_dir)
        return
    validate_destination(args.output_dir, args.input_dir)
    validate_destination(args.output_dir, args.cache_dir)
    missing = [str(path) for path in source_paths(args.input_dir, args.cache_dir).values() if not path.is_file()]
    if missing:
        parser.error("Variant data/cache are not ready; run build_density_cache.py after analysis completes. Missing: "
                     + ", ".join(missing))
    before = input_hashes(args.input_dir, args.cache_dir, include_draws=not args.reference_only)
    core = load_core(args.input_dir)
    if core["manifest"].get("synthetic_test_only"):
        raise ValueError("Synthetic fixtures cannot produce final panels")
    density, provenance = load_verified_density(args.input_dir, args.cache_dir, reference_only=args.reference_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from portable_io import validate_output
    args.source_data=validate_output(args.source_data)
    args.source_data.mkdir(parents=True,exist_ok=True)
    sources = source_paths(args.input_dir, args.cache_dir)
    source_index = {name: dict(path=indexed_source_path(path, args.source_data),
                               sha256=_hash(path)) for name, path in sources.items()}
    (args.source_data / "source_index.json").write_text(json.dumps(source_index, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.cache_dir / "figure_caption.md", args.output_dir / "source_figure_caption.md")
    records = {}
    with matplotlib.rc_context(STYLE):
        fig = prepare_figure(core, density)
        try:
            for key, asset in build_assets(fig).items():
                base = ASSET_FILES[key]
                records[key] = export_asset(fig, asset, args.output_dir, base)
                print(f"Exported {key}: {records[key]['size_mm']} mm", flush=True)
        finally:
            plt.close(fig)
    assert before == input_hashes(args.input_dir, args.cache_dir, include_draws=not args.reference_only), "Analysis inputs changed"
    manifest = dict(figure="Figure 8", reference_only=args.reference_only, primary_seed=20260921, monte_carlo_draws=800000, resampled=False,
                    n_free_inputs=16, fixed_sensitivity_q=.04, response_scale=configure_response_norm(core),
                    method_order=METHODS, panel_b_cost_order=COST_METHODS,
                    source_edition="targeted_RPT_current", panel_titles_removed=True, panel_letters_removed=True,
                    heatmap_method_names_retained=True, partial_charge_heading_wrapped=True,
                    panel_c_colour_bar_included=True, colour_bars_labelled=True, physical_scale_preserved=True,
                    panel_accessories=PANEL_ACCESSORIES, accessories_in_panels=True,
                    minimum_font_pt=FONT_PT, export_format="png", png_dpi=PNG_DPI, density_provenance=provenance,
                    input_hashes=before, script_sha256=_hash(__file__), renderer_sha256=_hash(ROOT / "visualization/tea_figure.py"),
                    source_hashes={name: _hash(path) for name, path in sources.items()},
                    source_storage="path_index",
                    source_index=indexed_source_path(args.source_data/"source_index.json",args.output_dir),
                    assets=records, python_executable=sys.executable,
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    (args.output_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_index(args.output_dir, records, core, args.source_data)
    verify_exports(args.output_dir, args.input_dir, args.cache_dir)


if __name__ == "__main__":
    main()
