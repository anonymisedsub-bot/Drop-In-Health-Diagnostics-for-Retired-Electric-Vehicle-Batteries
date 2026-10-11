"""Text-free structural Sankey; full-batch RPT has no dataset-construction branch.

Retain the existing 1-5-2-1-4 geometry and assembly colours. Normalized flow
weights are illustrative, not costs, cell counts or probabilities. Ribbons
taper to fill the equal-height levels, as in the original schematic.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as BezierPath
from matplotlib.text import Text

from export_tea_panels import PNG_DPI, verify_raster
from tea_figure import ROOT, METHODS, STYLE, _hash, matplotlib, np, plt, validate_destination


LEVEL_COUNTS = (1, 5, 2, 1, 4)
LEVEL_KEYS = (("retired_cells",), METHODS, ("dataset_construction", "model_deployment"),
              ("incremental_cost",), ("labour", "equipment", "electricity", "carbon"))
HEIGHT_MM = 55
PADDING_MM = 2
NODE_WIDTH_MM = 4.8
NODE_GAP_MM = 4
FILENAME = "sankey_1_5_2_1_4.png"


def reference_width(output):
    manifest = json.loads((Path(output) / "panel_manifest.json").read_text(encoding="utf-8"))
    return sum(float(manifest["assets"][key]["size_mm"][0]) for key in ("b", "c"))


def build_layout(width_mm, height_mm=HEIGHT_MM):
    level_height = height_mm - 2 * PADDING_MM
    if (level_height <= (max(LEVEL_COUNTS) - 1) * NODE_GAP_MM
            or width_mm <= 2 * PADDING_MM + len(LEVEL_COUNTS) * NODE_WIDTH_MM):
        raise ValueError("Canvas is too small for the five-level Sankey")
    palette = ("#BC7F88", "#DCB391", "#A8BBA4", "#99BCCD", "#B2A3C3")
    colours = (("#929FA7",), palette, (palette[3], palette[0]), ("#929FA7",), palette[1:])
    positions = np.linspace(PADDING_MM, width_mm - PADDING_MM - NODE_WIDTH_MM, len(LEVEL_COUNTS))
    nodes = []
    for level, keys in enumerate(LEVEL_KEYS):
        count = len(keys)
        node_height = (level_height - NODE_GAP_MM * (count - 1)) / count
        for row, key in enumerate(keys):
            nodes.append(dict(id=len(nodes), key=key, level=level, row=row, value=0.,
                              x=float(positions[level]), y=PADDING_MM + row * (node_height + NODE_GAP_MM),
                              width=NODE_WIDTH_MM, height=node_height, color=colours[level][row]))
    by_key = {node["key"]: node for node in nodes}
    allocations = [("retired_cells", method, .2) for method in METHODS]
    for method in METHODS:
        if method != "rpt":
            allocations.append((method, "dataset_construction", .1))
        allocations.append((method, "model_deployment", .2 if method == "rpt" else .1))
    allocations.extend((("dataset_construction", "incremental_cost", .4),
                        ("model_deployment", "incremental_cost", .6)))
    allocations.extend(("incremental_cost", key, .25) for key in LEVEL_KEYS[-1])
    by_key["retired_cells"]["value"] = 1.
    for _, target, value in allocations:
        by_key[target]["value"] += value
    incoming = {node["id"]: node["y"] for node in nodes}
    outgoing = incoming.copy()
    links = []
    for source_key, target_key, value in allocations:
        source, target = by_key[source_key], by_key[target_key]
        colour_node = source if source["level"] in (1, 2) else target
        link = dict(source=source["id"], target=target["id"], value=value,
                    source_height=source["height"] * value / source["value"],
                    target_height=target["height"] * value / target["value"],
                    source_y=outgoing[source["id"]], target_y=incoming[target["id"]], color=colour_node["color"])
        links.append(link)
        outgoing[source["id"]] += link["source_height"]
        incoming[target["id"]] += link["target_height"]
    return nodes, links


def render_sankey(width_mm, height_mm=HEIGHT_MM):
    nodes, links = build_layout(width_mm, height_mm)
    with matplotlib.rc_context(STYLE):
        fig = plt.figure(figsize=np.array([width_mm, height_mm]) / 25.4, dpi=PNG_DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, width_mm), ylim=(height_mm, 0), xticks=[], yticks=[])
        ax.set_axis_off()
        for link in links:
            source, target = nodes[link["source"]], nodes[link["target"]]
            x0, x1 = source["x"] + source["width"], target["x"]
            y0, y1 = link["source_y"], link["target_y"]
            h0, h1 = link["source_height"], link["target_height"]
            control = (x1 - x0) * .46
            vertices = [(x0, y0), (x0 + control, y0), (x1 - control, y1), (x1, y1),
                        (x1, y1 + h1), (x1 - control, y1 + h1),
                        (x0 + control, y0 + h0), (x0, y0 + h0), (x0, y0)]
            codes = [BezierPath.MOVETO, BezierPath.CURVE4, BezierPath.CURVE4, BezierPath.CURVE4,
                     BezierPath.LINETO, BezierPath.CURVE4, BezierPath.CURVE4, BezierPath.CURVE4,
                     BezierPath.CLOSEPOLY]
            ax.add_patch(PathPatch(BezierPath(vertices, codes), facecolor=link["color"],
                                   edgecolor="none", alpha=.46, zorder=1,
                                   gid=f"flow:{source['key']}:{target['key']}"))
        for node in nodes:
            ax.add_patch(Rectangle((node["x"], node["y"]), node["width"], node["height"],
                                   facecolor=node["color"], edgecolor="none", zorder=2,
                                   gid=f"node:{node['level']}:{node['row']}"))
    return fig, nodes, links


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualization/outputs")
    args = parser.parse_args(argv)
    output = args.output_dir
    validate_destination(output, ROOT / "results/reference")
    reference = output / "panel_manifest.json"
    reference_hash = _hash(reference)
    width = reference_width(output)
    destination = output / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, nodes, links = render_sankey(width)
    try:
        fig.canvas.draw()
        assert not [text for text in fig.findobj(Text) if text.get_visible() and text.get_text().strip()]
        fig.savefig(destination, dpi=PNG_DPI, facecolor="white", bbox_inches=None)
    finally:
        plt.close(fig)
    verify_raster(destination, [width, HEIGHT_MM], PNG_DPI)
    assert _hash(reference) == reference_hash
    manifest = dict(schematic_only=True, quantitative_tea_data=False, visible_text_count=0,
                    node_counts=LEVEL_COUNTS, node_count=len(nodes), link_count=len(links),
                    method_order=METHODS, rpt_dataset_branch=False,
                    topology="RPT-based routes only to deployment; four learning methods connect to both stages",
                    warning="Illustrative weights and tapered ribbon widths are not quantitative TEA results.",
                    equal_level_outer_heights=True, level_height_mm=HEIGHT_MM - 2 * PADDING_MM,
                    node_width_mm=NODE_WIDTH_MM, size_mm=[width, HEIGHT_MM], dpi=PNG_DPI, export_format="png",
                    reference_panels=["b", "c"], reference_manifest_sha256=reference_hash,
                    nodes=nodes, links=links, png_sha256=_hash(destination), script_sha256=_hash(__file__),
                    passed=True, agent_visual_review_performed=False,
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "sankey_schematic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {FILENAME}: {width:.2f} x {HEIGHT_MM} mm; RPT connects only to deployment")


if __name__ == "__main__":
    main()
