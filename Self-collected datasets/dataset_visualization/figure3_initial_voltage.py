"""Five-condition initial-voltage violin using auditable source cohorts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from figure3_panels import AXIS_LINEWIDTH, DPI, FONT_SIZE, _set_style, load_dataset
from scipy.io import loadmat


VIS_ROOT = Path(__file__).resolve().parent
LFP_MAT = VIS_ROOT.parents[1] / "LFP dataset/dataset/LFP_data_processed.mat"
LFP_SHA256 = "7c95928c620bf23c61456225fca8838a21951432b7ffe58ce75bd0d651b82d05"
FIGURE_SIZE_MM = (84.0, 52.8)
LFP_TAIL_UPPER_V = (2.73, 2.75, 2.77)
VIOLIN_FILLS = ("#745057", "#845A64", "#956A73", "#F6F0EF", "#F6F0EF")
VIOLIN_EDGE = "#60363D"
MEASURED_BACKGROUND = "#EBE1DF"
GROUP_LABEL_Y_MM = 9.5
GROUP_LABELS = ("LFP dataset", "Self-collected datasets")


@dataclass(frozen=True)
class InitialVoltageGroup:
    label: str
    kind: str
    tail_upper_v: float
    voltage: np.ndarray
    source_file: str
    source_sha256: str
    source_field: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simulate_initial_voltages(n: int, cutoff_v: float, *, seed: int,
                               mean_v: float, std_v: float) -> np.ndarray:
    """Reproduce the scalar rejection loop, restarting the RNG per cutoff."""
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=float)
    for index in range(n):
        while True:
            value = rng.normal(mean_v, std_v)
            if value > cutoff_v:
                values[index] = value
                break
    return values


def load_initial_voltage_groups() -> list[InitialVoltageGroup]:
    """Regenerate LFP simulations and use measured NMC partial-curve starts."""
    digest = _sha256(LFP_MAT)
    if digest != LFP_SHA256:
        raise ValueError("Figure 3j requires the packaged 3511-cell LFP cohort")
    n = loadmat(LFP_MAT)["processed_Qd_mat"].shape[1]
    groups = []
    for cutoff in LFP_TAIL_UPPER_V:
        values = _simulate_initial_voltages(n, cutoff, seed=999,
                                          mean_v=(2.7+3.0)/2, std_v=(3.0-2.7)/3)
        groups.append(InitialVoltageGroup(
            label=f"LFP {cutoff:.2f} V", kind="simulated", tail_upper_v=cutoff,
            voltage=values, source_file="LFP dataset/dataset/LFP_data_processed.mat",
            source_sha256=digest,
            source_field="scalar rejection sample, seed 999, Vinit > tail upper bound; no upper truncation"))
    for dataset in ("NMC_A", "NMC_B"):
        data = load_dataset(dataset)
        groups.append(InitialVoltageGroup(
            label=dataset, kind="measured", tail_upper_v=data.tail_high_v,
            voltage=data.vinit, source_file=f"Self-collected datasets/dataset/{data.source_path.name}",
            source_sha256=data.source_sha256,
            source_field="highest finite processed_Vd_partial_mat voltage per cell"))
    return groups


def render_violin(output_dir: Path, *, figure_dir: Path | None = None, formats=("png",)) -> dict:
    """Write one fixed-size PNG plus exact per-sample and quartile source data."""
    groups = load_initial_voltage_groups()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_style()
    width_mm, height_mm = FIGURE_SIZE_MM
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax = fig.add_axes([13.0 / width_mm, 17.0 / height_mm,
                       66.5 / width_mm, 31.8 / height_mm])
    ax.grid(False)
    ax.axvspan(3.5, 5.5, color=MEASURED_BACKGROUND, zorder=0)
    samples = [group.voltage for group in groups]
    positions = np.arange(1, 6)
    violins = ax.violinplot(samples, positions=positions, widths=0.52,
                            showmeans=False, showmedians=False, showextrema=False,
                            points=200, bw_method="scott")
    for body, color in zip(violins["bodies"], VIOLIN_FILLS):
        body.set_facecolor(color)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_linewidth(0.7)
        body.set_alpha(1.0)
    summaries = []
    for position, group in zip(positions, groups):
        q1, median, q3 = np.quantile(group.voltage, [0.25, 0.5, 0.75])
        mark_color = "white" if group.kind == "simulated" else VIOLIN_EDGE
        ax.hlines((q1, q3), position - 0.09, position + 0.09,
                  color=mark_color, linewidth=0.8, zorder=4)
        ax.hlines(median, position - 0.13, position + 0.13,
                  color=mark_color, linewidth=1.2, zorder=5)
        summaries.append({
            "label": group.label,
            "kind": group.kind,
            "tail_upper_V": group.tail_upper_v,
            "n": int(group.voltage.size),
            "minimum_V": float(np.min(group.voltage)),
            "q1_V": float(q1),
            "median_V": float(median),
            "q3_V": float(q3),
            "maximum_V": float(np.max(group.voltage)),
            "source_file": group.source_file,
            "source_sha256": group.source_sha256,
            "source_field": group.source_field,
        })
    ax.set_xticks(positions, ["2.73 V", "2.75 V", "2.77 V", "NMC_A", "NMC_B"])
    group_texts = [
        fig.text((13.0 + 66.5 * group_center) / width_mm,
                 GROUP_LABEL_Y_MM / height_mm, label, ha="center", va="center")
        for group_center, label in zip((0.3, 0.8), GROUP_LABELS)
    ]
    ax.set_ylabel("Initial voltage (V)")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(2.70, 3.92)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    group_label_to_xtick_clearance = float((
        min(label.get_window_extent(renderer).y0 for label in ax.get_xticklabels())
        - max(label.get_window_extent(renderer).y1 for label in group_texts)
    ) * 72 / fig.dpi)
    if group_label_to_xtick_clearance < 0:
        raise AssertionError("Violin group labels overlap x-axis tick labels")
    text_clearances = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        text_clearances.append((min(box.x0, fig.bbox.width - box.x1,
                                    box.y0, fig.bbox.height - box.y1) * 72 / fig.dpi,
                                artist.get_text()))
    minimum_text_clearance, nearest_text = min(text_clearances)
    minimum_text_clearance = float(minimum_text_clearance)
    if minimum_text_clearance < 0:
        raise AssertionError(f"Violin text {nearest_text!r} extends beyond fixed canvas: {minimum_text_clearance:.2f} pt")
    figure_dir = Path(figure_dir or VIS_ROOT / "outputs")
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_files = []
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    for fmt in formats:
        path = figure_dir / f"panel_j_initial_voltage.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches=None)
        figure_files.append({"file": path.name, "sha256": _sha256(path)})
    plt.close(fig)
    with (out / "source_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "kind", "tail_upper_V", "source_column_1based", "initial_voltage_V"])
        for group in groups:
            for column, voltage in enumerate(group.voltage, start=1):
                writer.writerow([group.label, group.kind, group.tail_upper_v,
                                 column, float(voltage)])
    with (out / "quartiles.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "kind", "tail_upper_V", "n", "minimum_V", "q1_V", "median_V", "q3_V", "maximum_V"])
        for item in summaries:
            writer.writerow([item[key] for key in
                             ("label", "kind", "tail_upper_V", "n", "minimum_V", "q1_V", "median_V", "q3_V", "maximum_V")])
    metadata = {
        "panel": "initial_voltage_violin_comparison",
        "figure3_panel": "j",
        "figure_files": figure_files,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "png_dpi": DPI,
        "font_family": "Arial",
        "font_size_pt": FONT_SIZE,
        "axis_linewidth_pt": AXIS_LINEWIDTH,
        "quartile_levels": [0.25, 0.5, 0.75],
        "quartile_marks": "Q1/median/Q3 short horizontal bars",
        "kde_bandwidth": "Scott",
        "violin_style": {
            "reference_group_background": "NMC_A/NMC_B",
            "background_color": MEASURED_BACKGROUND,
            "violin_fills": list(VIOLIN_FILLS),
            "violin_edge": VIOLIN_EDGE,
            "violin_width": 0.52,
            "quartile_marking": "three short horizontal lines",
        },
        "lfp_simulation": {
            "seed": 999,
            "parent_mean_V": (2.7 + 3.0) / 2,
            "parent_std_V": (3.0 - 2.7) / 3,
            "sampling": "separate scalar rejection loops; RNG restarted with the same seed for each tail bound",
            "acceptance": "Vinit > tail upper bound; no upper truncation",
            "cohort_mat": "LFP dataset/dataset/LFP_data_processed.mat",
            "cohort_sha256": LFP_SHA256,
        },
        "tight_crop": False,
        "group_labels": list(GROUP_LABELS),
        "group_label_y_mm": GROUP_LABEL_Y_MM,
        "group_label_to_xtick_clearance_pt": group_label_to_xtick_clearance,
        "minimum_text_canvas_clearance_pt": minimum_text_clearance,
        "datasets": summaries,
        "caveat": "Three LFP conditions reuse the same 3511 processed cells but independently simulate Vinit for each tail bound; NMC_A/NMC_B initial voltages are measured de-energisation starts.",
        "source_data_file": "source_data.csv",
        "quartiles_file": "quartiles.csv",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

