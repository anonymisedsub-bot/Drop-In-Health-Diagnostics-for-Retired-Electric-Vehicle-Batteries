"""Figure 3 NMC_A/NMC_B panels, using the exact model-training MAT cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "dataset"
REFERENCE_CAPACITY_AH = 2.5
DATASETS = {
    "NMC_A": {
        "source_batch": 4,
        "filename": "NMC_A_data_processed.mat",
        "sha256": "c87b8ca50809178597af222878b3c5e76692cc2f9e66a95d0d3daf89405e0e33",
        "n": 353,
        "tail_low_v": 3.00,
        "tail_high_v": 3.30,
    },
    "NMC_B": {
        "source_batch": 6,
        "filename": "NMC_B_data_processed.mat",
        "sha256": "5b0baca80cffb9a85bd48918bf1da81f1a1839fcff12e02cf846e5b40fb36549",
        "n": 334,
        "tail_low_v": 2.75,
        "tail_high_v": 3.20,
    },
}
PANEL_SIZES_MM = {
    "cell_number_vs_soh": (52.5, 50.0),
    "partial_discharge": (42.0, 52.8),
    "voltage_capacity_and_ic": (84.0, 52.8),
    "upstream_fragment_availability": (42.0, 52.8),
}
PANEL_LETTERS = {
    "NMC_A": {"cell_number_vs_soh": "d", "voltage_capacity_and_ic": "g", "partial_discharge": "k", "upstream_fragment_availability": "l"},
    "NMC_B": {"cell_number_vs_soh": "e", "voltage_capacity_and_ic": "h", "partial_discharge": "k", "upstream_fragment_availability": "l"},
}


def panel_stem(dataset, panel_key):
    return f"panel_{PANEL_LETTERS[dataset][panel_key]}_{dataset}"


MARINE_COLORS = ["#313695", "#74add1", "#e0f3f8", "#fee090", "#f46d43", "#a50026"]
VQ_IC_COLORS = ["#3B5B92", "#6FA8A3", "#D9C77E", "#D8905A", "#B85C5C"]
UPSTREAM_COLORS = ["#C97C7C", "#D89C6A", "#D6C27A", "#8FB18F", "#7FA6C7", "#A58AC0", "#C7968E"]
DPI = 600
FONT_SIZE = 9
BAR_VALUE_LABEL_FONT_SIZE = 7
CAPACITY_COLORBAR_LABEL = "Maximum capacity (Ah)"
AXIS_LINEWIDTH = 0.4
BAR_ALPHA = 0.6


@dataclass(frozen=True)
class DatasetData:
    key: str
    source_batch: int
    source_path: Path
    source_sha256: str
    tail_low_v: float
    tail_high_v: float
    grid: np.ndarray
    qmax: np.ndarray
    q_partial: np.ndarray
    v_partial: np.ndarray
    q_rpt: np.ndarray
    v_rpt: np.ndarray
    ic_rpt: np.ndarray
    vinit: np.ndarray

    @property
    def fragment_edges_v(self) -> np.ndarray:
        return self.tail_high_v + np.arange(7, dtype=float) * 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_prefix(mask: np.ndarray) -> bool:
    counts = mask.sum(axis=0)
    return bool(np.all(counts > 0) and np.array_equal(
        mask, np.arange(mask.shape[0])[:, None] < counts[None, :]
    ))


@lru_cache(maxsize=2)
def load_dataset(dataset: str) -> DatasetData:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    config = DATASETS[dataset]
    path = DATA_ROOT / config["filename"]
    source_hash = _sha256(path)
    if source_hash != config["sha256"]:
        raise ValueError(f"Unexpected input MAT SHA-256 for {dataset}: {source_hash}")
    mat = loadmat(path)
    keys = {
        "q_partial": "processed_Qd_partial_mat",
        "v_partial": "processed_Vd_partial_mat",
        "q_rpt": "processed_RPT_Q_full_mat",
        "v_rpt": "processed_RPT_V_full_mat",
        "ic_rpt": "processed_RPT_IC_full_mat",
    }
    arrays = {name: np.asarray(mat[key], dtype=float) for name, key in keys.items()}
    grid = np.asarray(mat["voltage_grid_mat"], dtype=float).reshape(-1)
    qmax = np.asarray(mat["processed_Qmax_mat"], dtype=float).reshape(-1)
    shape = (grid.size, int(config["n"]))
    if qmax.size != config["n"] or any(
        arrays[name].shape != shape for name in ("q_partial", "v_partial", "q_rpt", "v_rpt")
    ) or arrays["ic_rpt"].shape != (grid.size - 1, qmax.size):
        raise ValueError(f"Unexpected matrix dimensions in {path.name}")
    if not np.isfinite(grid).all() or not np.allclose(np.diff(grid), 0.005, atol=1e-10):
        raise ValueError("Voltage grid must be finite with 0.005-V spacing")
    if not np.isfinite(qmax).all() or np.any(qmax <= 0):
        raise ValueError("Qmax must contain positive finite values")
    for stage in ("partial", "rpt"):
        q = arrays[f"q_{stage}"]
        v = arrays[f"v_{stage}"]
        mask = np.isfinite(q)
        if not np.array_equal(mask, np.isfinite(v)) or not _finite_prefix(mask):
            raise ValueError(f"{stage} Q/V finite values must form matching prefixes")
        if not np.allclose(v[mask], np.broadcast_to(grid[:, None], shape)[mask], atol=1e-9):
            raise ValueError(f"{stage} voltage must match voltage_grid_mat")
    rpt_mask = np.isfinite(arrays["q_rpt"])
    if not np.allclose(arrays["q_rpt"][0], 0, atol=1e-9):
        raise ValueError("Updated RPT Q must be zero at the low-voltage endpoint")
    for column in range(qmax.size):
        q_observed = arrays["q_rpt"][rpt_mask[:, column], column]
        if np.any(np.diff(q_observed) < -1e-9):
            raise ValueError("Updated RPT Q must increase with voltage")
    expected_ic_mask = rpt_mask[:-1] & rpt_mask[1:]
    if not np.array_equal(np.isfinite(arrays["ic_rpt"]), expected_ic_mask):
        raise ValueError("RPT IC coverage must match adjacent RPT Q/V points")
    vinit = grid[np.isfinite(arrays["v_partial"]).sum(axis=0) - 1]
    if np.any(vinit < float(config["tail_high_v"]) - 1e-9):
        raise ValueError("Each measured partial curve must contain the full tail")
    return DatasetData(
        key=dataset,
        source_batch=int(config["source_batch"]),
        source_path=path,
        source_sha256=source_hash,
        tail_low_v=float(config["tail_low_v"]),
        tail_high_v=float(config["tail_high_v"]),
        grid=grid,
        qmax=qmax,
        q_partial=arrays["q_partial"],
        v_partial=arrays["v_partial"],
        q_rpt=arrays["q_rpt"],
        v_rpt=arrays["v_rpt"],
        ic_rpt=arrays["ic_rpt"],
        vinit=vinit,
    )


def soh_histogram(qmax: np.ndarray, *, method: str = "fd") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return capacity bins converted to SOH and their within-bin sample SD."""
    q = np.asarray(qmax, dtype=float).reshape(-1)
    if q.size == 0 or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("Qmax must be nonempty, positive and finite")
    if method == "fd":
        iqr = np.percentile(q, 75) - np.percentile(q, 25)
        width = 2 * iqr / np.cbrt(q.size) if iqr > 0 else (q.max() - q.min()) / 20
        if width <= 0:
            width = 0.1
        q_edges = np.arange(q.min(), q.max() + width, width)
    elif method == "scott":
        q_edges = np.histogram_bin_edges(q, bins="scott")
    else:
        raise ValueError(f"Unknown SOH histogram method: {method!r}")
    if q_edges.size < 5:
        q_edges = np.linspace(q.min(), q.max(), 10)
    counts, _ = np.histogram(q, bins=q_edges)
    soh = 100 * q / REFERENCE_CAPACITY_AH
    sd = np.full(counts.size, np.nan)
    for index, (left, right) in enumerate(zip(q_edges[:-1], q_edges[1:])):
        selected = (q >= left) & ((q < right) if index < counts.size - 1 else (q <= right))
        if selected.sum() >= 2:
            sd[index] = np.std(soh[selected], ddof=1)
    return 100 * q_edges / REFERENCE_CAPACITY_AH, counts, sd


def fragment_counts(vinit: np.ndarray, fragment_edges_v: np.ndarray, *,
                    exclusive: bool = False) -> np.ndarray:
    """Count nested cohorts, or cells with exactly 0..6 complete fragments."""
    starts = np.asarray(vinit, dtype=float).reshape(-1)
    if starts.size == 0 or not np.isfinite(starts).all():
        raise ValueError("Measured initial voltages must be nonempty and finite")
    edges = np.asarray(fragment_edges_v, dtype=float).reshape(-1)
    if edges.size != 7 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Six fragments require seven increasing finite voltage edges")
    upper = edges[1:]
    available = np.array([np.count_nonzero(starts >= edge - 1e-9) for edge in upper], dtype=int)
    none = starts.size - available[0]
    if exclusive:
        return np.r_[none, available[:-1] - available[1:], available[-1]]
    return np.r_[none, available]


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": AXIS_LINEWIDTH,
        "ytick.major.width": AXIS_LINEWIDTH,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": FONT_SIZE,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "path.simplify": False,
    })


def _canvas(panel_key: str, *, colorbar: bool = False):
    _set_style()
    width, height = PANEL_SIZES_MM[panel_key]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    if panel_key == "cell_number_vs_soh":
        left, right, bottom, top = 12.5, 15.5, 9.5, 6.5
    elif colorbar:
        left, right, bottom, top = 10.8, 13.5, 11.0, 3.5
    elif panel_key == "upstream_fragment_availability":
        left, right, bottom, top = 11.5, 2.5, 13.0, 3.5
    else:
        left, right, bottom, top = 10.5, 2.5, 11.0, 3.5
    ax = fig.add_axes([
        left / width,
        bottom / height,
        (width - left - right) / width,
        (height - bottom - top) / height,
    ])
    ax.grid(False)
    return fig, ax


def _paired_rpt_canvas():
    """Match the reference's 84-mm left/right RPT layout before adding one bar."""
    _set_style()
    width, height = PANEL_SIZES_MM["voltage_capacity_and_ic"]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    axes = [
        fig.add_axes([left / width, 11.0 / height, 22.5 / width, (height - 15.0) / height])
        for left in (11.0, 47.0)
    ]
    for ax in axes:
        ax.grid(False)
    return fig, axes


def _capacity_colors(qmax: np.ndarray, palette: list[str]):
    cmap = LinearSegmentedColormap.from_list("rebatt_capacity", palette, N=256)
    norm = Normalize(float(np.min(qmax)), float(np.max(qmax)))
    return cmap(norm(qmax)), cmap, norm


def _capacity_colorbar(fig, ax, cmap, norm):
    width_mm, _ = fig.get_size_inches() * 25.4
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    bounds = [data_ax.get_position() for data_ax in axes]
    right = max(box.x1 for box in bounds)
    bottom = min(box.y0 for box in bounds)
    top = max(box.y1 for box in bounds)
    bar_height = (top - bottom) * 0.5
    cax = fig.add_axes([
        right + (1.5 if len(axes) > 1 else 1.0) / width_mm,
        (bottom + top - bar_height) / 2,
        2.8 / width_mm,
        bar_height,
    ])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.outline.set_visible(False)
    colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 3))
    colorbar.ax.set_yticklabels([f"{value:.1f}" for value in np.linspace(norm.vmin, norm.vmax, 3)])
    colorbar.ax.tick_params(pad=1, length=2, width=AXIS_LINEWIDTH)
    colorbar.set_label(CAPACITY_COLORBAR_LABEL, labelpad=1)
    return colorbar


def _add_curves(ax, x_matrix: np.ndarray, y_matrix: np.ndarray,
                colors: np.ndarray, *, alpha: float, linewidth: float = 0.35):
    if x_matrix.shape != y_matrix.shape or x_matrix.shape[1] != len(colors):
        raise ValueError("Curve arrays and color count are inconsistent")
    segments = []
    for index in range(x_matrix.shape[1]):
        valid = np.isfinite(x_matrix[:, index]) & np.isfinite(y_matrix[:, index])
        segments.append(np.column_stack((x_matrix[valid, index], y_matrix[valid, index])))
    collection = LineCollection(segments, colors=colors, alpha=alpha, linewidths=linewidth)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _write_csv(path: Path, columns: dict[str, np.ndarray | list]) -> None:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("CSV columns must have equal lengths")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def _save(fig, out_dir: Path, panel_key: str, data: DatasetData,
          source_data_file: str, extra: dict, *, figure_dir: Path | None = None,
          formats=("png",)) -> dict:
    export_key = panel_key
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_clearances = []
    off_axis_tick_labels = set()
    for axis in fig.axes:
        for coordinate_axis, (lower, upper) in (
            (axis.xaxis, sorted(axis.get_xlim())),
            (axis.yaxis, sorted(axis.get_ylim())),
        ):
            for tick in coordinate_axis.get_major_ticks():
                if tick.get_loc() < lower - 1e-9 or tick.get_loc() > upper + 1e-9:
                    off_axis_tick_labels.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(match=matplotlib.text.Text):
        if (not artist.get_visible() or not artist.get_text().strip()
                or id(artist) in off_axis_tick_labels):
            continue
        box = artist.get_window_extent(renderer)
        clearance = min(box.x0, fig.bbox.width - box.x1,
                        box.y0, fig.bbox.height - box.y1) * 72 / fig.dpi
        text_clearances.append((clearance, artist.get_text()))
    minimum_text_canvas_clearance_pt, nearest_text = min(text_clearances)
    assert minimum_text_canvas_clearance_pt >= 0, (panel_key, nearest_text,
                                                    minimum_text_canvas_clearance_pt)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(figure_dir or ROOT / "outputs")
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = panel_stem(data.key, panel_key)
    figure_files = []
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    for fmt in formats:
        path = figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches=None)
        figure_files.append({"file": path.name, "sha256": _sha256(path)})
    metadata = {
        "panel": export_key,
        "dataset": data.key,
        "figure3_panel": PANEL_LETTERS[data.key][panel_key],
        "figure_files": figure_files,
        "source_batch": data.source_batch,
        "n_cells": int(data.qmax.size),
        "input_file": data.source_path.name,
        "input_sha256": data.source_sha256,
        "width_mm": PANEL_SIZES_MM[panel_key][0],
        "height_mm": PANEL_SIZES_MM[panel_key][1],
        "png_dpi": DPI,
        "font_family": "Arial",
        "font_size_pt": FONT_SIZE,
        "minimum_text_canvas_clearance_pt": minimum_text_canvas_clearance_pt,
        "nearest_text_to_canvas": nearest_text,
        "axis_linewidth_pt": AXIS_LINEWIDTH,
        "tight_crop": False,
        "source_data_file": source_data_file,
        **extra,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(fig)
    return metadata


def render_panel(dataset: str, panel_key: str, out_dir: Path, *,
                 figure_dir: Path | None = None, formats=("png",)) -> dict:
    """Render one Figure 3 panel; numeric data and images have separate roots."""
    if panel_key not in PANEL_SIZES_MM:
        raise ValueError(f"Unknown panel key: {panel_key!r}")
    data = load_dataset(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cell_index = np.arange(1, data.qmax.size + 1)

    if panel_key == "cell_number_vs_soh":
        binning_rule = "scott" if dataset == "NMC_B" else "fd"
        edges, counts, sd = soh_histogram(data.qmax, method=binning_rule)
        centers = (edges[:-1] + edges[1:]) / 2
        fig, ax = _canvas(panel_key)
        ax.bar(centers, counts, width=np.diff(edges), align="center",
               color="#a6cee3", edgecolor="black", linewidth=AXIS_LINEWIDTH,
               alpha=BAR_ALPHA)
        ax.set_xlabel("SOH (%)")
        ax.set_ylabel("Cell number")
        pad = max(1.0, (edges[-1] - edges[0]) * 0.02)
        ax.set_xlim(edges[0] - pad, edges[-1] + pad)
        ax.set_ylim(0, max(counts) * 1.22)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        panel_height_mm = PANEL_SIZES_MM[panel_key][1]
        legend_bottom = (panel_height_mm - 5.3) / panel_height_mm
        sample_legend = ax.legend(
            [Line2D([], [], color="none")], [f"$n$={data.qmax.size}"],
            loc="lower left", bbox_to_anchor=(10.0 / 52.5, legend_bottom),
            bbox_transform=fig.transFigure,
            handlelength=0, handletextpad=0, borderaxespad=0, borderpad=0,
        )
        right = ax.twinx()
        right.plot(centers, sd, color="black", marker="o", linewidth=0.85,
                   markersize=2.0, label="Intra-bin SD")
        right.fill_between(centers, sd, step="mid", color="grey", alpha=0.15)
        right.set_ylabel("Intra-bin SD (%)", labelpad=5)
        right.set_ylim(0, np.nanmax(sd) * 1.35 if np.isfinite(sd).any() else 1)
        right.yaxis.set_major_locator(MaxNLocator(3))
        right.spines["right"].set_visible(True)
        right.spines["left"].set_visible(False)
        sd_legend = right.legend(
            loc="lower right", bbox_to_anchor=(43.5 / 52.5, legend_bottom),
            bbox_transform=fig.transFigure,
            handlelength=1.0, borderaxespad=0, borderpad=0,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_boxes = [legend.get_window_extent(renderer)
                        for legend in (sample_legend, sd_legend)]
        legend_bottom_delta_pt = (
            abs(legend_boxes[0].y0 - legend_boxes[1].y0) * 72 / fig.dpi
        )
        assert legend_bottom_delta_pt < 0.01
        legend_horizontal_clearance_pt = (
            legend_boxes[1].x0 - legend_boxes[0].x1
        ) * 72 / fig.dpi
        assert legend_horizontal_clearance_pt >= 1
        legend_to_axes_clearance_pt = (
            min(box.y0 for box in legend_boxes) - ax.bbox.y1
        ) * 72 / fig.dpi
        assert legend_to_axes_clearance_pt >= 1
        minimum_legend_canvas_clearance_pt = min(
            *(box.x0 for box in legend_boxes),
            *(fig.bbox.width - box.x1 for box in legend_boxes),
            *(box.y0 for box in legend_boxes),
            *(fig.bbox.height - box.y1 for box in legend_boxes),
        ) * 72 / fig.dpi
        assert minimum_legend_canvas_clearance_pt >= 0
        minimum_y_label_clearance_pt = min(
            ax.yaxis.label.get_window_extent(renderer).x0,
            fig.bbox.width - right.yaxis.label.get_window_extent(renderer).x1,
        ) * 72 / fig.dpi
        assert minimum_y_label_clearance_pt >= 0
        _write_csv(out / "source_data.csv", {
            "mat_column_1based": cell_index,
            "maximum_capacity_Ah": data.qmax,
            "SOH_percent": 100 * data.qmax / REFERENCE_CAPACITY_AH,
        })
        _write_csv(out / "histogram.csv", {
            "left_SOH_percent": edges[:-1],
            "right_SOH_percent": edges[1:],
            "cell_number": counts,
            "intra_bin_SD_percentage_points": sd,
        })
        return _save(fig, out, panel_key, data, "source_data.csv", {
            "reference_capacity_Ah": REFERENCE_CAPACITY_AH,
            "dual_y_axes": True,
            "right_axis": "within-bin SOH sample SD (percentage points, ddof=1)",
            "histogram_count_sum": int(counts.sum()),
            "binning_rule": binning_rule,
            "histogram_bin_count": int(counts.size),
            "discharge_stage": "RPT capacity label",
            "legend_alignment": "bottom",
            "legend_bottom_figure_fraction": legend_bottom,
            "legend_bottom_delta_pt": legend_bottom_delta_pt,
            "legend_horizontal_clearance_pt": legend_horizontal_clearance_pt,
            "legend_to_axes_clearance_pt": legend_to_axes_clearance_pt,
            "minimum_legend_canvas_clearance_pt": minimum_legend_canvas_clearance_pt,
            "left_y_labelpad_pt": ax.yaxis.labelpad,
            "right_y_labelpad_pt": right.yaxis.labelpad,
            "minimum_y_label_clearance_pt": minimum_y_label_clearance_pt,
        }, figure_dir=figure_dir, formats=formats)

    if panel_key == "voltage_capacity_and_ic":
        colors, cmap, norm = _capacity_colors(data.qmax, VQ_IC_COLORS)
        fig, (ax_vq, ax_ic) = _paired_rpt_canvas()
        vq_lines = _add_curves(ax_vq, data.q_rpt, data.v_rpt, colors, alpha=0.55)
        voltage_mid = (data.v_rpt[:-1] + data.v_rpt[1:]) / 2
        ic_lines = _add_curves(ax_ic, voltage_mid, data.ic_rpt, colors,
                               alpha=0.55, linewidth=0.3)
        ax_vq.set_xlabel("Remaining capacity (Ah)")
        ax_vq.set_ylabel("Voltage (V)")
        ax_vq.set_xlim(-0.03, float(np.nanmax(data.q_rpt)) * 1.04)
        ax_vq.set_ylim(data.grid[0] - 0.02, float(np.nanmax(data.v_rpt)) + 0.02)
        ax_ic.set_xlabel("Voltage (V)")
        ax_ic.set_ylabel(r"dQ/dV (Ah V$^{-1}$)")
        ax_ic.set_xlim(data.grid[0] - 0.02, float(np.nanmax(voltage_mid)) + 0.02)
        ic_lower = float(np.nanmin(data.ic_rpt))
        ic_upper = float(np.nanmax(data.ic_rpt))
        ax_ic.set_ylim(min(0, ic_lower * 1.1), ic_upper * 1.05)
        for ax in (ax_vq, ax_ic):
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
        colorbar = _capacity_colorbar(fig, [ax_vq, ax_ic], cmap, norm)
        if len(fig.axes) != 3 or colorbar.norm is not norm:
            raise AssertionError("Paired RPT figure must have two data axes and one shared colorbar")
        shared_colors = bool(np.array_equal(vq_lines.get_colors(), ic_lines.get_colors()))
        if not shared_colors:
            raise AssertionError("V–Q and IC curves must have identical per-cell colors")
        bar_bounds = colorbar.ax.get_position()
        data_bounds = [ax_vq.get_position(), ax_ic.get_position()]
        data_height = max(box.y1 for box in data_bounds) - min(box.y0 for box in data_bounds)
        np.savez_compressed(out / "source_data.npz", mat_column_1based=cell_index,
                            rpt_remaining_capacity_Ah=data.q_rpt,
                            rpt_voltage_V=data.v_rpt,
                            ic_voltage_V=voltage_mid,
                            incremental_capacity_Ah_per_V=data.ic_rpt,
                            maximum_capacity_Ah=data.qmax)
        return _save(fig, out, panel_key, data, "source_data.npz", {
            "discharge_stage": "RPT (~0.5C)",
            "layout": "V–Q left, corresponding IC right",
            "data_panel_count": 2,
            "axes_count": len(fig.axes),
            "shared_colorbar_count": 1,
            "shared_curve_colors_verified": shared_colors,
            "colorbar_width_mm": float(bar_bounds.width * fig.get_figwidth() * 25.4),
            "colorbar_height_ratio": float(bar_bounds.height / data_height),
            "matrix_fields": ["processed_RPT_Q_full_mat", "processed_RPT_V_full_mat",
                              "processed_RPT_IC_full_mat"],
            "capacity_color_limits_Ah": [norm.vmin, norm.vmax],
            "colorbar_label": colorbar.ax.yaxis.label.get_text(),
            "colormap_anchors": VQ_IC_COLORS,
            "curve_alpha": 0.55,
            "additional_smoothing": False,
        }, figure_dir=figure_dir, formats=formats)

    if panel_key == "partial_discharge":
        colors, cmap, norm = _capacity_colors(data.qmax, MARINE_COLORS)
        fig, ax = _canvas(panel_key, colorbar=True)
        _add_curves(ax, data.v_partial, data.q_partial, colors, alpha=0.9)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Remaining capacity (Ah)")
        ax.set_xlim(data.grid[0] - 0.02, float(np.max(data.vinit)) + 0.02)
        ax.set_ylim(-0.03, float(np.nanmax(data.q_partial)) * 1.04)
        np.savez_compressed(out / "source_data.npz", mat_column_1based=cell_index,
                            voltage_V=data.v_partial,
                            remaining_capacity_Ah=data.q_partial,
                            measured_initial_voltage_V=data.vinit,
                            maximum_capacity_Ah=data.qmax)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        colorbar = _capacity_colorbar(fig, ax, cmap, norm)
        return _save(fig, out, panel_key, data, "source_data.npz", {
            "discharge_stage": "partial (~1C)",
            "matrix_fields": ["processed_Vd_partial_mat", "processed_Qd_partial_mat"],
            "capacity_color_limits_Ah": [norm.vmin, norm.vmax],
            "colorbar_label": colorbar.ax.yaxis.label.get_text(),
            "colormap_anchors": MARINE_COLORS,
            "curve_alpha": 0.9,
            "additional_smoothing": False,
        }, figure_dir=figure_dir, formats=formats)

    fragment_edges = data.fragment_edges_v
    counts = fragment_counts(data.vinit, fragment_edges)
    fig, ax = _canvas(panel_key)
    positions = np.arange(7)
    bars = ax.bar(positions, counts, width=0.72, color=UPSTREAM_COLORS,
                  alpha=BAR_ALPHA, linewidth=0)
    bar_labels = ax.bar_label(bars, labels=[str(int(value)) for value in counts],
                              padding=2, fontsize=BAR_VALUE_LABEL_FONT_SIZE,
                              rotation=90)
    ax.set_xlabel("Upstream fragment\nindex")
    ax.set_ylabel("Number of available cells")
    ax.set_xticks(positions, ["None", "1", "2", "3", "4", "5", "6"])
    ax.get_xticklabels()[0].set_ha("right")
    ax.set_xlim(-0.65, 6.65)
    ax.set_ylim(0, int(np.max(counts)) * 1.13)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
    minimum_x_tick_label_gap_pt = min(
        right.x0 - left.x1 for left, right in zip(tick_boxes, tick_boxes[1:])
    ) * 72 / fig.dpi
    assert minimum_x_tick_label_gap_pt >= 1
    label_boxes = [label.get_window_extent(renderer) for label in bar_labels]
    minimum_bar_label_canvas_clearance_pt = min(
        *(box.x0 for box in label_boxes),
        *(fig.bbox.width - box.x1 for box in label_boxes),
        *(box.y0 for box in label_boxes),
        *(fig.bbox.height - box.y1 for box in label_boxes),
    ) * 72 / fig.dpi
    bar_label_overlap_count = int(sum(
        label_boxes[i].overlaps(label_boxes[j])
        for i in range(len(label_boxes)) for j in range(i + 1, len(label_boxes))
    ))
    assert minimum_bar_label_canvas_clearance_pt >= 0
    assert bar_label_overlap_count == 0
    cell_columns = {
        "mat_column_1based": cell_index,
        "measured_initial_voltage_V": data.vinit,
    }
    _write_csv(out / "source_data.csv", {
        "fragment_index": ["0", "1", "2", "3", "4", "5", "6"],
        "lower_boundary_V": [np.nan, *fragment_edges[:-1]],
        "upper_boundary_V": [fragment_edges[1], *fragment_edges[1:]],
        "available_cells": counts,
    })
    count_definition = f"0 (none): Vinit < {fragment_edges[1]:.2f} V; k=1..6: Vinit >= upper endpoint of fragment k"
    _write_csv(out / "measured_initial_voltages.csv", cell_columns)
    return _save(fig, out, panel_key, data, "source_data.csv", {
        "discharge_stage": "partial (~1C)",
        "tail_region_V": [data.tail_low_v, data.tail_high_v],
        "upstream_edges_V": fragment_edges.tolist(),
        "initial_voltage_removal_V": 0.0,
        "count_definition": count_definition,
        "count_mode": "cumulative_available_fragment",
        "voltage_comparison_tolerance_V": 1e-9,
        "zero_tick_label": ax.get_xticklabels()[0].get_text(),
        "zero_tick_alignment": ax.get_xticklabels()[0].get_ha(),
        "x_tick_labels": [label.get_text() for label in ax.get_xticklabels()],
        "minimum_x_tick_label_gap_pt": minimum_x_tick_label_gap_pt,
        "available_counts": counts.tolist(),
        "bar_value_labels": [label.get_text() for label in bar_labels],
        "bar_value_label_count": len(bar_labels),
        "bar_value_label_font_size_pt": BAR_VALUE_LABEL_FONT_SIZE,
        "bar_value_label_rotation_deg": bar_labels[0].get_rotation(),
        "minimum_bar_label_canvas_clearance_pt": minimum_bar_label_canvas_clearance_pt,
        "bar_label_overlap_count": bar_label_overlap_count,
    }, figure_dir=figure_dir, formats=formats)

