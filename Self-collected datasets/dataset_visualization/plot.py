"""Generate Figure 3 d/e/g/h/j and the NMC_A/NMC_B parts of k/l."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure3_panels import DATASETS, PANEL_SIZES_MM, _sha256, load_dataset, panel_stem, render_panel
from figure3_initial_voltage import render_violin

ROOT = Path(__file__).resolve().parent
MODULE_ROOT = ROOT.parent
GROUPS = {
    "soh": "cell_number_vs_soh",
    "rpt": "voltage_capacity_and_ic",
    "partial": "partial_discharge",
    "availability": "upstream_fragment_availability",
    "initial-voltage": None,
}


def validate_output(path):
    path = Path(path).expanduser().resolve()
    for allowed in (ROOT / "outputs", ROOT / "results"):
        if path == allowed or allowed in path.parents:
            return path
    protected = [ROOT, *(MODULE_ROOT / name for name in
                 ("dataset", "model_construction", "provenance", "tests", "preprocessing"))]
    if path in (MODULE_ROOT, MODULE_ROOT.parent) or any(
        path == item or item in path.parents or path in item.parents for item in protected
    ):
        raise ValueError("Choose dataset_visualization/outputs, results, or a fresh run directory.")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output must be a directory: {path}")
    return path


def path_reference(path, start):
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(Path(path).resolve())


def cell_identity(dataset, results):
    """Tie every plotted MAT column to its public raw workbook and training cohort."""
    data = load_dataset(dataset)
    mapping_path = MODULE_ROOT / "dataset/raw/source_mapping.csv"
    mapping = pd.read_csv(mapping_path)
    rows = mapping[mapping.dataset == dataset].copy()
    # Column names follow the public extraction manifest, not source workbook IDs.
    id_column = "new_id"
    if id_column not in rows:
        raise ValueError(f"Expected {id_column} in raw/source_mapping.csv")
    rows = rows.sort_values(id_column)
    if rows[id_column].tolist() != list(range(1, len(data.qmax) + 1)):
        raise ValueError(f"Raw workbook mapping does not cover {dataset} in MAT-column order")
    if rows["mat_column_1based"].tolist() != rows[id_column].tolist():
        raise ValueError(f"Raw workbook IDs differ from MAT columns for {dataset}")
    if set(rows["processed_mat_sha256"]) != {data.source_sha256}:
        raise ValueError(f"Raw workbook mapping refers to a different MAT for {dataset}")
    rows["maximum_capacity_Ah"] = data.qmax
    rows["measured_initial_voltage_V"] = data.vinit
    rows.to_csv(results / f"{dataset}_cell_identity.csv", index=False)
    return {"dataset": dataset, "n_cells": len(rows), "input_sha256": data.source_sha256,
            "mapping_sha256": _sha256(mapping_path), "tail_region_V": [data.tail_low_v, data.tail_high_v],
            "upstream_edges_V": data.fragment_edges_v.tolist()}


def overview(figures):
    """A labelled inspection sheet assembled from the generated panel PNGs."""
    fig = plt.figure(figsize=(9, 14), facecolor="white")
    grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.2], hspace=0.28, wspace=0.12)
    for row, letters in enumerate((("d", "e"), ("g", "h"), ("k", "k"), ("l", "l"))):
        for col, (dataset, letter) in enumerate(zip(DATASETS, letters)):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(plt.imread(figures / f"panel_{letter}_{dataset}.png"))
            ax.set_title(f"{letter}  |  {dataset}", fontsize=11, loc="left")
            ax.axis("off")
    ax = fig.add_subplot(grid[4, :])
    ax.imshow(plt.imread(figures / "panel_j_initial_voltage.png"))
    ax.set_title("j  |  Initial voltage comparison", fontsize=11, loc="left")
    ax.axis("off")
    fig.savefig(figures / "Figure_3_NMC_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_all(output=None, results=None, datasets=None, only=None, formats=("png",)):
    figures = validate_output(output or ROOT / "outputs")
    numbers = validate_output(results or (Path(output) / "results" if output else ROOT / "results"))
    datasets = list(DATASETS) if datasets is None else list(datasets)
    groups = list(GROUPS) if only is None else list(only)
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError("Dataset must be NMC_A, NMC_B, or both")
    if not groups or any(name not in GROUPS for name in groups):
        raise ValueError(f"Choose figure groups from {list(GROUPS)}")
    if not formats or any(fmt not in ("png", "pdf", "svg") for fmt in formats):
        raise ValueError("Formats must be png, pdf, svg, or a comma-separated subset")
    figures.mkdir(parents=True, exist_ok=True)
    numbers.mkdir(parents=True, exist_ok=True)
    # j is a cross-dataset panel and always includes both NMC cohorts plus LFP.
    identity_datasets = list(DATASETS) if "initial-voltage" in groups else datasets
    identities = [cell_identity(name, numbers) for name in identity_datasets]
    panels = []
    for dataset in datasets:
        for group in groups:
            key = GROUPS[group]
            if key is None:
                continue
            stem = panel_stem(dataset, key)
            print(f"[Figure 3] {stem}", flush=True)
            meta = render_panel(dataset, key, numbers / stem, figure_dir=figures, formats=formats)
            panels.append({"stem": stem, "dataset": dataset, "panel": meta["figure3_panel"],
                           "metadata": f"{stem}/metadata.json", "figure_files": meta["figure_files"]})
    if "initial-voltage" in groups:
        print("[Figure 3] panel_j_initial_voltage", flush=True)
        meta = render_violin(numbers / "panel_j_initial_voltage", figure_dir=figures, formats=formats)
        panels.append({"stem": "panel_j_initial_voltage", "dataset": "LFP + NMC_A + NMC_B", "panel": "j",
                       "metadata": "panel_j_initial_voltage/metadata.json", "figure_files": meta["figure_files"]})
    has_overview = set(datasets) == set(DATASETS) and set(groups) == set(GROUPS) and "png" in formats
    if has_overview:
        overview(figures)
    manifest = {
        "figure": 3, "scope": "d/e/g/h/j and NMC parts of k/l", "panels": panels,
        "figures_directory": path_reference(figures, numbers), "datasets": identities,
        "figure_count": len(panels), "overview": "Figure_3_NMC_overview.png" if has_overview else None,
        "source_code": {p.name: _sha256(p) for p in ROOT.glob("*.py")},
        "source_data": [{"file": path_reference(p, numbers), "sha256": _sha256(p)}
                        for p in sorted(numbers.rglob("*")) if p.is_file() and p.name != "panel_manifest.json"],
        "interpretation": "Descriptive data panels. Initial NMC voltages are measured; LFP voltages are simulated. Figure 3l uses cumulative fragment availability.",
    }
    (numbers / "panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"panels": len(panels), "figures": str(figures), "results": str(numbers)}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--only", nargs="+", choices=list(GROUPS))
    parser.add_argument("--output", type=Path, help="Figure output; default: dataset_visualization/outputs")
    parser.add_argument("--results", type=Path, help="Numeric output; default: dataset_visualization/results")
    parser.add_argument("--formats", default="png", help="png (default), or png,pdf,svg")
    args = parser.parse_args()
    run_all(args.output, args.results, None if args.dataset == "both" else [args.dataset],
            args.only, tuple(args.formats.split(",")))


if __name__ == "__main__":
    main()
