"""Shared paths and exports for the LFP visualization notebooks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_ROOT = PROJECT_ROOT.parent
NOTEBOOKS = {
    "capacity": "Capacity distribution/plot.ipynb",
    "raw-curves": "Raw charge-discharge curves/plot.ipynb",
    "complete-curves": "Complete and segmented discharge curves/plot.ipynb",
    "segmented-curves": "Complete and segmented discharge curves/combined_2_plus_2_plot.ipynb",
    "tail-segments": "Tail+segment plot/plot.ipynb",
    "tail-soc": "Tail+upstream fragments/plot.ipynb",
    "tsne": "Input tsne distribution/plot.ipynb",
    "correlation": "Correlation analysis/plot.ipynb",
    "tail-ranges": "Voltage and SOC ranges of different tail settings/plot.ipynb",
}
_datasets = {}
_figures = {}
_tables = {}


def begin_category(category):
    _figures[category] = []
    _tables[category] = []
    _datasets.clear()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_reference(path, start):
    try:
        return os.path.relpath(path, start)
    except ValueError:  # Windows exports may be on another drive.
        return str(Path(path).resolve())


def validate_output(path):
    """Permit dedicated visualization exports, protect bundled data and code."""
    path = Path(path).expanduser().resolve()
    for allowed in (PROJECT_ROOT / "outputs", PROJECT_ROOT / "results"):
        if path == allowed or allowed in path.parents:
            return path
    protected = [MODULE_ROOT / name for name in
                 ("dataset", "model_construction", "outputs", "provenance", "tests")]
    protected.append(PROJECT_ROOT)
    if path in (MODULE_ROOT, MODULE_ROOT.parent) or any(
        path == item or item in path.parents or path in item.parents for item in protected
    ):
        raise ValueError("Choose dataset_visualization/outputs, results, or a fresh run directory.")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output must be a directory: {path}")
    return path


def get_dataset_path(filename):
    """Use the existing packaged MAT, without copying it into this folder."""
    if filename not in ("LFP_data_processed.mat", "CETL-data.mat"):
        raise ValueError(f"Unknown LFP data file: {filename}")
    explicit = os.environ.get("LFP_PROCESSED_MAT") if filename == "LFP_data_processed.mat" else None
    path = Path(explicit) if explicit else Path(os.environ.get("BATTERY_DATA_DIR", MODULE_ROOT / "dataset")) / filename
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    _datasets[filename] = {"path": path_reference(path, PROJECT_ROOT), "sha256": sha256(path)}
    return path


def export_root(kind):
    default = PROJECT_ROOT / ("outputs" if kind == "figures" else "results")
    variable = "LFP_FIGURE_OUTPUT" if kind == "figures" else "LFP_FIGURE_RESULTS"
    return validate_output(os.environ.get(variable, default))


def save_figure(fig, category, filename, **kwargs):
    """Preserve source dimensions and DPI; optionally export editable vectors."""
    target = export_root("figures") / category
    target.mkdir(parents=True, exist_ok=True)
    formats = os.environ.get("LFP_FIGURE_FORMATS", "png").split(",")
    if not formats or any(fmt not in ("png", "pdf", "svg") for fmt in formats):
        raise ValueError("Figure formats must be png, pdf, svg, or a comma-separated subset.")
    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    kwargs.setdefault("bbox_inches", "tight")
    for fmt in dict.fromkeys(formats):
        path = target / (Path(filename).stem + "." + fmt)
        fig.savefig(path, **kwargs)
        _figures.setdefault(category, []).append({
            "path": path_reference(path, export_root("results")),
            "sha256": sha256(path), "dpi": kwargs.get("dpi", fig.dpi),
            "figsize_inches": fig.get_size_inches().tolist(),
        })


def save_table(category, filename, **columns):
    path = export_root("results") / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(columns)
    frame.to_csv(path, index=False)
    _tables.setdefault(category, []).append({
        "path": path_reference(path, export_root("results")),
        "sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns),
    })
    return path


def finish_category(category):
    """Record the exact input data, notebook and exports from this execution."""
    import scipy
    import sklearn
    path = export_root("results") / category / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = PROJECT_ROOT / NOTEBOOKS[category]
    payload = {
        "dataset": "LFP dataset", "category": category,
        "notebook": NOTEBOOKS[category], "notebook_sha256": sha256(notebook),
        "datasets": _datasets, "figures": _figures.get(category, []),
        "tables": _tables.get(category, []),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__},
        "interpretation": "Descriptive dataset analysis; no trained model is used. See README for sampling and SOC definitions.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
