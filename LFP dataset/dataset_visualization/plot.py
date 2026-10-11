"""Run the original LFP visualization notebooks, individually or as a batch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from notebook_setup import NOTEBOOKS, PROJECT_ROOT, validate_output


def run_all(output=None, results=None, only=None, formats="png", data=None, threads=2):
    figures = validate_output(output or PROJECT_ROOT / "outputs")
    numbers = validate_output(results or (Path(output) / "results" if output else PROJECT_ROOT / "results"))
    selected = only or list(NOTEBOOKS)
    for category in selected:
        if category not in NOTEBOOKS:
            raise ValueError(f"Unknown figure category: {category}")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not formats or any(fmt not in ("png", "pdf", "svg") for fmt in formats.split(",")):
        raise ValueError("Figure formats must be png, pdf, svg, or a comma-separated subset.")
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "LFP_FIGURE_OUTPUT": str(figures), "LFP_FIGURE_RESULTS": str(numbers),
                "LFP_FIGURE_FORMATS": formats,
                "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads)})
    if data:
        env["LFP_PROCESSED_MAT"] = str(Path(data).resolve())
    started = time.perf_counter()
    for category in selected:
        print(f"[LFP figures] {category}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", category,
                        "--threads", str(threads)], cwd=PROJECT_ROOT, env=env, check=True)
    report = {"categories": selected, "figures": str(figures), "results": str(numbers),
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps(report, indent=2), flush=True)
    return report


def execute_notebook(category, threads):
    # Source notebooks are plain Python. Executing their cells in one namespace
    # avoids requiring Jupyter for the command-line workflow.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
    notebook = PROJECT_ROOT / NOTEBOOKS[category]
    document = json.loads(notebook.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "__file__": str(notebook)}
    with threadpool_limits(limits=threads):
        try:
            for index, cell in enumerate(document["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"{notebook}:cell{index}", "exec"), scope)
        finally:
            plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(NOTEBOOKS))
    parser.add_argument("--output", type=Path, help="Figure directory; default: dataset_visualization/outputs")
    parser.add_argument("--results", type=Path, help="Numeric source directory; default: dataset_visualization/results")
    parser.add_argument("--data", type=Path, help="Alternative processed MAT; raw curves still use BATTERY_DATA_DIR or packaged raw MAT")
    parser.add_argument("--formats", default="png", help="png (default), or comma-separated png,pdf,svg")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", choices=list(NOTEBOOKS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        for name, path in NOTEBOOKS.items():
            print(f"{name}: {path}")
        return
    if args.worker:
        execute_notebook(args.worker, args.threads)
    else:
        run_all(args.output, args.results, args.only, args.formats, args.data, args.threads)


if __name__ == "__main__":
    main()
