"""Targeted-RPT TEA figure primitives; read analysis inputs, write figures only."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
FIGURE_SOURCE = ROOT / "results/reference/figure_source"
DATA_FILES = ("summary.csv", "method_summary.csv", "sensitivity.csv", "two_factor.csv", "run_manifest.json")
CACHE_FILES = ("cost_density.csv", "cost_density_provenance.json", "figure_caption.md")
ORIGINAL_PARAMETERS = (
    "cell_mass_g", "retention", "initial_soc", "label_fraction", "charge_efficiency",
    "electricity_price", "labour_wage", "testers_per_operator", "tester_purchase_price",
    "installation_fraction", "equipment_life_years", "annual_utilisation",
    "equipment_overhead_fraction", "grid_intensity", "carbon_price",
)
DENSITY_COLUMNS = ("method", "log_cost", "density", "count", "log_bin_width")
DENSITY_SOURCE_FILES = ("draws.csv.gz", "method_summary.csv", "run_manifest.json")
METHODS = ("proposed", "pulse", "eis", "partial_charge", "rpt")
COST_METHODS = ("proposed_no_targeted_rpt", *METHODS)
BASELINE_SUMMARY = ROOT / "results/baseline/method_summary.csv"
COMPARATORS = METHODS[1:]
METHOD_NAMES = {"pulse": "Pulse-based", "eis": "EIS-based",
                "partial_charge": "Partial-charge-based", "rpt": "RPT-based"}
LABELS = {"proposed_no_targeted_rpt": "Proposed (no targeted RPTs)",
          "proposed": "Proposed+targeted RPTs", **METHOD_NAMES}
# Method styles stay fixed when the display order changes.
COLORS = {"proposed": "#BC7F88", "partial_charge": "#DCB391", "rpt": "#A8BBA4",
          "pulse": "#99BCCD", "eis": "#B2A3C3"}
EDGES = {"proposed": "#78434D", "partial_charge": "#9A6945", "rpt": "#536F58",
         "pulse": "#3E6E87", "eis": "#75618C"}
COST_COLORS = {**COLORS, "proposed_no_targeted_rpt": "#E4CAD0"}
COST_EDGES = {**EDGES, "proposed_no_targeted_rpt": "#AD8C94"}
MARKERS = {"proposed": "o", "partial_charge": "s", "rpt": "D", "pulse": "^", "eis": "v"}
METRICS = {"cost_USD": "USD/tonne", "energy_kWh": "kWh/tonne", "co2_kg": "kg-CO2/tonne"}
SAVINGS_LABELS = {
    "cost_USD": "Cost savings\n(USD/tonne)",
    "energy_kWh": "Electricity savings\n(kWh/tonne)",
    "co2_kg": "CO\u2082 emissions savings\n(kg-CO\u2082/tonne)",
}
FONT_PT = 10.5
RESPONSE_BAR_WIDTH_MM = 3
STYLE = {
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": 11,
    "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
    "text.usetex": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "axes.grid": False, "legend.frameon": False,
    "text.color": "#171717", "axes.labelcolor": "#171717",
    "xtick.color": "#171717", "ytick.color": "#171717",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": None,
}
ROSE = LinearSegmentedColormap.from_list("rose", ("#FAF5F5", "#DBB6BD", "#7B4453"))
CMAP = matplotlib.colormaps["cividis"]
NORM = Normalize(30, 95)
GROUP_SPECS = (
    dict(letter="j", grid_id="label_soc", title="Fraction \u00d7 initial SOC",
         xlabel="Labelled set fraction (%)", ylabel="Average initial SOC (%)",
         xticks=[5, 20, 30], yticks=[20, 40, 60]),
    dict(letter="k", grid_id="retention_soc", title="Capacity \u00d7 initial SOC",
         xlabel="Average capacity retention (%)", ylabel="Average initial SOC (%)",
         xticks=[60, 75, 90], yticks=[20, 40, 60]),
    dict(letter="l", grid_id="label_retention", title="Fraction \u00d7 capacity",
         xlabel="Labelled set fraction (%)", ylabel="Average capacity retention (%)",
         xticks=[5, 20, 30], yticks=[60, 75, 90]),
)
LAYOUT = {
    "a": (15.5, 244, 228, 55), "b": (25.6, 198, 69.5, 33.6),
    "c": (127.2, 198, 41.2, 28), "d": (25.6, 153, 41.2, 25),
    "e": (81.4, 153, 43, 25), "f": (140, 153, 41, 25),
    "g": (20, 99, 42, 24), "h": (78.5, 99, 42, 24), "i": (137, 99, 42, 24),
}


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_paths(input_dir=None, cache_dir=None):
    inputs = Path(input_dir) if input_dir is not None else ROOT / "results/runs/full"
    cache = Path(cache_dir) if cache_dir is not None else FIGURE_SOURCE
    return {**{name: inputs / name for name in DATA_FILES},
            "baseline_method_summary.csv": BASELINE_SUMMARY,
            **{name: cache / name for name in CACHE_FILES}}


def input_hashes(input_dir=None, cache_dir=None, *, include_draws=True):
    inputs = Path(input_dir) if input_dir is not None else ROOT / "results/runs/full"
    paths = source_paths(inputs, cache_dir)
    if include_draws:
        paths["draws.csv.gz"] = inputs / "draws.csv.gz"
    return {name: _hash(path) for name, path in paths.items()}


def validate_manifest(manifest):
    if manifest.get("main_seed") != 20260921 or manifest.get("main_draws") != 800000:
        raise ValueError("Use only the 800,000-draw primary seed 20260921")
    base = manifest.get("base_reference_inputs", {})
    if any(base.get(k) != v for k, v in dict(initial_soc=.4, retention=.8, label_fraction=.15).items()):
        raise ValueError("Unexpected current reference inputs")
    if base.get("deployment_rpt_fraction") != .04:
        raise ValueError("Expected reference deployment_rpt_fraction q=0.04")
    if tuple(manifest.get("sampled_parameters", [])) != ORIGINAL_PARAMETERS + ("deployment_rpt_fraction",):
        raise ValueError("Expected the original 15 inputs followed by deployment_rpt_fraction (16 inputs)")


def validate_fixed_q(frame, name):
    # A missing q column uses the validated manifest reference, preserving the old schema.
    for column in ("deployment_rpt_fraction", "q"):
        if column in frame and not np.allclose(frame[column], .04, rtol=0, atol=1e-12):
            raise ValueError(f"Expected q=0.04 throughout {name}")


def configure_response_norm(core):
    """Expand the common scale in 5-point steps only when source values need it."""
    frame = core["two_factor"]
    displayed = frame[(frame.metric == "cost_USD") & np.isclose(frame.reference_label_fraction, .15)
                      & frame.grid_id.isin([spec["grid_id"] for spec in GROUP_SPECS])
                      & frame.comparator.isin(COMPARATORS)]
    values = displayed.reduction_pct.to_numpy()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite response values")
    low, high = float(values.min()), float(values.max())
    NORM.vmin, NORM.vmax = min(30., 5 * np.floor(low / 5)), max(95., 5 * np.ceil(high / 5))
    return dict(minimum_pct=low, maximum_pct=high, vmin=NORM.vmin, vmax=NORM.vmax,
                original_limits=[35, 95], default_variant_limits=[30, 95])


def load_core(input_dir):
    directory = Path(input_dir)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    # Preserve the established parsing precision, including contour anchor placement.
    frames = {key: pd.read_csv(directory / name, **({"float_precision": "round_trip"} if key == "totals" else {})) for key, name in
              (("summary", "summary.csv"), ("totals", "method_summary.csv"),
               ("sensitivity", "sensitivity.csv"), ("two_factor", "two_factor.csv"))}
    for key, method_column, methods in (("summary", "comparison", COMPARATORS), ("totals", "method", METHODS)):
        frame = frames[key]
        expected = {(method, metric) for method in methods for metric in METRICS}
        if (len(frame) != len(expected) or frame.duplicated([method_column, "metric"]).any()
                or set(zip(frame[method_column], frame.metric)) != expected):
            raise ValueError(f"Incomplete or duplicate {key}")
        if not (frame.n_draws == 800000).all() or not (frame.n_valid == 800000).all():
            raise ValueError("All comparisons must use every primary-seed draw")
        numeric = frame.select_dtypes(include="number").to_numpy()
        if not np.isfinite(numeric).all():
            raise ValueError(f"Non-finite {key}")
        intervals = (("q025", "median", "q975"),) if key == "totals" else (
            ("saving_q025", "saving_median", "saving_q975"),
            ("reduction_q025_pct", "reduction_median_pct", "reduction_q975_pct"))
        if any((np.diff(frame[list(columns)].to_numpy(), axis=1) < 0).any() for columns in intervals):
            raise ValueError(f"Unordered quantiles: {key}")
    for key in ("sensitivity", "two_factor"):
        frame = frames[key]
        validate_fixed_q(frame, key)
        flags = frame.feasible.astype(str).str.lower().map({"true": True, "false": False})
        if flags.isna().any() or not flags.all():
            raise ValueError(f"The displayed domains must be feasible: {key}")
        frame["feasible"] = flags.astype(bool)
        if not frame.comparator.isin(COMPARATORS).all() or not frame.metric.isin(METRICS).all():
            raise ValueError(f"Unknown comparator or metric in {key}")
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
            raise ValueError(f"Non-finite {key}")
    sensitivity = frames["sensitivity"]
    keys = ["varying_parameter", "reference_label_fraction", "comparator", "metric", "value"]
    if sensitivity.duplicated(keys).any():
        raise ValueError("Duplicate sensitivity values")
    for parameter in ("retention", "initial_soc", "label_fraction"):
        selected = sensitivity[(sensitivity.varying_parameter == parameter)
                               & np.isclose(sensitivity.reference_label_fraction, .15)
                               & (sensitivity.metric == "cost_USD")]
        if selected.empty or not np.allclose(selected.value, selected[parameter]):
            raise ValueError(f"Missing or inconsistent {parameter} sweep")
        for fixed in ("retention", "initial_soc", "label_fraction"):
            if fixed != parameter and not np.allclose(selected[fixed], manifest["base_reference_inputs"][fixed]):
                raise ValueError(f"Sweep changes the fixed input {fixed}")
        points = set(selected.value)
        if len(points) < 2 or any(set(selected.loc[selected.comparator == method, "value"]) != points
                                 for method in COMPARATORS):
            raise ValueError(f"Incomplete {parameter} method comparison")
    original = pd.read_csv(BASELINE_SUMMARY, float_precision="round_trip")
    original = original[(original.method == "proposed") & (original.metric == "cost_USD")].copy()
    if len(original) != 1 or not (original.n_draws == 800000).all() or not (original.n_valid == 800000).all():
        raise ValueError("Ordinary Proposed requires the complete frozen primary-run summary")
    original["method"] = "proposed_no_targeted_rpt"
    cost_totals = pd.concat([original, frames["totals"].query("metric == 'cost_USD'")], ignore_index=True)
    core = dict(frames, manifest=manifest, cost_totals=cost_totals)
    configure_response_norm(core)
    for spec in GROUP_SPECS:
        for method in COMPARATORS:
            source_grid(core, spec["grid_id"], method)
    return core


def load_verified_density(input_dir, cache_dir=FIGURE_SOURCE, *, reference_only=False):
    cache_dir = Path(cache_dir)
    provenance = json.loads((cache_dir / "cost_density_provenance.json").read_text(encoding="utf-8"))
    original_hashes = {}
    if reference_only:
        # Explicit read-only validation of compact packaged artifacts. This does
        # not assert that the omitted historical raw draws were re-read.
        if (Path(input_dir).resolve() != (ROOT / "results/reference").resolve()
                or cache_dir.resolve() != FIGURE_SOURCE.resolve()):
            raise ValueError("Reference-only validation is limited to packaged reference artifacts")
        integrity = json.loads((ROOT / "reference_integrity.json").read_text(encoding="utf-8"))
        original_hashes = integrity["original_source_sha256"]
        for destination, expected in integrity["packaged_reference_sha256"].items():
            if _hash(ROOT / destination) != expected:
                raise ValueError(f"Packaged reference artifact changed: {destination}")

    if set(provenance.get("source_hashes", {})) != set(DENSITY_SOURCE_FILES):
        raise ValueError("Density provenance must hash current draws, method_summary and run_manifest")
    validate_manifest(json.loads((Path(input_dir) / "run_manifest.json").read_text(encoding="utf-8")))
    for name, expected in provenance["source_hashes"].items():
        actual = (original_hashes["outputs/" + name] if reference_only and name == "draws.csv.gz"
                  else _hash(Path(input_dir) / name))
        if actual != expected:
            raise ValueError(f"Density cache source changed: {name}")
    if set(provenance.get("caption_source_hashes", {})) != set(DATA_FILES):
        raise ValueError("Caption provenance must cover the current figure source tables and manifest")
    for name, expected in provenance["caption_source_hashes"].items():
        if _hash(Path(input_dir) / name) != expected:
            raise ValueError(f"Caption source changed: {name}; rebuild the figure sources")
    if provenance["primary_seed"] != 20260921 or provenance["n_draws_per_method"] != 800000:
        raise ValueError("Density cache is not from the primary run")
    if (provenance.get("n_free_inputs") != 16 or provenance.get("bin_count") != 160
            or provenance.get("smoothing") != dict(sigma_bins=1., mode="reflect", truncate=4.)
            or provenance.get("all_cost_quantiles_reproduced") is not True):
        raise ValueError("Density cache algorithm or 16-input provenance differs; rebuild it")
    if provenance.get("builder_sha256") != (original_hashes.get("build_density_cache.py") if reference_only
                                           else _hash(ROOT / "visualization/build_density_cache.py")):
        raise ValueError("Density builder changed; rebuild the cache")
    if provenance.get("no_targeted_rpt_summary_sha256") != _hash(BASELINE_SUMMARY):
        raise ValueError("Ordinary Proposed reference changed; rebuild the cache")
    if provenance.get("no_targeted_rpt_cost_column") != "data_cost_USD":
        raise ValueError("Ordinary Proposed must use the pre-targeting labelling costs")
    for name, field in (("cost_density.csv", "density_sha256"), ("figure_caption.md", "caption_sha256")):
        if provenance.get(field) != _hash(cache_dir / name):
            raise ValueError(f"Density cache artifact changed: {name}")
    density = pd.read_csv(cache_dir / "cost_density.csv", float_precision="round_trip")
    if tuple(density.columns) != DENSITY_COLUMNS:
        raise ValueError("Density cache columns differ")
    if set(density.method) != set(COST_METHODS):
        raise ValueError("Density cache methods differ")
    reference_bins = density.loc[density.method == METHODS[0], "log_cost"].to_numpy()
    for name, rows in density.groupby("method"):
        if rows["count"].sum() != 800000 or len(rows) != 160:
            raise ValueError(f"Density cache does not cover all draws for {name}")
        if (not np.isfinite(rows.select_dtypes(include="number").to_numpy()).all()
                or (rows.density < 0).any() or (rows["count"] < 0).any()
                or (rows["count"] % 1 != 0).any() or (rows.log_bin_width <= 0).any()
                or not (np.diff(rows.log_cost) > 0).all()):
            raise ValueError("Invalid density data")
        np.testing.assert_array_equal(rows.log_cost, reference_bins)
        np.testing.assert_allclose(np.diff(rows.log_cost), rows.log_bin_width.iloc[0], rtol=1e-12)
        np.testing.assert_allclose(rows.log_bin_width, rows.log_bin_width.iloc[0], rtol=1e-12)
        np.testing.assert_allclose((rows.density * rows.log_bin_width).sum(), 1, rtol=1e-12)
    if reference_only:
        provenance = dict(provenance, validation_scope="packaged_reference_artifacts; raw_draws_not_present")
    return density, provenance


def source_grid(core, grid_id, method):
    frame = core["two_factor"]
    selected = frame[(frame.grid_id == grid_id) & np.isclose(frame.reference_label_fraction, .15)
                     & (frame.comparator == method) & (frame.metric == "cost_USD")]
    if len(selected) != 2601 or not selected.feasible.all() or selected.duplicated(["x_value", "y_value"]).any():
        raise ValueError(f"Expected a complete feasible 51-by-51 grid: {grid_id}/{method}")
    grid = selected.pivot(index="y_value", columns="x_value", values="reduction_pct").sort_index().sort_index(axis=1)
    if grid.shape != (51, 51) or not np.isfinite(grid.to_numpy()).all():
        raise ValueError(f"Incomplete grid: {grid_id}/{method}")
    validate_fixed_q(selected, f"{grid_id}/{method}")
    if grid.to_numpy().min() < NORM.vmin - 1e-9 or grid.to_numpy().max() > NORM.vmax + 1e-9:
        raise ValueError("The shared colour scale would clip source values")
    return grid


def validate_destination(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    from portable_io import validate_output
    validate_output(output)
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError("Figure exports must be separate from the analysis inputs")


def _box(fig, x, y, width, height):
    fw, fh = fig.get_size_inches() * 25.4
    return x / fw, y / fh, width / fw, height / fh


def _text(fig, x, y, value, **kwargs):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.text(x / fw, y / fh, value, fontsize=FONT_PT, **kwargs)


def _axes(fig, letter):
    ax = fig.add_axes(_box(fig, *LAYOUT[letter]), gid=f"panel:{letter}")
    ax.tick_params(length=3, width=.65)
    return ax


def _workflow(ax, base):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    p, q = base["label_fraction"], base["deployment_rpt_fraction"]
    ax.add_patch(Rectangle((.06, .52), .035, .23, facecolor="#FBF9F9", edgecolor="#545454", lw=.8))
    ax.add_patch(Rectangle((.06, .52), .035, .23 * base["initial_soc"], facecolor=COLORS["proposed"], edgecolor="none"))
    ax.add_patch(Ellipse((.0775, .75), .035, .035, facecolor="#E5D0D4", edgecolor="#545454", lw=.8))
    ax.add_patch(Rectangle((.071, .768), .013, .015, facecolor="#999999", edgecolor="#545454", lw=.6))
    ax.text(.08, .36, f"1 tonne cells\nr = {base['retention']:g}\nSOC = {base['initial_soc']:g}",
            ha="center", va="center", fontsize=FONT_PT)
    ax.plot([.12, .19, .19], [.63, .63, .81], color="#505050", lw=.8)
    ax.plot([.19, .19], [.63, .39], color="#505050", lw=.8)
    rows = (
        (.25, .69, .34, .24, "#F5E8EB", f"Labelling: p = {100*p:g}%\nOriginal full RPT"),
        (.25, .22, .34, .34, "#E9F0F3", "All deployment cells: (1-p)N\nRequired de-energisation to zero SOC\nDiagnosis during discharge"),
        (.66, .15, .34, .48, "#F5E8EB", f"Targeted full RPT: q = {100*q:g}%\nq(1-p)N cells; start at zero SOC\n1.5 h charge + 10 min rest\n+ r h discharge; grid E_cell/eta"),
    )
    for x, y, width, height, fill, label in rows:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor="#A5A5A5", lw=.6))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=FONT_PT)
    for start, end in (((.19, .81), (.25, .81)), ((.19, .39), (.25, .39)),
                       ((.59, .39), (.66, .39)), ((.59, .81), (.66, .81)),
                       ((.83, .63), (.83, .70))):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=.8))
    ax.text(.83, .81, "Incremental time, electricity,\ncost and CO2", ha="center", va="center", fontsize=FONT_PT)
    ax.text(.42, .11, f"Remaining {100*(1-q):g}%: no extra RPT", ha="center", va="center", fontsize=FONT_PT)
    ax.text(.55, -.04, "Common baseline subtracted once: direct discharge from initial SOC to zero", ha="center", fontsize=FONT_PT)


def _unit_edges(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not (np.diff(values) > 0).all():
        raise ValueError("Grid coordinates must be strictly increasing")
    edges = np.r_[values[0], (values[:-1] + values[1:]) / 2, values[-1]]
    return (edges - values[0]) / (values[-1] - values[0])


def _contour_levels(values):
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-8:
        return []
    candidates = np.arange(5 * np.ceil(NORM.vmin / 5), NORM.vmax, 5.)
    candidates = candidates[(candidates > low + .08 * (high - low)) & (candidates < high - .08 * (high - low))]
    if not len(candidates):
        return []
    targets = low + (high - low) * np.array([.33, .67])
    return sorted(set(float(candidates[np.argmin(abs(candidates - target))]) for target in targets))


def _surface(ax, grid, gid):
    x, y = grid.columns.to_numpy() * 100, grid.index.to_numpy() * 100
    z = grid.to_numpy()
    xe, ye = x[0] + np.ptp(x) * _unit_edges(x), y[0] + np.ptp(y) * _unit_edges(y)
    mesh = ax.pcolormesh(xe, ye, z, cmap=CMAP, norm=NORM, shading="flat",
                         edgecolors="face", linewidth=.1, antialiased=False, gid=gid)
    levels = _contour_levels(z)
    if np.ptp(z) < 1e-8:
        ax.text(.5, .5, f"{z.mean():g}%", transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_PT, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=1))
    elif levels:
        contour = ax.contour(x, y, z, levels=levels, colors="#3C4248", linewidths=.5)
        contour.set_gid(gid + ":isolines")
        points = []
        for index, segments in enumerate(contour.allsegs):
            paths = [segment for segment in segments if len(segment) >= 2]
            if not paths:
                continue
            path = np.concatenate(paths)
            normalized = (path - [x[0], y[0]]) / [np.ptp(x), np.ptp(y)]
            # Two-digit, horizontal 10.5-pt labels need clearance inside a 17-mm map.
            interior = ((normalized >= .18) & (normalized <= .82)).all(axis=1)
            if not interior.any():
                continue
            path, normalized = path[interior], normalized[interior]
            anchor = [.32 + .34 * index, .35 + .35 * index]
            points.append(path[np.argmin(np.sum((normalized - anchor) ** 2, axis=1))])
        labels = ax.clabel(contour, fmt="%g", fontsize=FONT_PT, inline=True, inline_spacing=1.5, manual=points)
        for label in labels:
            label.set_rotation(0)
            label.set_color("#202020")
            label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=.92, pad=.45))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    return mesh


def _place_on_a4(fig):
    width, height = fig.get_size_inches() * 25.4
    offset = ((210 - width) / 2, (297 - height) / 2)
    axes = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    texts = [(text, text.get_position()) for text in fig.texts]
    legends = [(legend, legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).frozen())
               for legend in fig.legends]
    fig.set_size_inches(210 / 25.4, 297 / 25.4)
    for ax, box in axes:
        ax.set_position(((offset[0] + box.x0 * width) / 210, (offset[1] + box.y0 * height) / 297,
                         box.width * width / 210, box.height * height / 297))
    for text, (x, y) in texts:
        text.set_position(((offset[0] + x * width) / 210, (offset[1] + y * height) / 297))
    for legend, box in legends:
        legend.set_bbox_to_anchor(((offset[0] + box.x0 * width) / 210, (offset[1] + box.y0 * height) / 297,
                                   box.width * width / 210, box.height * height / 297), transform=fig.transFigure)


def compact_response_layout(fig):
    """Fit common heatmap gaps to rendered titles/ticks without resizing maps."""
    axes = {ax.get_gid(): ax for ax in fig.axes}
    labels = {text.get_gid(): text for text in fig.texts}
    groups = [[axes[f"response:{spec['grid_id']}:{method}"] for method in COMPARATORS]
              for spec in GROUP_SPECS]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mm_per_px = 25.4 / fig.dpi
    width, height = fig.get_size_inches() * 25.4
    horizontal, vertical = [], []
    for group in groups:
        boxes = [ax.get_window_extent(renderer) for ax in group]
        tight = [ax.get_tightbbox(renderer) for ax in group]
        for left, right in ((0, 1), (2, 3)):
            horizontal.append((tight[left].x1 - boxes[left].x1
                               + boxes[right].x0 - tight[right].x0) * mm_per_px)
        for top, bottom in ((0, 2), (1, 3)):
            vertical.append((boxes[top].y0 - tight[top].y0
                             + tight[bottom].y1 - boxes[bottom].y1) * mm_per_px)
    # Keep 0.8 mm of clear space between decorated axes, rounded up to 0.1 mm.
    gap_x = np.ceil((max(horizontal) + .8) * 10 - 1e-8) / 10
    gap_y = np.ceil((max(vertical) + .8) * 10 - 1e-8) / 10
    for group in groups:
        anchor = group[0].get_position().frozen()
        for index, ax in enumerate(group):
            row, column = divmod(index, 2)
            ax.set_position((anchor.x0 + column * (anchor.width + gap_x / width),
                             anchor.y0 - row * (anchor.height + gap_y / height),
                             anchor.width, anchor.height))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for spec, group in zip(GROUP_SPECS, groups):
        boxes = [ax.get_tightbbox(renderer) for ax in group]
        x_label = labels[f"response-label:{spec['grid_id']}:x"]
        y_label = labels[f"response-label:{spec['grid_id']}:y"]
        x_center = (group[0].get_position().x0 + group[1].get_position().x1) / 2
        y_center = (group[0].get_position().y1 + group[2].get_position().y0) / 2
        x_label.set_position((x_center, (min(box.y0 for box in boxes) * mm_per_px - 1) / height))
        label_half_width = y_label.get_window_extent(renderer).width * mm_per_px / 2
        y_label.set_position(((min(box.x0 for box in boxes) * mm_per_px - label_half_width - 1) / width,
                              y_center))
    bar = axes["colorbar:response"]
    bottom = min(ax.get_position().y0 for ax in groups[-1])
    top = max(ax.get_position().y1 for ax in groups[-1])
    left = bar.get_position().x0
    bar.set_position((left, bottom, RESPONSE_BAR_WIDTH_MM / width, top - bottom))
    labels["colorbar-label:response"].set_position(
        (left + (RESPONSE_BAR_WIDTH_MM + 8.7) / width, (bottom + top) / 2))


def render_current(core, density):
    validate_manifest(core["manifest"])
    configure_response_norm(core)
    with matplotlib.rc_context(STYLE):
        fig = plt.figure(figsize=np.array([190, 280]) / 25.4, dpi=180)
        _workflow(_axes(fig, "a"), core["manifest"]["base_reference_inputs"])
        _text(fig, 14.5, 239, "Monte Carlo: 800,000 draws; seed 20260921; 16 independent inputs\n"
              "p ~ Tri(0.05, 0.15, 0.30); q ~ Tri(0.03, 0.04, 0.05)")
        ax = _axes(fig, "b")
        totals = core["cost_totals"].set_index("method")
        for y, name in enumerate(COST_METHODS):
            rows = density[density.method == name]
            height = rows.density.to_numpy() / rows.density.max() * .65
            ax.fill_between(10 ** rows.log_cost.to_numpy(), y, y - height,
                            facecolor=COST_COLORS[name], edgecolor=COST_EDGES[name], lw=.6)
            record = totals.loc[name]
            ax.plot([record.q025, record.q975], [y + .12] * 2, color="#292929", lw=.9)
            ax.plot(record["median"], y + .12, "o", color="#222222", markersize=3, gid=f"cost-median:{name}")
        ax.set_xscale("log")
        half_bin = density.log_bin_width.iloc[0] / 2
        ax.set_xlim(10 ** (density.log_cost.min() - half_bin), 10 ** (density.log_cost.max() + half_bin))
        ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=2, width=.5)
        ax.set_yticks(range(len(COST_METHODS)), [LABELS[name] for name in COST_METHODS])
        ax.set_ylim(len(COST_METHODS) - .5, -.95)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Incremental cost (USD/tonne)")
        ax = _axes(fig, "c")
        matrix = core["summary"].pivot(index="comparison", columns="metric", values="reduction_median_pct")
        matrix = matrix.loc[list(COMPARATORS), list(METRICS)].to_numpy()
        matrix_image = ax.imshow(matrix, vmin=0, vmax=100, cmap=ROSE, aspect="auto", interpolation="nearest")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                rgb = np.array(ROSE(value / 100)[:3])
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = float(linear @ [.2126, .7152, .0722])
                color = "white" if 1.05 / (luminance + .05) >= (luminance + .05) / .05 else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=FONT_PT, color=color)
        ax.set_yticks(range(4), [LABELS[name] for name in COMPARATORS])
        ax.set_xticks(range(3), ["Cost", "Electricity", "CO\u2082"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes(_box(fig, 171.7, 198, 2.2, 28), gid="colorbar:reductions")
        cb = fig.colorbar(matrix_image, cax=cax, ticks=[0, 50, 100])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=FONT_PT, length=2)
        cb.set_label("Median reduction (%)", fontsize=FONT_PT, labelpad=6)
        for k, metric in enumerate(METRICS):
            ax = _axes(fig, chr(100 + k))
            subset = core["summary"].query("metric == @metric").set_index("comparison")
            for y, name in enumerate(COMPARATORS):
                record = subset.loc[name]
                ax.plot([record.saving_q025, record.saving_q975], [y, y], color=EDGES[name], lw=1.2,
                        gid=f"saving-interval:{metric}:{name}")
                ax.plot(record.saving_median, y, marker=MARKERS[name], color=COLORS[name],
                        markeredgecolor=EDGES[name], markersize=4.2, linestyle="none",
                        gid=f"saving-median:{metric}:{name}")
            ax.set_yticks(range(4), [LABELS[name] for name in COMPARATORS] if k == 0 else [])
            ax.set_ylim(3.6, -.6)
            locator = MaxNLocator(nbins=3, min_n_ticks=3)
            low, high = float(subset.saving_q025.min()), float(subset.saving_q975.max())
            ax.axvline(0, color="#777777", lw=.65, linestyle="--", zorder=0,
                       gid=f"saving-zero:{metric}")
            if low < 0:
                padding = .05 * max(high - low, 1)
                ticks = locator.tick_values(low - padding, max(0, high) + padding)
                ax.set_xlim(ticks[0], ticks[-1])
                ax.xaxis.set_major_locator(FixedLocator(ticks))
            else:
                ax.set_xlim(left=0)
                ax.xaxis.set_major_locator(locator)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(SAVINGS_LABELS[metric])
        handles = [Line2D([], [], color=EDGES[name], marker=MARKERS[name], markersize=3.7,
                          lw=1, label=LABELS[name]) for name in COMPARATORS]
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(100.65 / 190, 136 / 280),
                   ncol=4, columnspacing=1.5, handlelength=2).set_gid("legend:methods")
        cost_handles = [Patch(facecolor=COST_COLORS[name], edgecolor=COST_EDGES[name], label=LABELS[name])
                        for name in COST_METHODS]
        fig.legend(handles=cost_handles, loc="center", bbox_to_anchor=(100 / 190, 234 / 280),
                   ncol=3, columnspacing=1.5, handlelength=1.4).set_gid("legend:cost_methods")
        statistics = [Line2D([], [], color="#292929", marker="o", markersize=3, linestyle="none", label="Median"),
                      Line2D([], [], color="#292929", lw=.9, label="95% simulation interval")]
        fig.legend(handles=statistics, loc="center", bbox_to_anchor=(100 / 190, 182 / 280),
                   ncol=2, columnspacing=1.5, handlelength=2).set_gid("legend:mc_statistics")
        parameters = (("retention", [60, 75, 90], "Average capacity retention (%)"),
                      ("initial_soc", [20, 40, 60], "Average initial SOC (%)"),
                      ("label_fraction", [5, 10, 20, 30], "Labelled set fraction (%)"))
        for j, (parameter, ticks, xlabel) in enumerate(parameters):
            ax = _axes(fig, chr(103 + j))
            selected = core["sensitivity"][(core["sensitivity"].varying_parameter == parameter)
                        & np.isclose(core["sensitivity"].reference_label_fraction, .15)
                        & (core["sensitivity"].metric == "cost_USD")]
            for name in COMPARATORS:
                rows = selected[selected.comparator == name].sort_values("value")
                ax.plot(rows.value * 100, rows.reduction_pct, color=EDGES[name], lw=1,
                        marker=MARKERS[name], markersize=3, markevery=5, gid=f"oat:{parameter}:{name}")
            ax.set_xlim(selected.value.min() * 100, selected.value.max() * 100)
            ax.set_ylim(min(30, 5 * np.floor(selected.reduction_pct.min() / 5)),
                        max(100, 5 * np.ceil(selected.reduction_pct.max() / 5)))
            ax.set_xticks(ticks)
            ax.set_yticks([40, 60, 80, 100])
            if j == 0:
                ax.set_ylabel("Cost reduction (%)")
            ax.set_xlabel(xlabel)
        for index, spec in enumerate(GROUP_SPECS):
            left = 21 + 58.5 * index
            for j, method in enumerate(COMPARATORS):
                row, column = divmod(j, 2)
                ax = fig.add_axes(_box(fig, left + 29 * column, 53 - 26 * row, 17, 17),
                                  gid=f"response:{spec['grid_id']}:{method}")
                ax.set_box_aspect(1)
                mesh = _surface(ax, source_grid(core, spec["grid_id"], method), f"continuous:{spec['grid_id']}:{method}")
                ax.set_xticks(spec["xticks"])
                ax.set_yticks(spec["yticks"])
                ax.tick_params(length=1.8, width=.5, pad=2, labelsize=FONT_PT)
                ax.tick_params(axis="x", labelbottom=row == 1, bottom=row == 1, pad=5.5)
                ax.tick_params(axis="y", labelleft=column == 0, left=column == 0)
                title = "Partial-charge-\nbased" if method == "partial_charge" else LABELS[method]
                ax.set_title(title, color="black", fontsize=FONT_PT, pad=7)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            _text(fig, left + 23, 17.5, spec["xlabel"], ha="center", va="top",
                  gid=f"response-label:{spec['grid_id']}:x")
            _text(fig, left - 12, 48.5, spec["ylabel"], rotation=90, ha="center", va="center",
                  gid=f"response-label:{spec['grid_id']}:y")
        cax = fig.add_axes(_box(fig, 185, 27, RESPONSE_BAR_WIDTH_MM, 43), gid="colorbar:response")
        middle = 5 * np.round((NORM.vmin + NORM.vmax) / 10)
        cb = fig.colorbar(mesh, cax=cax, orientation="vertical", ticks=[NORM.vmin, middle, NORM.vmax])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=FONT_PT, length=1.8, width=.5, pad=2)
        _text(fig, 196.7, 48.5, "Cost reduction (%)", rotation=90, ha="center", va="center",
              gid="colorbar-label:response")
        _place_on_a4(fig)
        compact_response_layout(fig)
        return fig
