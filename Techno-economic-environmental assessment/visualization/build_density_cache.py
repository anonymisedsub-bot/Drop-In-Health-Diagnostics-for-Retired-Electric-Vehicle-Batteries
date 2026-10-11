"""Rebuild targeted-RPT figure sources from every current primary-seed cost draw."""

import argparse
import json
from pathlib import Path
import sys

import scipy
from scipy.ndimage import gaussian_filter1d

from tea_figure import (
    ROOT, FIGURE_SOURCE, METHODS, COST_METHODS, BASELINE_SUMMARY, LABELS, DENSITY_COLUMNS, DENSITY_SOURCE_FILES,
    np, pd, _hash, load_core, configure_response_norm, validate_destination,
)


N_DRAWS = 800000
N_BINS = 160
SMOOTHING = dict(sigma_bins=1., mode="reflect", truncate=4.)


def read_cost_draws(path, expected_draws=N_DRAWS, chunksize=200000):
    """Index all costs by draw ID so duplicates and missing records cannot hide."""
    costs = {method: np.full(expected_draws, np.nan) for method in COST_METHODS}
    columns = ["draw_id", "method", "cost_USD", "data_cost_USD", "feasible"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, float_precision="round_trip"):
        if not chunk.method.isin(METHODS).all():
            raise ValueError("Unknown method in cost draws")
        flags = chunk.feasible.astype(str).str.lower().map({"true": True, "false": False})
        if flags.isna().any() or not flags.all():
            raise ValueError("Every density draw must be feasible")
        for method, rows in chunk.groupby("method", sort=False):
            raw_ids = rows.draw_id.to_numpy(dtype=float)
            if (not np.isfinite(raw_ids).all() or (raw_ids % 1 != 0).any()
                    or (raw_ids < 0).any() or (raw_ids >= expected_draws).any()):
                raise ValueError(f"Invalid draw IDs for {method}")
            ids = raw_ids.astype(np.int64)
            if len(np.unique(ids)) != len(ids) or np.isfinite(costs[method][ids]).any():
                raise ValueError(f"Duplicate draw IDs for {method}")
            values = rows.cost_USD.to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Log-cost densities require finite positive costs: {method}")
            costs[method][ids] = values
            if method == "proposed":
                original = rows.data_cost_USD.to_numpy(dtype=float)
                if not np.isfinite(original).all() or (original <= 0).any() or (original > values).any():
                    raise ValueError("Invalid pre-targeting Proposed costs")
                costs["proposed_no_targeted_rpt"][ids] = original
    for method, values in costs.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Missing cost draws for {method}; expected all {expected_draws}")
    return costs


def density_from_costs(costs, totals):
    """160 common log10 bins, raw integer counts, and normalized sigma=1 density."""
    if set(costs) != set(COST_METHODS):
        raise ValueError("Cost draws must cover the six cost scenarios")
    reference = totals[totals.metric == "cost_USD"]
    if (len(reference) != len(COST_METHODS) or reference.method.duplicated().any()
            or set(reference.method) != set(COST_METHODS)):
        raise ValueError("Missing or duplicate cost quantile rows in method_summary")
    reference = reference.set_index("method")
    logs = {}
    for method in COST_METHODS:
        values = np.asarray(costs[method], dtype=float)
        if values.shape != (N_DRAWS,):
            raise ValueError(f"Expected all 800000 cost draws for {method}")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid cost draws for {method}")
        row = reference.loc[method]
        if row.n_draws != N_DRAWS or row.n_valid != N_DRAWS:
            raise ValueError(f"Incomplete method_summary for {method}")
        actual = np.quantile(values, [.025, .5, .975], method="linear")
        if not np.allclose(actual, row[["q025", "median", "q975"]].to_numpy(dtype=float),
                           rtol=1e-12, atol=1e-10):
            raise ValueError(f"Cost quantiles differ from method_summary for {method}")
        logs[method] = np.log10(values)
    low = min(values.min() for values in logs.values())
    high = max(values.max() for values in logs.values())
    if low == high:
        raise ValueError("A density requires a nonzero log-cost range")
    edges = np.linspace(low, high, N_BINS + 1)
    width = float(edges[1] - edges[0])
    centers = (edges[:-1] + edges[1:]) / 2
    frames = []
    for method in COST_METHODS:
        counts, _ = np.histogram(logs[method], bins=edges)
        if counts.sum() != N_DRAWS:
            raise ValueError(f"Histogram lost draws for {method}")
        smoothed = gaussian_filter1d(counts.astype(float), sigma=SMOOTHING["sigma_bins"],
                                     mode=SMOOTHING["mode"], truncate=SMOOTHING["truncate"])
        frames.append(pd.DataFrame(dict(method=method, log_cost=centers,
                                        density=smoothed / (smoothed.sum() * width),
                                        count=counts, log_bin_width=width)))
    return pd.concat(frames, ignore_index=True)[list(DENSITY_COLUMNS)]


def current_caption(core):
    bounds = configure_response_norm(core)
    increases = []
    for row in core["summary"].itertuples():
        fraction = getattr(row, "fraction_negative_saving", 0)
        if row.metric in ("energy_kWh", "co2_kg") and fraction > 0:
            quantity = "purchased-electricity" if row.metric == "energy_kWh" else "CO2"
            increases.append(f"Relative to {LABELS[row.comparison]}, {quantity} savings are negative "
                             f"in {100*fraction:.2f}% of primary draws. ")
    return (
        "# Targeted-RPT figure caption\n\n"
        "Fig. | Incremental testing burdens with targeted deployment RPT. "
        "a, Proposed diagnoses all deployment cells, (1-p)N, during required de-energisation. "
        "After that discharge, q(1-p)N selected cells receive a full RPT from zero SOC: "
        "1.5 h charge + 10 min rest + r h discharge, with E_cell/eta of extra purchased electricity "
        "per selected cell. The common initial-SOC-to-zero discharge baseline is subtracted once. "
        "The training fraction p sets the labelling pool; q sets additional deployment retesting. "
        "High uncertainty and high estimated capacity describe the selection policy, not a validated "
        "risk distribution. Selected cells are modelled using the batch representative retention; "
        "no accuracy improvement is established here. "
        "b, Incremental cost half-violins from the supplied 800,000-draw primary-seed density cache per method, using "
        "six rows: Proposed (no targeted RPT), Proposed+targeted RPT, Pulse-based, EIS-based, "
        "Partial-charge-based and RPT-based. Ordinary Proposed uses the labelling-stage "
        "costs from the same draws (q=0), checked against the supplied q=0 cost summary; "
        "it is a panel-b reference, not an additional comparator in panels c-l. The two Proposed "
        "scenarios share a rose colour family, with the non-targeted scenario lighter. Densities use "
        "160 shared equal-width log10-cost bins and one-bin Gaussian smoothing (reflect boundary, "
        "four-sigma truncation); widths are normalized separately. Black points and intervals show "
        "medians and 2.5th-97.5th percentiles. The logarithmic x axis has minor ticks at 2-9 per decade. "
        "c, Median paired percentage reductions relative to each comparator. d-f, Paired cost, "
        "purchased-electricity and CO2 savings, shown as medians and central 95% simulation intervals, "
        "not confidence intervals. Absolute quantities are per tonne; positive savings favour Proposed, "
        "whereas negative savings indicate an increase in the corresponding burden. Simulation "
        "intervals can cross zero; negative intervals are fully displayed and dashed lines mark zero savings. "
        + "".join(increases) +
        "Panels b-f use seed 20260921 with 16 independent triangular inputs, including "
        "p ~ Tri(0.05, 0.15, 0.30), initial SOC ~ Tri(0.2, 0.4, 0.6), and "
        "q ~ Tri(0.03, 0.04, 0.05). The 15 shared inputs use seed 20260921 and q uses a separate "
        "independent stream. g-i, Deterministic one-factor cost sensitivity to retention, initial SOC "
        "and labelled fraction. Unscanned inputs stay at r=0.8, initial SOC=0.4 and p=0.15 as applicable; "
        "q=0.04 (4% of the deployment pool) is fixed throughout g-l. Average capacity retention (%) is "
        "100r relative to nominal capacity. j-l, Continuous two-factor cost-reduction surfaces: "
        "labelled fraction and initial SOC at r=0.8, capacity and initial SOC at p=0.15, and labelled "
        "fraction and capacity at initial SOC=0.4. Each 2-by-2 group shows Pulse-based and EIS-based "
        "in the top row, and Partial-charge-based and RPT-based in the bottom row. Horizontal and "
        "vertical axes correspond to the first and second factors. All twelve maps use the current "
        f"51-by-51 grids and a common cividis colour scale from {bounds['vmin']:g} to {bounds['vmax']:g}%. "
        f"Displayed source bounds are {bounds['minimum_pct']:.6f} to {bounds['maximum_pct']:.6f}%. "
        "The default 30-95% scale expands in five-point steps if "
        "current values require it; no source values are clipped. Contour labels give cost reduction "
        "in percent, with local interpolation between grid nodes and no surface fitting or smoothing. "
        "Proposed-versus-RPT responses are read from the current data and depend on initial SOC. "
        "Economic and equipment inputs remain fixed in g-l. These are protocol-conditioned model "
        "results, not industrial validation. Current source data accompany the panels.\n\n"
        "Layout: panels j-l have row-major method ordering and automatic gaps around titles and ticks. "
        "The common borderless response colour bar is 3 mm wide and matches the complete two-row "
        "heatmap axes height. All panel and shared-asset text is 10.5 pt at its recorded physical size. "
        "Panels are exported as PNG at 300 dpi for Figure 8 assembly.\n"
    )


def validate_cache_destination(cache_dir, input_dir):
    """Allow the dedicated cache child while protecting all analysis tables."""
    cache_dir, input_dir = Path(cache_dir).resolve(), Path(input_dir).resolve()
    if cache_dir == input_dir / "figure_source":
        from portable_io import validate_output
        validate_output(cache_dir)
    else:
        validate_destination(cache_dir, input_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/runs/full/figure_source")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/runs/full")
    args = parser.parse_args(argv)
    validate_cache_destination(args.cache_dir, args.input_dir)
    # Export and cache destinations must never overwrite one another.
    validate_destination(args.cache_dir, ROOT / "visualization/outputs")
    inputs = args.input_dir
    required = ("summary.csv", "method_summary.csv", "sensitivity.csv", "two_factor.csv",
                "run_manifest.json", "draws.csv.gz")
    missing = [name for name in required if not (inputs / name).is_file()]
    if missing:
        parser.error("Variant data are not ready: " + ", ".join(missing))
    before = {name: _hash(inputs / name) for name in required}
    core = load_core(inputs)
    if core["manifest"].get("synthetic_test_only"):
        raise ValueError("Synthetic fixtures cannot produce final figure sources")
    print("Reading all 800,000 current cost draws per method...", flush=True)
    costs = read_cost_draws(inputs / "draws.csv.gz")
    baseline_hash = _hash(BASELINE_SUMMARY)
    density = density_from_costs(costs, core["cost_totals"])
    caption = current_caption(core)
    if before != {name: _hash(inputs / name) for name in required}:
        raise ValueError("Analysis inputs changed during cache construction; retry after the run finishes")
    if baseline_hash != _hash(BASELINE_SUMMARY):
        raise ValueError("Ordinary Proposed reference changed during cache construction")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    density.to_csv(args.cache_dir / "cost_density.csv", index=False, float_format="%.17g", lineterminator="\n")
    (args.cache_dir / "figure_caption.md").write_text(caption, encoding="utf-8", newline="\n")
    provenance = dict(
        source_hashes={name: before[name] for name in DENSITY_SOURCE_FILES},
        caption_source_hashes={name: before[name] for name in required if name != "draws.csv.gz"},
        n_draws_per_method=N_DRAWS, primary_seed=20260921, n_free_inputs=16,
        bin_count=N_BINS, bin_space="log10", shared_edges=True,
        log_bin_edges=[float(density.log_cost.min() - density.log_bin_width.iloc[0] / 2),
                       float(density.log_cost.max() + density.log_bin_width.iloc[0] / 2)],
        smoothing=SMOOTHING, quantile_method="linear", all_cost_quantiles_reproduced=True,
        cost_scenario_order=COST_METHODS, no_targeted_rpt_cost_column="data_cost_USD",
        no_targeted_rpt_summary_sha256=baseline_hash,
        draw_ids_verified="Each method has every integer draw_id from 0 through 799999 exactly once",
        density_sha256=_hash(args.cache_dir / "cost_density.csv"),
        caption_sha256=_hash(args.cache_dir / "figure_caption.md"),
        response_scale=configure_response_norm(core), builder_sha256=_hash(__file__),
        python_executable=sys.executable, numpy_version=np.__version__,
        pandas_version=pd.__version__, scipy_version=scipy.__version__,
    )
    (args.cache_dir / "cost_density_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified and cached six cost scenarios: {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
