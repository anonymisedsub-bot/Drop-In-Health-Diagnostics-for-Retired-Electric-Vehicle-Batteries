"""Separate targeted-RPT variant: paired Monte Carlo and deterministic sensitivity."""

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

from tea_model import METHODS, METRICS, equipment_hourly_rate, evaluate
from portable_io import validate_output


ROOT = Path(__file__).resolve().parent
COMPARATORS = METHODS[1:]
QUANTILES = [0.025, 0.5, 0.975]
OUTPUT_FILES = (
    "parameter_registry_snapshot.json", "analysis_design_snapshot.json",
    "baseline_reconciliation.json", "reference_baselines.csv",
    "sensitivity_methods.csv", "sensitivity.csv", "two_factor.csv",
    "summary.csv", "method_summary.csv", "joint_events.json", "targeted_rpt_summary.csv",
    "uniform_shape_summary.csv", "uniform_joint_events.json",
    "parameter_draws.csv.gz", "draws.csv.gz", "paired_savings.csv.gz",
    "targeted_rpt_draws.csv.gz", "q_reference_scenarios.csv", "old_new_summary.csv",
    "old_new_method_summary.csv", "original_reproduction.json",
)


def _source_bytes():
    paths = [ROOT / name for name in ("03_parameter_registry.json", "04_analysis_design.json",
                                    "tea_model.py", "run_analysis.py", "08_execution_protocol.md")]
    paths.extend(ROOT.joinpath("baseline").glob("*.py"))
    paths.extend(ROOT.joinpath("baseline").glob("*.json"))
    paths.extend(ROOT.joinpath("results/baseline").glob("*.csv"))
    paths.extend(ROOT.joinpath("results/baseline").glob("*.json"))
    return {str(path.relative_to(ROOT.parent)): path.read_bytes() for path in paths}


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def load_configuration():
    """Freeze configuration and provenance from the same startup source bytes."""
    sources = _source_bytes()
    registry, design = (json.loads(sources[str((ROOT / name).relative_to(ROOT.parent))].decode("utf-8"))
                        for name in ("03_parameter_registry.json", "04_analysis_design.json"))
    # Match tea_model.reference_inputs without reopening the live registry.
    base = {p["id"]: p.get("proposal_base", p.get("value")) for p in registry["parameters"]
            if "proposal_base" in p or "value" in p}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    return _freeze(dict(registry=registry, design=design, base=base, source_hashes=hashes))


def check_sources_unchanged(expected):
    try:
        current = {name: hashlib.sha256(content).hexdigest() for name, content in _source_bytes().items()}
    except OSError as exc:
        raise RuntimeError("Source files changed or became unreadable during analysis") from exc
    changed = sorted(name for name in set(expected) | set(current) if expected.get(name) != current.get(name))
    if changed:
        raise RuntimeError("Source files changed during analysis: " + ", ".join(changed))


def sample_inputs(n, seed, family="triangular", *, config):
    registry, design = config["registry"], config["design"]
    parameters = {p["id"]: p for p in registry["parameters"]}
    names = design["monte_carlo"]["sampled_parameters"]
    original_names = [name for name in names if name != "deployment_rpt_fraction"]
    lows, modes, highs = (np.array([parameters[k][field] for k in original_names]) for field in ("low", "mode", "high"))
    rng = np.random.default_rng(seed)
    if family == "triangular":
        samples = rng.triangular(lows, modes, highs, size=(n, len(original_names)))
    elif family == "uniform":
        samples = rng.uniform(lows, highs, size=(n, len(original_names)))
    else:
        raise ValueError("Unknown distribution family")
    frame = pd.DataFrame(samples, columns=original_names)
    q = parameters["deployment_rpt_fraction"]
    q_rng = np.random.default_rng(np.random.SeedSequence([seed, design["targeted_rpt"]["independent_seed_sequence_tag"]]))
    frame["deployment_rpt_fraction"] = (q_rng.triangular(q['low'],q['mode'],q['high'],n)
        if family == "triangular" else q_rng.uniform(q['low'],q['high'],n))
    frame.insert(0, "draw_id", np.arange(n))
    return frame


def evaluate_draws(frame, batch_size=20000, *, base):
    values = np.empty((len(frame), len(METHODS), len(METRICS)))
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        inputs = {**base, **{k: batch[k].to_numpy() for k in batch if k != "draw_id"}}
        result = evaluate(inputs)
        for j, method in enumerate(METHODS):
            for k, metric in enumerate(METRICS):
                values[start:start + len(batch), j, k] = result[method][metric]
    return values


def _quantiles(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, QUANTILES) if len(finite) else np.full(3, np.nan)


def paired_values(values, method_index, metric_index):
    comparator = values[:, method_index, metric_index]
    proposed = values[:, 0, metric_index]
    saving = comparator - proposed
    relative = np.full_like(saving, np.nan)
    np.divide(100 * saving, comparator, out=relative, where=np.isfinite(saving) & (comparator > 0))
    return saving, relative


def summarise(values):
    rows, totals, joint = [], [], {}
    n = len(values)
    for j, method in enumerate(METHODS):
        for k, metric in enumerate(METRICS):
            y = values[:, j, k]
            q = _quantiles(y)
            valid = np.isfinite(y)
            totals.append(dict(method=method, metric=metric, n_draws=n, n_valid=int(valid.sum()),
                               coverage=float(valid.mean()), q025=q[0], median=q[1], q975=q[2]))
            if j == 0:
                continue
            saving, relative = paired_values(values, j, k)
            good = np.isfinite(saving)
            q, qr = _quantiles(saving), _quantiles(relative)
            rows.append(dict(comparison=method, metric=metric, n_draws=n, n_valid=int(good.sum()),
                             n_valid_relative=int(np.isfinite(relative).sum()), coverage=float(good.mean()),
                             saving_q025=q[0], saving_median=q[1], saving_q975=q[2],
                             reduction_q025_pct=qr[0], reduction_median_pct=qr[1], reduction_q975_pct=qr[2],
                             fraction_negative_saving=float(np.mean(saving[good] < 0)) if good.any() else np.nan,
                             fraction_positive_saving=float(np.mean(saving[good] > 0)) if good.any() else np.nan))
    for k, metric in enumerate(METRICS):
        delta = values[:, 1:, k] - values[:, :1, k]
        good = np.isfinite(delta).all(axis=1)
        positive = (delta > 0).all(axis=1)
        joint[metric] = dict(n_draws=n, n_valid=int(good.sum()), coverage=float(good.mean()),
                             fraction_all_comparators_positive=float(positive[good].mean()) if good.any() else None,
                             minimum_absolute_saving_quantiles=_quantiles(np.min(delta[good], axis=1)).tolist())
    return pd.DataFrame(rows), pd.DataFrame(totals), joint


def grid_values(low, high, points, references=()):
    values = list(np.linspace(low, high, points))
    for reference in references:
        near = next((i for i, value in enumerate(values) if abs(value - reference) < 1e-12), None)
        if near is None:
            values.append(reference)
        else:
            values[near] = reference
    return sorted(set(values))


def sensitivity_cases(*, design, base):
    spec = design["one_at_a_time"]
    cases = []
    for parameter in spec["parameters"]:
        bounds = spec["ranges"][parameter]
        budgets = [s["label_fraction"] for s in design["reference_label_scenarios"]] if parameter != "label_fraction" else [base["label_fraction"]]
        values = grid_values(bounds["low"], bounds["high"], spec["points_per_parameter"], spec["include_reference_values"][parameter])
        for budget in budgets:
            for value in values:
                row = dict(sweep_id=f"{parameter}_p{budget:g}", varying_parameter=parameter, value=value,
                           reference_label_fraction=budget, retention=base["retention"],
                           initial_soc=base["initial_soc"], label_fraction=budget)
                row[parameter] = value
                cases.append(row)
    return cases


def _case_outputs(cases, *, base):
    frame = pd.DataFrame(cases)
    frame["deployment_rpt_fraction"] = base["deployment_rpt_fraction"]
    parameters = {**base, **{key: frame[key].to_numpy() for key in ("retention", "initial_soc", "label_fraction")}}
    result = evaluate(parameters)
    methods, paired = [], []
    for method in METHODS:
        data = frame.copy()
        data["method"] = method
        for key in (*METRICS, "time_channel_h", "feasible"):
            data[key] = result[method][key]
        methods.append(data)
        if method == "proposed":
            continue
        for metric in METRICS:
            data = frame.copy()
            data["comparator"] = method
            data["metric"] = metric
            data["saving"] = result[method][metric] - result["proposed"][metric]
            ratio = np.full(len(frame), np.nan)
            np.divide(data["saving"].to_numpy() * 100, result[method][metric], out=ratio,
                      where=np.isfinite(data["saving"]) & (result[method][metric] > 0))
            data["reduction_pct"] = ratio
            data["feasible"] = result[method]["feasible"] & result["proposed"]["feasible"]
            paired.append(data)
    return pd.concat(methods, ignore_index=True), pd.concat(paired, ignore_index=True)


def deterministic_analysis(out, *, design, base):
    oat_cases = sensitivity_cases(design=design, base=base)
    method_frame, paired_frame = _case_outputs(oat_cases, base=base)
    method_frame.to_csv(out / "sensitivity_methods.csv", index=False, float_format="%.17g")
    paired_frame.to_csv(out / "sensitivity.csv", index=False, float_format="%.17g")
    cases = []
    for grid in design["two_factor_grids"]:
        budgets = [s["label_fraction"] for s in design["reference_label_scenarios"]] if grid.get("repeat_for_reference_label_scenarios") else [base["label_fraction"]]
        for budget in budgets:
            for x in np.linspace(grid["x"]["low"], grid["x"]["high"], grid["x"]["points"]):
                for y in np.linspace(grid["y"]["low"], grid["y"]["high"], grid["y"]["points"]):
                    row = dict(grid_id=grid["id"], reference_label_fraction=budget,
                               x_parameter=grid["x"]["parameter"], y_parameter=grid["y"]["parameter"], x_value=x, y_value=y,
                               retention=base["retention"], initial_soc=base["initial_soc"], label_fraction=budget)
                    row[grid["x"]["parameter"]] = x
                    row[grid["y"]["parameter"]] = y
                    cases.append(row)
    _, grids = _case_outputs(cases, base=base)
    grids.to_csv(out / "two_factor.csv", index=False, float_format="%.17g")
    return dict(oat_cases=len(oat_cases), two_factor_cases=len(cases))


def baseline_analysis(out, *, design, base):
    spec = design["baselines"][1]
    b1_inputs = spec["reference_inputs"]
    b1 = evaluate(b1_inputs, equipment_rate=spec["equipment_rate_USD_per_channel_h"])
    expected = spec["expected_formula_audit_cost_USD_per_tonne"]
    errors = {m: float(b1[m]["cost_USD"][0] - expected[m]) for m in METHODS}
    if max(abs(v) for v in errors.values()) > 1e-8:
        raise AssertionError("B1 regression mismatch")
    rows = []
    for budget in (0.065, 0.15):
        results = evaluate({**base, "label_fraction": budget})
        for method, values in results.items():
            rows.append({"label_fraction": budget, "deployment_rpt_fraction": base["deployment_rpt_fraction"], "method": method,
                         **{k: (bool(v[0]) if k == "feasible" else float(v[0])) for k, v in values.items()}})
    pd.DataFrame(rows).to_csv(out / "reference_baselines.csv", index=False, float_format="%.17g")
    result = dict(B0_submitted_totals=design["baselines"][0]["expected_cost_USD_per_tonne"],
                  B1_costs={m: float(b1[m]["cost_USD"][0]) for m in METHODS},
                  B1_errors_USD=errors, B1_passed=True,
                  B1_reference_inputs=b1_inputs, B2_reference_inputs=base,
                  B2_equipment_rate_USD_per_channel_h=float(equipment_hourly_rate(base)),
                  B2_charge_efficiency=base["charge_efficiency"],
                  protocol_mode="user_selected_original_fixed_conditioning_times",
                  B0_workbook_verification="Frozen legacy totals only; original TEA_results_R1/outputs/source_audit.json contains the prior workbook audit; no new workbook audit is claimed")
    write_json(out / "baseline_reconciliation.json", result)


def variant_comparison(out, inputs, values, *, base):
    """Compare with q=0 on identical draws; check the archived run when sizes match."""
    old_inputs = inputs.copy()
    old_inputs["deployment_rpt_fraction"] = 0.0
    old_values = evaluate_draws(old_inputs, base=base)
    old_summary, old_methods, _ = summarise(old_values)
    new_summary, new_methods, _ = summarise(values)
    old_summary.merge(new_summary, on=["comparison", "metric"], suffixes=("_original", "_targeted"),
                      validate="one_to_one").to_csv(out / "old_new_summary.csv", index=False, float_format="%.17g")
    old_methods.merge(new_methods, on=["method", "metric"], suffixes=("_original", "_targeted"),
                      validate="one_to_one").to_csv(out / "old_new_method_summary.csv", index=False, float_format="%.17g")
    frozen = json.loads((ROOT / "results/baseline/run_manifest.json").read_text(encoding="utf-8"))
    checked = len(inputs) == frozen["main_draws"]
    if checked:
        for name, actual, keys in (("summary.csv", old_summary, ["comparison", "metric"]),
                                   ("method_summary.csv", old_methods, ["method", "metric"])):
            expected = pd.read_csv(ROOT / "results/baseline" / name, float_precision="round_trip")
            actual, expected = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
            pd.testing.assert_index_equal(actual.index, expected.index)
            np.testing.assert_allclose(actual.to_numpy(dtype=float), expected[actual.columns].to_numpy(dtype=float),
                                       rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(old_values[:, 1:, :], values[:, 1:, :])
    write_json(out / "original_reproduction.json", dict(
        frozen_summary_checked=checked, unchanged_comparators_exact=True,
        interpretation="q=0 on identical shared draws reproduces the preserved original; tolerances only check numerical identity, not Monte Carlo convergence."))
    target = evaluate({**base, **{k: inputs[k].to_numpy() for k in inputs if k != "draw_id"}})["proposed"]
    fields = [k for k in target if k.startswith("targeted_rpt_") or k == "record_only_deployment_cell_count"]
    raw = inputs[["draw_id", "label_fraction", "deployment_rpt_fraction"]].copy()
    for key in fields:
        raw[key] = target[key]
    raw["targeted_fraction_of_entire_batch"] = raw.deployment_rpt_fraction * (1 - raw.label_fraction)
    with gzip.open(out / "targeted_rpt_draws.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        raw.to_csv(stream, index=False, float_format="%.17g")
    rows = []
    for key in raw.columns[3:]:
        lo, med, hi = _quantiles(raw[key].to_numpy())
        rows.append(dict(metric=key, n_draws=len(inputs), q025=lo, median=med, q975=hi))
    pd.DataFrame(rows).to_csv(out / "targeted_rpt_summary.csv", index=False, float_format="%.17g")
    rows = []
    for p in (0.065, 0.15):
        for q in (0, 0.03, 0.04, 0.05):
            ref = evaluate({**base, "label_fraction": p, "deployment_rpt_fraction": q})
            for method, data in ref.items():
                row = dict(label_fraction=p, deployment_rpt_fraction=q, method=method,
                           **{k: float(data[k][0]) for k in (*METRICS, "time_channel_h", *fields)})
                for key in METRICS:
                    row["reduction_pct_" + key] = float(100 * (1 - ref["proposed"][key][0] / data[key][0]))
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / "q_reference_scenarios.csv", index=False, float_format="%.17g")


def export_draws(out, inputs, values, batch_size=20000, *, base):
    if values.shape != (len(inputs), len(METHODS), len(METRICS)):
        raise ValueError("Retained metric dimensions do not match the parameter draws")
    with gzip.open(out / "parameter_draws.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        inputs.to_csv(stream, index=False, float_format="%.17g")
    fields = (*METRICS, "time_channel_h", "data_time_channel_h", "deployment_time_channel_h",
              "data_energy_kWh", "deployment_energy_kWh", "data_cost_USD", "deployment_cost_USD",
              "gross_time_channel_h", "baseline_time_channel_h")
    with gzip.open(out / "draws.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        for start in range(0, len(inputs), batch_size):
            batch = inputs.iloc[start:start + batch_size]
            result = evaluate({**base, **{k: batch[k].to_numpy() for k in batch if k != "draw_id"}})
            frames = []
            for j, method in enumerate(METHODS):
                totals = np.column_stack([result[method][metric] for metric in METRICS])
                np.testing.assert_allclose(
                    totals, values[start:start + len(batch), j, :], rtol=0, atol=0, equal_nan=True,
                    err_msg=f"Exported metrics differ for {method}, rows {start}:{start + len(batch)}")
                frame = pd.DataFrame({"draw_id": batch["draw_id"].to_numpy(), "method": method,
                                      **{key: result[method][key] for key in fields}, "feasible": result[method]["feasible"]})
                frame["inapplicability_reason"] = np.where(frame["feasible"], "", "arrival_SOC_plus_0.25_exceeds_one")
                frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(stream, index=False, header=(start == 0), float_format="%.17g")
    with gzip.open(out / "paired_savings.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        for j, method in enumerate(METHODS[1:], 1):
            data = {"draw_id": inputs["draw_id"].to_numpy(), "comparison": method}
            for k, metric in enumerate(METRICS):
                data["saving_" + metric], data["reduction_pct_" + metric] = paired_values(values, j, k)
            data["feasible"] = np.isfinite(values[:, j, :]).all(axis=1)
            pd.DataFrame(data).to_csv(stream, index=False, header=(j == 1), float_format="%.17g")


def write_json(path, data):
    def clean(obj):
        if isinstance(obj, Mapping):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-draws", type=int, default=800000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/runs/full")
    args = parser.parse_args()
    if args.max_draws <= 0:
        parser.error("--max-draws must be positive")
    started = time.monotonic()
    out = args.output_dir.resolve()
    try:
        validate_output(out)
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    # These markers must never describe a partially overwritten rerun.
    pending_manifest = out / "run_manifest.json.tmp"
    for name in ("run_manifest.json", "verification.json", pending_manifest.name):
        (out / name).unlink(missing_ok=True)
    config = load_configuration()
    registry, design, base = config["registry"], config["design"], config["base"]
    write_json(out / "parameter_registry_snapshot.json", registry)
    write_json(out / "analysis_design_snapshot.json", design)
    baseline_analysis(out, design=design, base=base)
    case_counts = deterministic_analysis(out, design=design, base=base)
    print("Deterministic analyses complete: " + json.dumps(case_counts), flush=True)
    inputs = sample_inputs(args.max_draws, design["monte_carlo"]["seeds"][0], config=config)
    values = evaluate_draws(inputs, base=base)
    summary, methods, joint = summarise(values)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    methods.to_csv(out / "method_summary.csv", index=False, float_format="%.17g")
    write_json(out / "joint_events.json", joint)
    print("Primary summaries complete; checking q=0 reproduction and exporting targeted-RPT additions...", flush=True)
    variant_comparison(out, inputs, values, base=base)
    uniform_inputs = sample_inputs(len(inputs), design["monte_carlo"]["seeds"][0], "uniform", config=config)
    uniform_summary, _, uniform_joint = summarise(evaluate_draws(uniform_inputs, base=base))
    uniform_summary.to_csv(out / "uniform_shape_summary.csv", index=False, float_format="%.17g")
    write_json(out / "uniform_joint_events.json", uniform_joint)
    del uniform_inputs
    print(f"Exporting {len(inputs)} primary draws and paired results...", flush=True)
    export_draws(out, inputs, values, base=base)
    missing = [name for name in OUTPUT_FILES if not (out / name).is_file()]
    if missing:
        raise RuntimeError("Missing output files: " + ", ".join(missing))
    manifest = dict(completed_at_UTC=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__,
                    elapsed_seconds=time.monotonic() - started, main_seed=design["monte_carlo"]["seeds"][0],
                    main_draws=len(inputs), executed_seeds=design["monte_carlo"]["seeds"],
                    sampled_parameters=design["monte_carlo"]["sampled_parameters"],
                    case_counts=case_counts, numerical_status="fixed_primary_seed_run_no_acceptance_thresholds",
                    source_hashes=config["source_hashes"], targeted_rpt=design["targeted_rpt"],
                    execution_status="complete", base_reference_inputs=base,
                    scope="Exploratory incremental testing burdens, not profit/NPV or full life-cycle assessment",
                    primary_distribution="16 independent triangular inputs: original 15 shared; q applies only to Proposed deployment",
                    protocol_mode="user_selected_original_fixed_conditioning_times",
                    omissions=["instrument auxiliary electricity", "compute energy", "specialised EIS capital premium", "selection/handling overhead not already in channel labour", "discharge energy recovery"],
                    output_files=sorted(OUTPUT_FILES))
    try:
        write_json(pending_manifest, manifest)
        check_sources_unchanged(config["source_hashes"])
        pending_manifest.replace(out / "run_manifest.json")
    finally:
        pending_manifest.unlink(missing_ok=True)
    print("Analysis complete: " + json.dumps(dict(draws=len(inputs), seconds=manifest["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
