"""Reproducible deterministic sensitivity and paired Monte Carlo for TEA R1."""

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


ROOT = Path(__file__).resolve().parent
COMPARATORS = METHODS[1:]
QUANTILES = [0.025, 0.5, 0.975]
OUTPUT_FILES = (
    "parameter_registry_snapshot.json", "analysis_design_snapshot.json",
    "baseline_reconciliation.json", "reference_baselines.csv",
    "sensitivity_methods.csv", "sensitivity.csv", "two_factor.csv",
    "convergence.json", "summary.csv", "method_summary.csv", "joint_events.json",
    "uniform_shape_summary.csv", "uniform_joint_events.json",
    "parameter_draws.csv.gz", "draws.csv.gz", "paired_savings.csv.gz",
)


def _source_bytes():
    paths = [ROOT / name for name in ("03_parameter_registry.json", "04_analysis_design.json",
                                    "tea_model.py", "run_analysis.py", "08_execution_protocol.md")]
    paths.extend(sorted((ROOT.parent / "TEA_results").glob("*.m")))
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
    lows, modes, highs = (np.array([parameters[k][field] for k in names]) for field in ("low", "mode", "high"))
    rng = np.random.default_rng(seed)
    if family == "triangular":
        samples = rng.triangular(lows, modes, highs, size=(n, len(names)))
    elif family == "uniform":
        samples = rng.uniform(lows, highs, size=(n, len(names)))
    else:
        raise ValueError("Unknown distribution family")
    frame = pd.DataFrame(samples, columns=names)
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
            rows.append({"label_fraction": budget, "method": method,
                         **{k: (bool(v[0]) if k == "feasible" else float(v[0])) for k, v in values.items()}})
    pd.DataFrame(rows).to_csv(out / "reference_baselines.csv", index=False, float_format="%.17g")
    result = dict(B0_submitted_totals=design["baselines"][0]["expected_cost_USD_per_tonne"],
                  B1_costs={m: float(b1[m]["cost_USD"][0]) for m in METHODS},
                  B1_errors_USD=errors, B1_passed=True,
                  B1_reference_inputs=b1_inputs, B2_reference_inputs=base,
                  B2_equipment_rate_USD_per_channel_h=float(equipment_hourly_rate(base)),
                  B2_charge_efficiency=base["charge_efficiency"],
                  protocol_mode="user_selected_original_fixed_conditioning_times",
                  B0_workbook_verification="See source_audit.json; expected values above are not a substitute for workbook audit")
    write_json(out / "baseline_reconciliation.json", result)


def _signature(summary, method_summary):
    signature = {}
    for row in summary.to_dict("records"):
        for column in ("saving_q025", "saving_median", "saving_q975", "reduction_q025_pct", "reduction_median_pct", "reduction_q975_pct"):
            signature[f"paired:{row['comparison']}:{row['metric']}:{column}"] = row[column]
    for row in method_summary.to_dict("records"):
        for column in ("q025", "median", "q975"):
            signature[f"method:{row['method']}:{row['metric']}:{column}"] = row[column]
    return signature


def _max_changes(keys, changes):
    groups = {"cost_USD": [], "energy_kWh": [], "co2_kg": [], "relative_percentage_points": []}
    for key, value in zip(keys, changes):
        group = "relative_percentage_points" if key.endswith("_pct") else key.split(":")[2]
        groups[group].append(float(value))
    return {group: max(values) for group, values in groups.items()}


def convergence_analysis(max_draws, *, config):
    design = config["design"]
    spec = design["monte_carlo"]
    raw = spec["convergence_targets"]
    tolerance = dict(cost_USD=raw["cost_quantile_absolute_change_USD_per_tonne"],
                     energy_kWh=raw["energy_quantile_absolute_change_kWh_per_tonne"],
                     co2_kg=raw["co2_quantile_absolute_change_kg_per_tonne"],
                     relative_percentage_points=raw["relative_reduction_quantile_change_percentage_points"])
    levels = [spec["pilot_draws"], spec["first_full_draws"]]
    n = spec["next_convergence_draws"]
    while n <= max_draws:
        levels.append(n)
        n *= 2
    levels = sorted(set(v for v in levels if v <= max_draws))
    if not levels or levels[-1] != max_draws:
        raise ValueError("max_draws must be 10000, 50000, or a doubling of 100000")
    records, previous = [], None
    main_inputs = main_values = None
    for n in levels:
        signatures = []
        for i, seed in enumerate(spec["seeds"]):
            inputs = sample_inputs(n, seed, config=config)
            values = evaluate_draws(inputs, base=config["base"])
            summary, methods, _ = summarise(values)
            signatures.append(_signature(summary, methods))
            if i == 0:
                main_inputs, main_values = inputs, values
        keys = sorted(signatures[0])
        matrix = np.array([[s[key] for key in keys] for s in signatures])
        seed_changes = _max_changes(keys, np.ptp(matrix, axis=0))
        size_changes = _max_changes(keys, np.max(np.abs(matrix - previous), axis=0)) if previous is not None else None
        passed = n >= spec["next_convergence_draws"] and size_changes is not None and all(
            seed_changes[k] <= tolerance[k] and size_changes[k] <= tolerance[k] for k in tolerance)
        record = dict(n_draws=n, seeds=spec["seeds"], seed_max_quantile_range=seed_changes,
                      sample_size_max_quantile_change=size_changes, passed=bool(passed))
        records.append(record)
        print("Convergence " + json.dumps(record), flush=True)
        previous = matrix
        if passed:
            break
    return main_inputs, main_values, dict(tolerances=tolerance, history=records,
        final_draws=len(main_inputs), passed=records[-1]["passed"],
        status="targets_met" if records[-1]["passed"] else "strict_targets_not_met_at_execution_cap",
        interpretation="Monte Carlo numerical stability only; not validation of parameter distributions or industrial performance")


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    started = time.monotonic()
    out = args.output_dir.resolve()
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
    inputs, values, convergence = convergence_analysis(args.max_draws, config=config)
    write_json(out / "convergence.json", convergence)
    summary, methods, joint = summarise(values)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    methods.to_csv(out / "method_summary.csv", index=False, float_format="%.17g")
    write_json(out / "joint_events.json", joint)
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
                    main_draws=len(inputs), convergence_seeds=design["monte_carlo"]["seeds"],
                    sampled_parameters=design["monte_carlo"]["sampled_parameters"],
                    case_counts=case_counts, numerical_convergence_passed=convergence["passed"],
                    numerical_status=convergence["status"], source_hashes=config["source_hashes"],
                    execution_status="complete", base_reference_inputs=base,
                    scope="Exploratory incremental testing burdens, not profit/NPV or full life-cycle assessment",
                    primary_distribution="15 independent triangular free inputs; shared across methods",
                    protocol_mode="user_selected_original_fixed_conditioning_times",
                    omissions=["instrument auxiliary electricity", "compute energy", "specialised EIS capital premium", "independent deployment validation labels", "discharge energy recovery"],
                    output_files=sorted(OUTPUT_FILES))
    try:
        write_json(pending_manifest, manifest)
        check_sources_unchanged(config["source_hashes"])
        pending_manifest.replace(out / "run_manifest.json")
    finally:
        pending_manifest.unlink(missing_ok=True)
    print("Analysis complete: " + json.dumps(dict(draws=len(inputs), converged=convergence["passed"], seconds=manifest["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
