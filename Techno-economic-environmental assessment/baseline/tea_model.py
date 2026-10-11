"""Vectorised incremental TEA with explicitly retained conditioning-time approximations."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("proposed", "partial_charge", "rpt", "pulse", "eis")
METRICS = ("cost_USD", "energy_kWh", "co2_kg")


def reference_inputs():
    registry = json.loads((ROOT / "03_parameter_registry.json").read_text(encoding="utf-8"))
    return {
        p["id"]: p.get("proposal_base", p.get("value"))
        for p in registry["parameters"]
        if "proposal_base" in p or "value" in p
    }


def equipment_hourly_rate(p):
    return (
        p["tester_purchase_price"] * (1 + p["installation_fraction"])
        * (1 + p["equipment_overhead_fraction"])
        / (p["equipment_life_years"] * 8760 * p["annual_utilisation"] * p["channels_per_tester"])
    )


def _broadcast_and_validate(inputs):
    keys = list(inputs)
    arrays = np.broadcast_arrays(*(np.atleast_1d(np.asarray(inputs[k], dtype=float)) for k in keys))
    p = dict(zip(keys, arrays))
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Inputs must be finite scalar or one-dimensional arrays")
    for name in ("functional_mass_g", "cell_mass_g", "nominal_capacity_Ah", "nominal_voltage_V",
                 "discharge_C_rate", "testers_per_operator", "channels_per_tester", "equipment_life_years"):
        if np.any(p[name] <= 0):
            raise ValueError(f"{name} must be positive")
    for name in ("retention", "charge_efficiency", "annual_utilisation"):
        if np.any((p[name] <= 0) | (p[name] > 1)):
            raise ValueError(f"{name} must be in (0, 1]")
    for name in ("initial_soc", "label_fraction", "partial_charge_soc_increment", "pulse_final_soc"):
        if np.any((p[name] < 0) | (p[name] > 1)):
            raise ValueError(f"{name} must be in [0, 1]")
    if any(np.any(v < 0) for v in arrays):
        raise ValueError("Negative physical/economic inputs are unsupported")
    if np.any(p["auxiliary_power_channel_kW"] != 0) or np.any(p["rest_channel_occupancy_fraction"] != 1):
        raise ValueError("Only the selected zero-auxiliary, on-tester-rest structure is implemented")
    if np.any(p["soc_point_count"] != 5):
        raise ValueError("Only the submitted five-SOC-point protocols are implemented")
    return p


def evaluate(inputs, protocol_mode="fixed_conditioning", equipment_rate=None):
    """Return per-functional-unit stage ledgers; invalid methods are NaN, not zero.

    The user-selected mode preserves original fixed SOC-conditioning durations.
    Full charge, arrival charge, baseline discharge and full discharge follow
    the revised SOC/retention formulas. Discharge energy has no export credit.
    """
    if protocol_mode != "fixed_conditioning":
        raise ValueError("Only user-selected fixed_conditioning is implemented")
    p = _broadcast_and_validate(inputs)
    n = p["functional_mass_g"] / p["cell_mass_g"]
    nl, nu = n * p["label_fraction"], n * (1 - p["label_fraction"])
    r, s = p["retention"], p["initial_soc"]
    energy = p["nominal_capacity_Ah"] * p["nominal_voltage_V"] * r / 1000
    full_discharge = r / p["discharge_C_rate"]
    baseline = s * full_discharge
    charge = (1 - s) * p["full_charge_time_h"]
    rest = p["rpt_rest_time_h"]
    label_gross = charge + rest + full_discharge
    relaxation = p["multi_soc_relaxation_h"]
    count = p["soc_point_count"]
    pulse_time = p["pulse_deployment_time_h"]
    zero = np.zeros_like(n)
    rate = equipment_hourly_rate(p) if equipment_rate is None else np.broadcast_to(equipment_rate, n.shape)
    if np.any(~np.isfinite(rate)) or np.any(rate < 0):
        raise ValueError("Equipment rate must be finite and nonnegative")
    labour_rate = p["labour_wage"] / (p["channels_per_tester"] * p["testers_per_operator"])

    pulse_gross = (
        baseline + 2 * rest + count * (p["pulse_soc_conditioning_time_h"] + relaxation + pulse_time)
        + (1 - p["pulse_final_soc"]) * p["full_charge_time_h"] + full_discharge
    )
    eis_gross = (
        charge + 3 * rest + full_discharge + p["full_charge_time_h"]
        + count * (p["eis_soc_conditioning_time_h"] + relaxation + p["eis_dataset_measurement_time_h"])
    )
    delta = p["partial_charge_soc_increment"]
    # Each tuple carries data/deployment gross time, counts, charge and discharge energy.
    workflows = {
        "proposed": (nl * label_gross, nu * baseline, nl, nu,
                     nl * (1 - s) * energy, zero, nl * energy + nu * s * energy),
        "partial_charge": (nl * label_gross,
                           nu * (baseline + p["partial_charge_time_h"] + p["partial_discharge_extra_time_h"]), nl, nu,
                           nl * (1 - s) * energy, nu * delta * energy, nl * energy + nu * (s + delta) * energy),
        "rpt": (zero, n * label_gross, zero, n, zero, n * (1 - s) * energy, n * energy),
        "pulse": (nl * pulse_gross, nu * (baseline + pulse_time), nl, nu,
                  nl * energy, zero, nl * (1 + s) * energy + nu * s * energy),
        "eis": (nl * eis_gross, nu * (baseline + p["eis_deployment_time_h"]), nl, nu,
                nl * (2 - s) * energy, zero, nl * 2 * energy + nu * s * energy),
    }
    result = {}
    for method, (data_gross, dep_gross, data_count, dep_count, data_stored, dep_stored, discharged) in workflows.items():
        data_base, dep_base = data_count * baseline, dep_count * baseline
        data_time, dep_time = data_gross - data_base, dep_gross - dep_base
        data_energy, dep_energy = data_stored / p["charge_efficiency"], dep_stored / p["charge_efficiency"]
        time = data_time + dep_time
        grid_energy = data_energy + dep_energy
        carbon = grid_energy * p["grid_intensity"]
        energy_rate = p["electricity_price"] + p["grid_intensity"] * p["carbon_price"] / 1000
        values = {
            "time_channel_h": time,
            "energy_kWh": grid_energy,
            "co2_kg": carbon,
            "labour_cost_USD": time * labour_rate,
            "equipment_cost_USD": time * rate,
            "electricity_cost_USD": grid_energy * p["electricity_price"],
            "carbon_cost_USD": carbon * p["carbon_price"] / 1000,
            "data_time_channel_h": data_time,
            "deployment_time_channel_h": dep_time,
            "data_energy_kWh": data_energy,
            "deployment_energy_kWh": dep_energy,
            "data_co2_kg": data_energy * p["grid_intensity"],
            "deployment_co2_kg": dep_energy * p["grid_intensity"],
            "data_cost_USD": data_time * (labour_rate + rate) + data_energy * energy_rate,
            "deployment_cost_USD": dep_time * (labour_rate + rate) + dep_energy * energy_rate,
            "data_gross_time_channel_h": data_gross,
            "data_baseline_time_channel_h": data_base,
            "deployment_gross_time_channel_h": dep_gross,
            "deployment_baseline_time_channel_h": dep_base,
            "gross_time_channel_h": data_gross + dep_gross,
            "baseline_time_channel_h": data_base + dep_base,
            "charge_stored_kWh": data_stored + dep_stored,
            "initial_stored_kWh": n * s * energy,
            "discharged_kWh": discharged,
            "data_cell_count": data_count,
            "deployment_cell_count": dep_count,
        }
        values["cost_USD"] = values["data_cost_USD"] + values["deployment_cost_USD"]
        feasible = ((s + delta <= 1 + 1e-12) | (nu == 0)) if method == "partial_charge" else np.ones(n.shape, dtype=bool)
        result[method] = {key: np.where(feasible, value, np.nan) for key, value in values.items()}
        result[method]["feasible"] = feasible
    return result
