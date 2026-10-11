"""Independent frozen-model plus analytic-cycle audit of the exported variant."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_analysis import METHODS, METRICS, check_sources_unchanged, load_configuration, sample_inputs, summarise, write_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/runs/full'


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-9, equal_nan=True)


def old_model():
    spec = importlib.util.spec_from_file_location('frozen_verification_model', ROOT / 'baseline/tea_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_expected(frame, base, frozen):
    p = {**base, **{k: frame[k].to_numpy() for k in base if k in frame}}
    p.pop('deployment_rpt_fraction', None)
    old = frozen.evaluate(p)
    n = p['functional_mass_g'] / p['cell_mass_g']
    count = n * (1 - p['label_fraction']) * frame['deployment_rpt_fraction'].to_numpy()
    dt = count * (p['full_charge_time_h'] + p['rpt_rest_time_h'] + p['retention'] / p['discharge_C_rate'])
    de = count * p['nominal_voltage_V'] * p['nominal_capacity_Ah'] * p['retention'] / (1000 * p['charge_efficiency'])
    dc = dt * (p['labour_wage'] / (p['channels_per_tester'] * p['testers_per_operator'])
               + p['tester_purchase_price'] * (1+p['installation_fraction']) * (1+p['equipment_overhead_fraction'])
               / (p['equipment_life_years'] * 8760 * p['annual_utilisation'] * p['channels_per_tester']))
    dc += de * (p['electricity_price'] + p['grid_intensity'] * p['carbon_price'] / 1000)
    dco2 = de * p['grid_intensity']
    proposed = old['proposed']
    for key in ('time_channel_h', 'deployment_time_channel_h', 'gross_time_channel_h'):
        proposed[key] = proposed[key] + dt
    for key in ('energy_kWh', 'deployment_energy_kWh'):
        proposed[key] = proposed[key] + de
    for key in ('cost_USD', 'deployment_cost_USD'):
        proposed[key] = proposed[key] + dc
    proposed['co2_kg'] = proposed['co2_kg'] + dco2
    target = dict(targeted_rpt_cell_count=count, targeted_rpt_time_channel_h=dt,
                  targeted_rpt_energy_kWh=de, targeted_rpt_cost_USD=dc, targeted_rpt_co2_kg=dco2,
                  record_only_deployment_cell_count=n*(1-p['label_fraction'])-count)
    return old, target


def main(argv=None):
    global OUT
    from portable_io import validate_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        OUT = validate_output(args.input_dir)
    except ValueError as exc:
        parser.error(str(exc))
    marker = OUT / 'verification.json'
    marker.unlink(missing_ok=True)
    manifest = json.loads((OUT / 'run_manifest.json').read_text())
    config = load_configuration()
    check_sources_unchanged(manifest['source_hashes'])
    assert manifest['execution_status'] == 'complete'
    assert isinstance(manifest['main_draws'], int) and manifest['main_draws'] > 0
    assert len(manifest['sampled_parameters']) == 16
    frame = read_csv(OUT / 'parameter_draws.csv.gz')
    n = len(frame)
    np.testing.assert_array_equal(frame.draw_id, np.arange(n))
    regenerated = sample_inputs(n, manifest['main_seed'], config=config)
    np.testing.assert_array_equal(frame.to_numpy(), regenerated.to_numpy())
    del regenerated
    assert n == manifest['main_draws']
    # Regenerate the original stream with the frozen pre-targeting sampler.
    spec = importlib.util.spec_from_file_location('frozen_verification_runner', ROOT / 'baseline/run_analysis.py')
    original_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_runner)
    original_config = {name: json.loads((ROOT / 'baseline' / filename).read_text(encoding='utf-8'))
                       for name, filename in [('registry', '03_parameter_registry.json'), ('design', '04_analysis_design.json')]}
    original = original_runner.sample_inputs(n, manifest['main_seed'], config=original_config)
    np.testing.assert_array_equal(frame[original.columns].to_numpy(), original.to_numpy())
    del original
    print('Original 15 inputs regenerated with the frozen sampler; independent q stream reproduced.', flush=True)
    frozen = old_model()
    retained = np.full((n, len(METHODS), len(METRICS)), np.nan)
    seen = np.zeros((n, len(METHODS)), dtype=bool)
    for chunk in read_csv(OUT / 'draws.csv.gz', chunksize=100000):
        for method, rows in chunk.groupby('method', sort=False):
            ids = rows.draw_id.to_numpy(dtype=int)
            j = METHODS.index(method)
            assert not seen[ids, j].any()
            expected, _ = independent_expected(frame.iloc[ids], config['base'], frozen)
            data = expected[method]
            for key in rows.columns:
                if key in data:
                    same(rows[key], data[key])
            assert rows.feasible.all()
            assert rows.inapplicability_reason.isna().all()
            same(rows.cost_USD, rows.data_cost_USD + rows.deployment_cost_USD)
            same(rows.time_channel_h, rows.gross_time_channel_h - rows.baseline_time_channel_h)
            same(rows.energy_kWh, rows.data_energy_kWh + rows.deployment_energy_kWh)
            retained[ids, j, :] = rows[list(METRICS)].to_numpy()
            seen[ids, j] = True
    assert seen.all()
    print(f'All {n * len(METHODS):,} method ledgers match the frozen model plus analytic RPT cycle.', flush=True)
    pair_seen = np.zeros((n, 4), dtype=bool)
    for chunk in read_csv(OUT / 'paired_savings.csv.gz', chunksize=100000):
        for method, rows in chunk.groupby('comparison', sort=False):
            ids = rows.draw_id.to_numpy(dtype=int)
            j = METHODS.index(method)
            assert not pair_seen[ids, j-1].any()
            for k, metric in enumerate(METRICS):
                delta = retained[ids, j, k] - retained[ids, 0, k]
                same(rows['saving_' + metric], delta)
                same(rows['reduction_pct_' + metric], 100 * delta / retained[ids, j, k])
            pair_seen[ids, j-1] = True
    assert pair_seen.all()
    summary, methods, _ = summarise(retained)
    for name, expected in [('summary.csv', summary), ('method_summary.csv', methods)]:
        actual = read_csv(OUT / name)
        for column in expected.select_dtypes(include='number'):
            same(actual[column], expected[column])
    target_seen = np.zeros(n, dtype=bool)
    for chunk in read_csv(OUT / 'targeted_rpt_draws.csv.gz', chunksize=50000):
        ids = chunk.draw_id.to_numpy(dtype=int)
        assert not target_seen[ids].any()
        _, target = independent_expected(frame.iloc[ids], config['base'], frozen)
        for key, expected in target.items():
            same(chunk[key], expected)
        target_seen[ids] = True
    assert target_seen.all()
    for name in ('sensitivity.csv', 'two_factor.csv'):
        data = read_csv(OUT / name)
        assert np.all(data.deployment_rpt_fraction == .04)
        for (method, metric), rows in data.groupby(['comparator', 'metric']):
            expected, _ = independent_expected(rows, config['base'], frozen)
            delta = expected[method][metric] - expected['proposed'][metric]
            same(rows.saving, delta)
            same(rows.reduction_pct, 100 * delta / expected[method][metric])
    del retained
    # Independent closed forms validate the whole uniform repeat, without the new evaluator.
    uniform = sample_inputs(n, manifest['main_seed'], 'uniform', config=config)
    expected_uniform = np.empty((n, len(METHODS), len(METRICS)))
    for start in range(0, n, 20000):
        batch = uniform.iloc[start:start+20000]
        expected, _ = independent_expected(batch, config['base'], frozen)
        for j, method in enumerate(METHODS):
            expected_uniform[start:start+len(batch), j, :] = np.column_stack([expected[method][m] for m in METRICS])
    uniform_summary, _, _ = summarise(expected_uniform)
    saved_uniform = read_csv(OUT / 'uniform_shape_summary.csv')
    for col in uniform_summary.select_dtypes(include='number'):
        same(saved_uniform[col], uniform_summary[col])
    check_sources_unchanged(manifest['source_hashes'])
    report = dict(status='passed', main_draws=n, method_ledgers_checked=int(seen.sum()),
                  paired_rows_checked=int(pair_seen.sum()), uniform_draws_checked=n,
                  original_fifteen_draws_identical=True, independent_q_reproducible=True,
                  original_draw_comparison="regenerated frozen sampler; historical raw file not required",
                  all_primary_protocols_feasible=True, targeted_rpt_denominator='deployment_pool',
                  targeted_rpt_initial_soc=0, sensitivity_fixed_q=.04,
                  audit=('Frozen original model plus independently written additive-cycle formulas; all primary method, '
                         'paired and targeted-RPT raw records checked; single/two-factor savings records, principal '
                         'method/paired summaries and uniform paired summary checked. Auxiliary summary files are '
                         'not independently audited by this script.'),
                  tolerances='Floating-point identity checks only; no Monte Carlo convergence thresholds imposed.')
    write_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
