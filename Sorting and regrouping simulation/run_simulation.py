"""Reproduce all five strategies using local fixed-model predictions and paired draws."""
from pathlib import Path
from datetime import datetime, timezone
import platform
import time
import numpy as np
import pandas as pd
import scipy
from strategy_config import METHODS, NAMES, IDS
from pack_core import Features, draw_indices
from sorting_core import select_policy, evaluate_policy
from rpt_core import plan_rpt, select_rpt, evaluate_rpt, no_rpt_metrics
from statistics_io import META, METRICS, load_inputs, summaries, sha, write_json
from result_tables import paired_effects, example_rows, c_example

HERE = Path(__file__).resolve().parent
OUT = HERE/'results/runs/full'


def run():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    # Invalidate any previous verification before beginning a new run.
    write_json(OUT/'verification.json', {'status': 'PENDING'})
    write_json(OUT/'run_manifest.json', {'status': 'RUNNING'})
    write_json(OUT/'artifact_manifest.json', {'status': 'PENDING'})
    conditions, data, input_hashes = load_inputs()
    draws = {}
    for seed in range(3):
        path = HERE/'inputs/draws'/f'seed_{seed}.npz'
        with np.load(path) as stored:
            idx = stored['draw_indices'].copy()
            assert idx.shape == (1000,300)
            np.testing.assert_array_equal(idx, draw_indices(609,300,1000,seed,20260924))
            np.testing.assert_array_equal(stored['sample_instance_ids'],np.arange(300))
            for c in conditions:
                if c['seed'] == seed:
                    np.testing.assert_array_equal(stored['source_cell_ids'],data[c['condition']].cell_id)
        draws[seed] = idx
        input_hashes[path.relative_to(HERE).as_posix()] = sha(path)
    input_hashes['inputs/manifest.json'] = sha(HERE/'inputs/manifest.json')
    input_hashes['inputs/model_inventory.csv'] = sha(HERE/'inputs/model_inventory.csv')
    frames, examples = [], []
    for meta in conditions:
        d = data[meta['condition']]
        source,pred,lower,upper,truth = (d[k].to_numpy() for k in ['cell_id','y_pred','lower','upper','y_true'])
        rows = []
        for batch_id,idx in enumerate(draws[meta['seed']]):
            f = Features(source[idx],pred[idx],lower[idx],upper[idx])
            y = truth[idx]
            selections = {m:select_policy(f,m) for m in METHODS[:-1]}
            plan = plan_rpt(f)
            rpt = select_rpt(f,y[plan['positions']],plan)
            for method in METHODS:
                scores = (evaluate_rpt(f,y,rpt) if method == 'quadrant_c' else
                    dict(**evaluate_policy(f,y,method,selections[method]),**no_rpt_metrics()))
                rows.append(dict(**meta,batch_id=batch_id,method=method,
                    strategy_name=NAMES[method],strategy_id=IDS[method],**scores))
            if batch_id == 0:
                batch_examples = example_rows(meta,d,idx,selections)
                for row in batch_examples:
                    row.update(rpt_selected=False,rpt_measurement=np.nan,
                        available_capacity=row['y_pred'],capacity_source='prediction')
                examples.extend(batch_examples)
                examples.extend(c_example(meta,d,idx,rpt))
        frames.append(pd.DataFrame(rows))
        print(f'{meta["condition"]}: 1000 paired batches x 5 strategies',flush=True)
    raw = pd.concat(frames,ignore_index=True)
    raw.to_csv(OUT/'batch_metrics.csv.gz',index=False)
    pd.DataFrame(examples).to_csv(OUT/'example_batch0_assignments.csv.gz',index=False)
    summary = summaries(raw,META+['method'],METRICS)
    mixture = summaries(raw,['pool_usage_ratio','n_train_total','tail','method'],METRICS,mixture=True)
    for frame in (summary,mixture):
        frame['strategy_name'] = frame.method.map(NAMES)
        frame['strategy_id'] = frame.method.map(IDS)
    summary.to_csv(OUT/'summary_by_model.csv',index=False)
    mixture.to_csv(OUT/'summary_by_pool_and_vtail.csv',index=False)
    paired = paired_effects(raw)
    paired.to_csv(OUT/'paired_batch_effects.csv.gz',index=False)
    metrics = [k for k in paired if k not in META+['batch_id','comparison']]
    summaries(paired,META+['comparison'],metrics).to_csv(OUT/'paired_summary_by_model.csv',index=False)
    summaries(paired,['pool_usage_ratio','n_train_total','tail','comparison'],metrics,mixture=True).to_csv(
        OUT/'paired_summary_by_pool_and_vtail.csv',index=False)
    assert all(sha(HERE/p)==h for p,h in input_hashes.items())
    manifest = dict(status='SIMULATED',release='FINAL',created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_type='simulation_only',ground_truth_type='measured_cell_labels',pack_outcome_type='minimum_cell_proxy',
        physical_pack_validation=False,rpt_type='simulated exact reveal of selected measured labels; zero added measurement noise',
        n_models=27,repetitions_per_model=1000,batch_size=300,source_pool_size=609,cells_per_pack=8,
        model_batches=27000,method_rows=len(raw),paired_rows=len(paired),methods=list(METHODS),strategy_names=NAMES,
        rpt_selection='high prediction>=14, width>all-batch median; top ceil(h/5); exactly uncertainty-guided rejection candidates',
        rpt_measurement_scope='selected occurrences only; no transfer to unselected duplicate source IDs',
        rpt_cost_unit='selected statistical occurrences; unique selected source count also recorded',
        rpt_acceptance='prediction>=14 AND (not selected for RPT OR measured capacity>=14)',
        rpt_grouping='available capacity (RPT if tested else prediction), source ID, occurrence; consecutive groups of8',
        input_sha256=input_hashes,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        dependencies=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        population_confidence_interval=False,seconds=round(time.perf_counter()-started,2))
    write_json(OUT/'run_manifest.json',manifest)
    print(f'Reproduced {len(raw)} result rows and {len(paired)} paired rows in {time.perf_counter()-started:.1f}s.',flush=True)


if __name__ == '__main__':
    run()
