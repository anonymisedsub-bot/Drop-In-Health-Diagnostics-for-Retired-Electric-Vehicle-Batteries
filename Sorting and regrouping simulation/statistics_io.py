"""Input integrity checks and frozen batch-summary arithmetic."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pack_core import summarize_values
from strategy_config import NAMES, IDS

HERE=Path(__file__).resolve().parent
METRICS = ['cap_mean','sd_mean','n_good_packs','pass_rate','yield_rate','rejection_rate',
           'rejection_precision','bad_cell_recall','good_cell_false_rejection_rate',
           'n_packs','n_accepted','n_leftover','n_unique_sources','utilization_rate',
           'n_rejected_true_bad','n_rejected_true_good','n_retained_true_bad',
           'n_additional_rejected_true_bad','n_additional_rejected_true_good']
META = ['condition','tail','iteration','seed','pool_usage_ratio','n_train_total']
METRICS += ['n_quadrant_high_high','n_rejected_quadrant','n_rejected_quadrant_true_bad','n_rejected_quadrant_true_good','quadrant_rejection_fraction']
from rpt_core import RPT_METRICS
METRICS += RPT_METRICS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_inputs():
    manifest = json.loads((HERE/'inputs/manifest.json').read_text(encoding='utf-8'))
    conditions = []
    data = {}
    input_hashes = {}
    for c in manifest['conditions']:
        path = HERE/'inputs'/c['prepared_path']
        actual = sha(path)
        if actual != c['prepared_sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        d = pd.read_csv(path).sort_values('cell_id',kind='stable').reset_index(drop=True)
        assert len(d)==609 and d.cell_id.nunique()==609
        info = {k:c[k] for k in META if k!='n_train_total'}
        info['n_train_total'] = int(c['n_train_total'])
        conditions.append(info)
        data[c['condition']] = d
        input_hashes[path.relative_to(HERE).as_posix()] = actual
    assert len(conditions)==27
    for seed in sorted({c['seed'] for c in conditions}):
        ds = [data[c['condition']] for c in conditions if c['seed']==seed]
        for d in ds[1:]:
            np.testing.assert_array_equal(d.cell_id,ds[0].cell_id)
            np.testing.assert_allclose(d.y_true,ds[0].y_true,rtol=0,atol=1e-12)
    return conditions,data,input_hashes

def summaries(frame, groups, metrics, mixture=False):
    rows=[]
    for key,g in frame.groupby(groups,sort=True):
        if not isinstance(key,tuple):
            key=(key,)
        meta=dict(zip(groups,key))
        for metric in metrics:
            s=summarize_values(g[metric])
            if mixture:
                s['mc_mean_ci_low']=s['mc_mean_ci_high']=np.nan
            rows.append(dict(**meta,metric=metric,n_total=len(g),**s,
                pooled_pack_pass_rate=float(g.n_good_packs.sum()/g.n_packs.sum())
                    if 'n_packs' in g and g.n_packs.sum() else np.nan,
                summary_scope='three_fixed_seed_mixture' if mixture else 'fixed_model_fixed_pool',
                range_type='central_95_percent_random_batch_range',
                ci_scope='not_provided_for_seed_mixture' if mixture else 'conditional_MC_mean_only'))
    return pd.DataFrame(rows)

