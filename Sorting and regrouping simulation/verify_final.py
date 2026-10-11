"""Verify standalone final inputs, frozen results, metric identities and figure sources."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from strategy_config import METHODS,NAMES,IDS
from statistics_io import load_inputs,sha

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/reference'
FIGURES=HERE/'visualization/outputs'


def content_sha(path):
    data=Path(path).read_bytes()
    if str(path).endswith('.gz'):
        data=gzip.decompress(data)
    return hashlib.sha256(data).hexdigest()


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)


def run(write_report=False):
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['release']=='FINAL' and manifest['method_rows']==135000
    assert manifest['repetitions_per_model']==1000 and manifest['paired_rows']==270000
    for p,h in manifest['input_sha256'].items():
        assert sha(HERE/p)==h,p
    bundled_reference=OUT.resolve()==(HERE/'results/reference').resolve()
    # Historical source hashes remain in the frozen run manifest unchanged.
    # Its original code is archived outside this runtime package, so verify the
    # available current source against the public inventory without claiming to
    # verify the historical implementation. New runs retain their source binding.
    if bundled_reference:
        inventory=json.loads((HERE/'artifact_manifest.json').read_text(encoding='utf-8'))
        source_hashes={p:h for p,h in inventory['files'].items() if Path(p).suffix=='.py'}
        current_sources={p.relative_to(HERE).as_posix() for p in HERE.rglob('*.py')
            if not {'runs','__pycache__','.pytest_cache'}.intersection(p.relative_to(HERE).parts)}
        source_manifest='artifact_manifest.json'
    else:
        source_hashes=manifest['source_sha256']
        current_sources={p.name for p in HERE.glob('*.py')}
        source_manifest='run_manifest.json'
    assert set(source_hashes)==current_sources, 'Current source inventory is incomplete or stale'
    for p,h in source_hashes.items():
        assert sha(HERE/p)==h,f'Current source hash mismatch: {p}'
    inputs=json.loads((HERE/'inputs/manifest.json').read_text(encoding='utf-8'))
    assert sha(HERE/'inputs/model_inventory.csv')==inputs['inventory_sha256']
    conditions,data,_=load_inputs()
    for seed in range(3):
        path=HERE/'inputs/draws'/f'seed_{seed}.npz'
        assert sha(path)==inputs['draw_sha256'][path.name]
        with np.load(path) as stored:
            expected=np.random.default_rng(np.random.SeedSequence([20260924,seed,3000924])).integers(
                0,609,size=(1000,300),dtype=np.int32)
            np.testing.assert_array_equal(stored['draw_indices'],expected)
            np.testing.assert_array_equal(stored['sample_instance_ids'],np.arange(300))
            for c in conditions:
                if c['seed']==seed:
                    np.testing.assert_array_equal(stored['source_cell_ids'],data[c['condition']].cell_id)
    assert (FIGURES/'panel_manifest.json').is_file(), 'Figure 7 must be rendered before verification'
    frozen=json.loads((HERE/'provenance/final_numeric_fingerprint.json').read_text(encoding='utf-8'))
    checked_frozen=0
    for name,h in frozen['uncompressed_sha256'].items():
        if 'figures' in Path(name).parts:
            continue  # Superseded plot-only tables; raw frozen results are checked below.
        checked_frozen+=1
        assert content_sha(OUT/name)==h,f'Frozen result mismatch: {name}'
    raw=pd.read_csv(OUT/'batch_metrics.csv.gz')
    paired=pd.read_csv(OUT/'paired_batch_effects.csv.gz')
    assert len(raw)==135000 and set(raw.method)==set(METHODS) and raw.condition.nunique()==27
    assert raw.groupby(['condition','method']).size().eq(1000).all()
    assert raw.groupby(['condition','batch_id']).size().eq(5).all()
    assert len(paired)==270000 and paired.comparison.nunique()==10
    assert paired.groupby(['condition','comparison']).size().eq(1000).all()
    assert not raw.duplicated(['condition','batch_id','method']).any()
    assert not paired.duplicated(['condition','batch_id','comparison']).any()
    assert (raw.n_rejected+raw.n_used+raw.n_leftover).eq(300).all()
    assert raw.n_leftover.between(0,7).all() and raw.n_used.eq(8*raw.n_packs).all()
    assert (raw.n_rejected_capacity+raw.n_rejected_strategy).eq(raw.n_rejected).all()
    assert (raw.n_rejected_true_bad+raw.n_rejected_true_good).eq(raw.n_rejected).all()
    assert raw.n_good_packs.le(raw.n_packs).all()
    for key,num,den in [('yield_rate','n_good_packs',37),('pass_rate','n_good_packs','n_packs'),
        ('rejection_rate','n_rejected',300),('rejection_precision','n_rejected_true_bad','n_rejected'),
        ('bad_cell_recall','n_rejected_true_bad','n_true_bad'),
        ('good_cell_false_rejection_rate','n_rejected_true_good','n_true_good'),('rpt_rate','n_rpt',300)]:
        close(raw[key],raw[num]/(raw[den].replace(0,np.nan) if isinstance(den,str) else den))
    c=raw.loc[raw.method.eq('quadrant_c')].set_index(['condition','batch_id'])
    a=raw.loc[raw.method.eq('quadrant_a')].set_index(['condition','batch_id']).reindex(c.index)
    assert c.n_rpt.eq(a.n_rejected_quadrant).all()
    assert c.n_rpt_true_good.eq(a.n_rejected_quadrant_true_good).all()
    assert c.n_rpt_true_bad.eq(a.n_rejected_quadrant_true_bad).all()
    assert c.n_accepted.eq(a.n_accepted+c.n_rpt_true_good).all()
    assert c.n_retained_true_bad.eq(a.n_retained_true_bad).all()
    assert raw.loc[raw.method.ne('quadrant_c'),'n_rpt'].eq(0).all()
    summary=pd.read_csv(OUT/'summary_by_model.csv')
    mixture=pd.read_csv(OUT/'summary_by_pool_and_vtail.csv')
    for frame in (raw,summary,mixture):
        assert frame.strategy_name.eq(frame.method.map(NAMES)).all()
        assert frame.strategy_id.eq(frame.method.map(IDS)).all()
    assert mixture.mc_mean_ci_low.isna().all() and mixture.mc_mean_ci_high.isna().all()
    examples=pd.read_csv(OUT/'example_batch0_assignments.csv.gz')
    assert len(examples)==40500
    assert not examples.duplicated(['condition','method','sample_instance_id']).any()
    import sys
    sys.path.insert(0,str(HERE/'visualization'))
    from render_figure7 import verify as verify_figure7
    verify_figure7(OUT,FIGURES)
    pngs=list(FIGURES.glob('*.png'))
    assert len(pngs)==13, 'Expected twelve Figure 7 panels plus one overview'
    assert not list(FIGURES.glob('*.svg')) and not list(FIGURES.glob('*.pdf'))
    banned=['capacity-plus-uncertainty','Seven sorting strategies','Strategy C','Targeted RPT C','S7:',
        'Six sorting strategies','Quadrant B','Quadrant A','A + targeted RPT','RPT-assisted']
    text_sources=[HERE/'visualization/FIGURE_7_CAPTION.md']
    if not bundled_reference:
        text_sources.append(OUT/'RESULTS_REPORT.md')
    for p in text_sources:
        txt=p.read_text(encoding='utf-8')
        assert not any(s in txt for s in banned),p
        assert 'targeted RPT' in txt,p
    result=dict(status='PASS',standalone=True,release='FINAL',raw_rows=135000,paired_rows=270000,
        methods=5,comparisons=10,frozen_result_files_checked=checked_frozen,
        input_conditions=27,random_draws_verified=True,metric_identities_verified=True,
        figure_sets=1,figure_exports=13,figure_panels=12,removed_method_absent_from_results_and_figures=True,
        current_source_verified=True,current_source_manifest=source_manifest,
        current_source_files_checked=len(source_hashes),historical_source_verified=False,
        historical_source_scope='archived_outside_runtime_package' if bundled_reference else 'not_applicable_to_new_run',
        results_report_checked=not bundled_reference)
    if write_report:
        (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (OUT/'artifact_manifest.json').write_text(json.dumps(dict(status='VERIFIED',verification=result),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
