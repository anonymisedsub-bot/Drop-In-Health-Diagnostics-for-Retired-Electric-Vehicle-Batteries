"""Build RPT summary tables and a report from reproduced five-strategy outputs."""
from pathlib import Path
import json
from portable_io import path_reference
import pandas as pd
from strategy_config import METHODS,NAMES,IDS

HERE=Path(__file__).resolve().parent; OUT=HERE/'results/runs/full'
FIGURES=HERE/'visualization/outputs/runs/full'
FILES=['batch_metrics.csv.gz','summary_by_model.csv','summary_by_pool_and_vtail.csv',
    'paired_batch_effects.csv.gz','paired_summary_by_model.csv','paired_summary_by_pool_and_vtail.csv',
    'example_batch0_assignments.csv.gz']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def run():
    data={name:pd.read_csv(OUT/name) for name in FILES}
    raw=data['batch_metrics.csv.gz']; summary=data['summary_by_model.csv']; mixture=data['summary_by_pool_and_vtail.csv']
    paired=data['paired_summary_by_pool_and_vtail.csv']
    assert len(raw)==135000 and len(data['paired_batch_effects.csv.gz'])==270000
    means=mixture.pivot(index=['n_train_total','tail','method'],columns='metric',values='mean')
    overview=[]; directions=[]
    for baseline in METHODS[:-1]:
        delta=means.xs('quadrant_c',level='method')-means.xs(baseline,level='method')
        directions.append(['Uncertainty-guided sorting + targeted RPT vs '+IDS[baseline]]+[
            f'{int((delta[k]>1e-10).sum())}/{int((abs(delta[k])<=1e-10).sum())}/{int((delta[k]<-1e-10).sum())}'
            for k in ['pass_rate','yield_rate','sd_mean']])
        for (n,tail),d in delta.iterrows():
            overview.append(dict(n_train_total=n,tail=tail,method='quadrant_c',baseline=baseline,
                pass_rate_difference_pp=100*d.pass_rate,yield_rate_difference_pp=100*d.yield_rate,
                rejection_rate_difference_pp=100*d.rejection_rate,n_good_packs_difference=d.n_good_packs,
                cap_mean_difference=d.cap_mean,sd_mean_difference=d.sd_mean))
    pd.DataFrame(overview).to_csv(OUT/'RPT_comparison_overview.csv',index=False)
    costs=means.xs('quadrant_c',level='method')[['n_rpt','rpt_rate','n_unique_rpt_sources','n_rpt_true_good',
        'n_rpt_true_bad','n_rpt_retained_used','n_rpt_good_leftover','rpt_good_fraction']]
    costs.to_csv(OUT/'RPT_workload.csv')
    report=['# Five-strategy comparison: uncertainty-guided sorting with and without targeted RPT',
        'The five strategies are capacity-estimate-only sorting, conservative sorting, lower-bound sorting, '
        'Uncertainty-guided sorting, and Uncertainty-guided sorting + targeted RPT. The last strategy adds targeted retesting to uncertainty-guided sorting.',
        'This reproduction runs all five strategies from packaged diagnostic records and fixed sampling indices, preserving the selected numerical results. '
        'There are 27 fixed models × 1,000 batches per model × five strategies = 135,000 strategy-batch records; ten pairwise comparisons yield 270,000 paired records. '
        'Each batch samples 300 occurrences with replacement from 609 test records. Each pool-usage-ratio/Vtail setting combines three seeds and 3,000 batches per strategy.',
        '## Method names and rules',table(['ID','Name','Rule'],[
            [IDS['point'],NAMES['point'],'Predicted capacity ≥14 Ah'],[IDS['conservative'],NAMES['conservative'],'Predicted capacity >14.3 Ah'],
            [IDS['lower_bound'],NAMES['lower_bound'],'Predicted lower endpoint ≥14 Ah and predicted capacity ≥14 Ah'],
            [IDS['quadrant_a'],NAMES['quadrant_a'],'Split at 14 Ah and the full-batch median uncertainty; reject low-capacity records and the highest-uncertainty 20% of the high/high quadrant'],
            [IDS['quadrant_c'],NAMES['quadrant_c'],'Send the highest-uncertainty 20% of the high/high quadrant, otherwise rejected by uncertainty-guided sorting, to RPT; retain or reject according to measured capacity']]),
        '## How uncertainty-guided sorting + targeted RPT works',
        'High capacity means a prediction ≥14 Ah. High uncertainty means an interval width greater than the median over all 300 batch occurrences. '
        'For h occurrences in the high/high quadrant, retest the ceil(h/5) occurrences with the largest widths. '
        'Retain measured capacity ≥14 Ah and reject <14 Ah. Retested and retained records use measured capacity for sorting; all other retained records use predicted capacity. '
        'Sort by capacity, original cell ID and occurrence position, then form packs from consecutive groups of eight. '
        'Records with predicted capacity <14 Ah that are not selected for retesting remain rejected.',
        'RPT simulates exact measurement by revealing an existing measured label only for a preselected occurrence. '
        'It adds measurement information and workload; no new physical RPT was performed.',
        '## Directions of mean changes across nine settings',table(['Comparison','Pack pass rate up/tied/down','Effective yield up/tied/down','Within-pack SD up/tied/down'],directions),
        'Uncertainty-guided sorting + targeted RPT has a higher effective yield than each other strategy in all nine settings. '
        'Against capacity-estimate-only, conservative and lower-bound sorting, pack pass rate increases and within-pack SD decreases in every setting. '
        'Against uncertainty-guided sorting, pack pass rate increases in eight settings and decreases in one. '
        'These are descriptive mean directions for fixed models and a fixed test pool, not population-level significance claims.',
        '## Targeted-RPT workload',table(['Training cells','Vtail','RPT occurrences/batch','Share of batch/%','Measured-good retained/batch','Measured-bad rejected/batch'],[
            [n,t,f'{r.n_rpt:.3f}',f'{100*r.rpt_rate:.3f}',f'{r.n_rpt_true_good:.3f}',f'{r.n_rpt_true_bad:.3f}']
            for (n,t),r in costs.iterrows()]),
        'Each batch has 300 occurrences; the RPT fraction is the retested occurrence count divided by 300. '
        'The 20% selection fraction uses the high-capacity/high-uncertainty quadrant count as its denominator and rounds upward.',
        f'![Uncertainty-guided sorting with and without targeted RPT](<{path_reference(FIGURES/'Figure_7_overview.png',OUT)}>)']
    headers=['Vtail','Method','Pack capacity/Ah','Within-pack SD/Ah','Qualified packs/batch','Pack pass rate/%','Effective yield/%','Rejection rate/%','RPT occurrences/batch']
    for n in [400,1000,1600]:
        ratio=mixture.loc[mixture.n_train_total.eq(n),'pool_usage_ratio'].iloc[0]*100
        rows=[]; quality=[]
        for tail in [46,50,54]:
            for method in METHODS:
                z=means.loc[n,tail,method]
                rows.append([tail,IDS[method],f'{z.cap_mean:.4f}',f'{z.sd_mean:.4f}',f'{z.n_good_packs:.3f}',
                    f'{100*z.pass_rate:.3f}',f'{100*z.yield_rate:.3f}',f'{100*z.rejection_rate:.3f}',f'{z.n_rpt:.3f}'])
                quality.append([tail,IDS[method],f'{100*z.rejection_precision:.3f}',f'{100*z.bad_cell_recall:.3f}',f'{100*z.good_cell_false_rejection_rate:.3f}'])
        report += [f'## Pool usage ratio={ratio:.2f}%; {n} training cells',
            'Table entries are means over 3,000 batches from three seeds. Figure 7 f–i shows the representative setting '
            'with 1,000 training cells and tail 50; j–l shows all nine settings.',table(headers,rows),
            table(['Vtail','Method','Rejection precision/%','Bad-cell recall/%','Good-cell false rejection rate/%'],quality)]
    report += ['## Single-batch example of uncertainty-guided sorting + targeted RPT',f'![Targeted-RPT batch example](<{path_reference(FIGURES/'panel_b.png',OUT)}>)',
        '## Statistical definitions and traceability',
        'Pack capacity is the minimum measured capacity among eight occurrences; within-pack SD uses ddof=1. A qualified pack has capacity ≥14 Ah. '
        'Pack pass rate = qualified packs / assembled packs; effective yield = qualified packs / 37. '
        'Rejection rate = finally rejected occurrences / 300. Retested records retained as qualified and unpacked leftovers are not rejections.',
        'Rejection precision = measured-bad rejected occurrences / all rejected occurrences; '
        'bad-cell recall = measured-bad rejected occurrences / all measured-bad occurrences; '
        'good-cell false rejection rate = measured-good rejected occurrences / all measured-good occurrences. Measured-bad means capacity <14 Ah.',
        'Plot intervals are central 95% random-batch ranges. Per-model summaries also store conditional Monte Carlo mean intervals, '
        'which reflect only finite simulation error. Three-seed mixtures are descriptive summaries without population mean confidence intervals. '
        'These intervals do not cover new populations, retraining or RPT measurement noise.',
        f'There are {int(raw.n_packs.eq(0).sum())} zero-pack batches across the five strategies. '
        'The main-table pass rate is the mean of per-batch ratios; the CSV also stores total qualified packs / total assembled packs.',
        'Sampling with replacement can produce several statistical copies of the same cell. RPT workload counts each selected occurrence '
        'and separately records unique sources; revealed measurements never propagate to unselected copies of the same source. '
        'These results use simulated exact retesting and a minimum-single-cell-capacity pack proxy for fixed models and a fixed test pool. '
        'They are not physical RPT or assembled-pack validation, and do not model test time, monetary cost or measurement noise.',
        'The package includes simulation code, complete inputs and fixed sampling records. '
        'The numerical fingerprint in provenance/ supports result-table comparison by the verifier.',
        '## Files',
        '- `strategy_config.py`: the five methods, official names and IDs.\n'
        '- `run.py --mode full`: rebuild the five-strategy numerical results and figures.\n'
        '- `batch_metrics.csv.gz`: 135,000 batch records; `paired_batch_effects.csv.gz`: 270,000 paired comparisons.\n'
        '- `RPT_workload.csv` and `RPT_comparison_overview.csv`: retesting workload and differences against the other four methods.\n'
        '- Tables combine all training budgets; filter by `n_train_total`. Figure 7 uses 600 dpi PNG panels in the run visualization output directory.\n'
        '- `verification.json`: input integrity, frozen results, metric identities, method names and plotting-source verification.']
    (OUT/'RESULTS_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='TABLES_READY',figure_sets=1,figure_panels=12,figure='Figure 7',strategy_ids=IDS)
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared numerical result tables and report for Figure 7.')


if __name__=='__main__': run()
