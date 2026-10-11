"""Portable sorting/regrouping entry point with standardized output folders."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from portable_io import output_paths

HERE=Path(__file__).resolve().parent

def default_output(mode):
    return output_paths(mode, reference_verify=True)['results']

def demo(output, figures):
    # Actual neural inference runs in its own process to avoid legacy config collisions.
    subprocess.run([sys.executable,str(HERE/'model/infer.py'),str(output)],check=True)
    from pack_core import Features,draw_indices
    from sorting_core import select_policy,evaluate_policy
    from rpt_core import plan_rpt,select_rpt,evaluate_rpt,no_rpt_metrics
    from strategy_config import METHODS,NAMES
    d=pd.read_csv(output/'model_predictions_t46_i2_s0.csv')
    d['lower']=np.minimum(d.raw_lower,d.raw_upper); d['upper']=np.maximum(d.raw_lower,d.raw_upper)
    rows=[]
    for batch,idx in enumerate(draw_indices(609,300,20,0,20260924)):
        f=Features(d.cell_id.to_numpy()[idx],d.y_pred.to_numpy()[idx],d.lower.to_numpy()[idx],d.upper.to_numpy()[idx])
        y=d.y_true.to_numpy()[idx]; plan=plan_rpt(f); rpt=select_rpt(f,y[plan['positions']],plan)
        for method in METHODS:
            metrics=evaluate_rpt(f,y,rpt) if method=='quadrant_c' else dict(**evaluate_policy(f,y,method),**no_rpt_metrics())
            rows.append(dict(batch_id=batch,method=method,strategy_name=NAMES[method],**metrics))
    pd.DataFrame(rows).to_csv(output/'demo_batch_metrics.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['method','strategy_name'],sort=False)[['yield_rate','pass_rate','rejection_rate','rpt_rate']].mean().reset_index()
    summary.to_csv(output/'demo_figure_source.csv',index=False)
    subprocess.run([sys.executable,str(HERE/'visualization/plot_demo.py'),
                    str(output/'demo_figure_source.csv'),str(figures)],check=True)
    (output/'demo_manifest.json').write_text(json.dumps(dict(evaluation_type='simulation_only',physical_pack_validation=False,model='historical t46_i2_s0',n_batches=20,source_pool=609,pack_capacity='minimum measured cell capacity proxy',rpt='exact reveal of selected measured labels, without added measurement noise',predictions='new checkpoint inference; frozen full outputs use saved workbook predictions'),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['demo','full','verify','model','figures'],default='demo')
    parser.add_argument('--output',type=Path,help='Custom root containing results/ and visualization/; defaults to this module.')
    args=parser.parse_args()
    try: paths=output_paths(args.mode,args.output,reference_verify=True)
    except ValueError as exc: parser.error(str(exc))
    output=paths['results']; figures=paths['figures']; sources=paths['figure_data']
    if args.mode not in ('verify','figures'): output.mkdir(parents=True,exist_ok=True)
    if args.mode=='demo': demo(output,figures)
    elif args.mode=='model': subprocess.run([sys.executable,str(HERE/'model/infer.py'),str(output)],check=True)
    else:
        if args.mode=='full':
            import run_simulation, refresh_results
            run_simulation.OUT=output; run_simulation.run()
            refresh_results.OUT=output; refresh_results.FIGURES=figures; refresh_results.run()
        if args.mode in ('full','figures'):
            subprocess.run([sys.executable,str(HERE/'visualization/render_figure7.py'),
                '--data',str(output),'--output',str(figures),'--source-data',str(sources)],check=True)
        if args.mode in ('full','verify'):
            import verify_final
            verify_final.OUT=output; verify_final.FIGURES=figures
            verify_final.run(write_report=args.mode=='full')

if __name__=='__main__': main()
