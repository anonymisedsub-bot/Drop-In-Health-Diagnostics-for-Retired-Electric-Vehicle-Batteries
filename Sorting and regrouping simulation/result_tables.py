"""Paired effects and traceable first-batch cell assignments."""
import itertools
import numpy as np
import pandas as pd
from statistics_io import META
from strategy_config import METHODS,NAMES,IDS
COMPARISONS=[(new,base) for base,new in itertools.combinations(METHODS[:-1],2)]+[(METHODS[-1],base) for base in METHODS[:-1]]

def paired_effects(raw):
    pieces=[]
    keys=META+['batch_id']
    metrics=['cap_mean','sd_mean','pass_rate','yield_rate','rejection_rate','n_good_packs',
             'rejection_precision','bad_cell_recall','good_cell_false_rejection_rate']
    for method,baseline in COMPARISONS:
        a=raw.loc[raw.method.eq(method),keys+metrics]
        b=raw.loc[raw.method.eq(baseline),keys+metrics]
        z=a.merge(b,on=keys,validate='one_to_one',suffixes=('_method','_baseline'))
        out=z[keys].copy()
        out['comparison']=method+'_vs_'+baseline
        for metric in metrics:
            rate='rate' in metric or metric in ('rejection_precision','bad_cell_recall')
            suffix='_difference_pp' if rate else '_difference'
            out[metric+suffix]=(z[metric+'_method']-z[metric+'_baseline'])*(100 if rate else 1)
        out['capacity_improvement_percent']=100*(z.cap_mean_method-z.cap_mean_baseline)/z.cap_mean_baseline.replace(0,np.nan)
        out['sd_reduction_percent']=100*(z.sd_mean_baseline-z.sd_mean_method)/z.sd_mean_baseline.replace(0,np.nan)
        pieces.append(out)
    return pd.concat(pieces,ignore_index=True)

def example_rows(meta,d,idx,selections):
    rows=[]
    for method,s in selections.items():
        ranks={int(pos):rank for rank,pos in enumerate(s['accepted'])}
        used=8*(len(ranks)//8)
        q=s['quadrant']
        for i,j in enumerate(idx):
            rank=ranks.get(i,-1)
            capacity_ok=bool(q['high_capacity'][i]) if q is not None else d.y_pred.iloc[j]>=14
            quadrant=next(k for k in ('high_high','high_low','low_high','low_low') if q[k][i]) if q is not None else ''
            rows.append(dict(**meta,batch_id=0,method=method,strategy_name=NAMES[method],sample_instance_id=i,
                source_row=int(j),source_cell_id=int(d.cell_id.iloc[j]),y_pred=float(d.y_pred.iloc[j]),
                lower=float(d.lower.iloc[j]),upper=float(d.upper.iloc[j]),y_true=float(d.y_true.iloc[j]),
                median_uncertainty=q['median_uncertainty'] if q is not None else np.nan,quadrant=quadrant,
                rejected_in_quadrant=bool(q['rejected_in_quadrant'][i]) if q is not None else False,
                accepted_rank=rank,pack_number=rank//8 if 0<=rank<used else -1,
                disposition=('capacity_rejected' if not capacity_ok else 'strategy_rejected') if rank<0
                    else ('assembled' if rank<used else 'leftover')))
    return rows

def c_example(meta,d,idx,s):
    rows=[]; rank={int(p):i for i,p in enumerate(s['accepted'])}; used=8*(len(rank)//8)
    plan=s['plan']; q=plan['quadrant']
    for i,j in enumerate(idx):
        rr=rank.get(i,-1); tested=bool(plan['mask'][i]); accepted=rr>=0
        rows.append(dict(**meta,batch_id=0,method='quadrant_c',strategy_name=NAMES['quadrant_c'],
            sample_instance_id=i,source_row=int(j),source_cell_id=int(d.cell_id.iloc[j]),
            y_pred=float(d.y_pred.iloc[j]),lower=float(d.lower.iloc[j]),upper=float(d.upper.iloc[j]),
            y_true=float(d.y_true.iloc[j]),median_uncertainty=q['median_uncertainty'],
            quadrant=next(k for k in ('high_high','high_low','low_high','low_low') if q[k][i]),
            rpt_selected=tested,rpt_measurement=float(d.y_true.iloc[j]) if tested else np.nan,
            available_capacity=float(s['available_capacity'][i]),capacity_source='rpt' if tested else 'prediction',
            accepted_rank=rr,pack_number=rr//8 if 0<=rr<used else -1,
            disposition=('assembled' if rr<used else 'leftover') if accepted else
                ('rpt_rejected' if tested else 'capacity_rejected')))
    return rows
