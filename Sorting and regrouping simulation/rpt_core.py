"""Targeted RPT planning is truth-free; selection sees only selected measurements."""
import numpy as np
import pack_core
import sorting_core
from pack_core import Features
from quadrant_policy import quadrant_rejections


RPT_METRICS=['n_rpt','rpt_rate','n_unique_rpt_sources','n_rpt_true_good','n_rpt_true_bad',
    'rpt_good_fraction','n_rpt_retained_used','n_rpt_good_leftover','rpt_selection_fraction']


def plan_rpt(features):
    q=quadrant_rejections(features.source_ids,features.pred,features.upper-features.lower,'quadrant_a')
    mask=q['rejected_in_quadrant'].copy()
    return dict(positions=np.flatnonzero(mask),mask=mask,quadrant=q)


def select_rpt(features,selected_measurements,plan=None):
    plan=plan_rpt(features) if plan is None else plan
    measurements=np.asarray(selected_measurements,dtype=float)
    if measurements.shape!=plan['positions'].shape or not np.isfinite(measurements).all():
        raise ValueError('Expected finite measurements for selected RPT positions only')
    capacity=features.pred.copy()
    capacity[plan['positions']]=measurements
    accepted=plan['quadrant']['high_capacity'].copy()
    accepted[plan['positions']]=measurements>=14.
    pos=np.flatnonzero(accepted)
    pos=pos[np.lexsort((pos,features.source_ids[pos],capacity[pos]))]
    return dict(accepted=pos,available_capacity=capacity,plan=plan,
                rpt_good=plan['positions'][measurements>=14.],rpt_bad=plan['positions'][measurements<14.])


def no_rpt_metrics():
    return {key:(np.nan if key in ('rpt_good_fraction','rpt_selection_fraction') else 0) for key in RPT_METRICS}


def evaluate_rpt(features,truth,selection):
    # Full measured labels are available only to this scoring function.
    pos=selection['accepted']; plan=selection['plan']; q=plan['quadrant']
    out=pack_core.evaluate(features,truth,'quadrant_c',selection=(pos,np.zeros(len(truth),dtype=bool)))
    nr=len(plan['positions']); good=len(selection['rpt_good']); bad=len(selection['rpt_bad'])
    used=set(pos[:8*(len(pos)//8)])
    rpt_used=sum(int(i) in used for i in selection['rpt_good'])
    hh=int(q['high_high'].sum())
    out.update(median_uncertainty=q['median_uncertainty'],
        **{'n_quadrant_'+key:int(q[key].sum()) for key in ('high_high','high_low','low_high','low_low')},
        n_rejected_quadrant=bad,n_rejected_quadrant_true_bad=bad,n_rejected_quadrant_true_good=0,
        quadrant_rejection_fraction=pack_core.divide(bad,hh),
        n_rpt=nr,rpt_rate=nr/len(truth),n_unique_rpt_sources=len(np.unique(features.source_ids[plan['positions']])),
        n_rpt_true_good=good,n_rpt_true_bad=bad,rpt_good_fraction=pack_core.divide(good,nr),
        n_rpt_retained_used=rpt_used,n_rpt_good_leftover=good-rpt_used,
        rpt_selection_fraction=pack_core.divide(nr,hh))
    return out
