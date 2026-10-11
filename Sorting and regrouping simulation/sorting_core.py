"""Four prediction-only sorting policies; truth is used only for scoring."""
import numpy as np
import pack_core
from quadrant_policy import quadrant_rejections
from strategy_config import METHODS as ALL_METHODS
METHODS=ALL_METHODS[:-1]
Features=pack_core.Features


def select_policy(features,method):
    if method not in METHODS:
        raise ValueError(method)
    n=len(features.pred)
    if method in pack_core.METHODS:
        pos,global_high=pack_core.select(features,method)
        return dict(accepted=pos,global_high=global_high,quadrant=None)
    q=quadrant_rejections(features.source_ids,features.pred,features.upper-features.lower,method)
    accepted=q['high_capacity']&~q['rejected_in_quadrant']
    pos=np.flatnonzero(accepted)
    pos=pos[np.lexsort((pos,features.source_ids[pos],features.pred[pos]))]
    return dict(accepted=pos,global_high=np.zeros(n,dtype=bool),quadrant=q)


def evaluate_policy(features,truth,method,selection=None):
    s=select_policy(features,method) if selection is None else selection
    out=pack_core.evaluate(features,truth,method,selection=(s['accepted'],s['global_high']))
    q=s['quadrant']
    out.update(median_uncertainty=np.nan,n_quadrant_high_high=0,n_quadrant_high_low=0,
               n_quadrant_low_high=0,n_quadrant_low_low=0,n_rejected_quadrant=0,
               n_rejected_quadrant_true_bad=0,n_rejected_quadrant_true_good=0,
               quadrant_rejection_fraction=np.nan)
    if q is not None:
        truth=np.asarray(truth)
        eligible=int(q['high_capacity'].sum())
        extra=int(q['rejected_in_quadrant'].sum())
        out['n_eligible']=eligible
        out['n_rejected_capacity']=len(truth)-eligible
        out['n_rejected_strategy']=extra
        out['strategy_rejection_rate']=pack_core.divide(extra,eligible)
        out['additional_rejection_rate_full_pool']=extra/len(truth)
        out['median_uncertainty']=q['median_uncertainty']
        for k in ('high_high','high_low','low_high','low_low'):
            out['n_quadrant_'+k]=int(q[k].sum())
        out['n_rejected_quadrant']=extra
        out['n_rejected_quadrant_true_bad']=int((q['rejected_in_quadrant']&(truth<14)).sum())
        out['n_rejected_quadrant_true_good']=int((q['rejected_in_quadrant']&(truth>=14)).sum())
        out['quadrant_rejection_fraction']=pack_core.divide(extra,int(q['high_high'].sum()))
    return out
