"""Prediction-only quadrant membership and highest-width rejection candidates.

This component does not decide the disposition of low-capacity records.
"""
import numpy as np


def quadrant_rejections(source_ids,pred,width,method):
    source_ids=np.asarray(source_ids)
    pred=np.asarray(pred,dtype=float)
    width=np.asarray(width,dtype=float)
    n=len(pred)
    if not n or any(a.shape!=(n,) or not np.isfinite(a).all() for a in (source_ids,pred,width)):
        raise ValueError('Expected nonempty aligned finite one-dimensional arrays')
    if (width<0).any():
        raise ValueError('Uncertainty width must be nonnegative')
    if method=='quadrant_a':
        high_capacity=pred>=14.
    else:
        raise ValueError(method)
    median=float(np.median(width))
    high_uncertainty=width>median
    hh=high_capacity&high_uncertainty
    positions=np.flatnonzero(hh)
    ranked=positions[np.lexsort((positions,source_ids[positions],width[positions]))]
    n_rejected=(len(ranked)+4)//5
    rejected=np.zeros(n,dtype=bool)
    if n_rejected:
        rejected[ranked[-n_rejected:]]=True
    return dict(median_uncertainty=median,high_capacity=high_capacity,high_uncertainty=high_uncertainty,
                high_high=hh,high_low=high_capacity&~high_uncertainty,
                low_high=~high_capacity&high_uncertainty,low_low=~high_capacity&~high_uncertainty,
                rejected_in_quadrant=rejected)
