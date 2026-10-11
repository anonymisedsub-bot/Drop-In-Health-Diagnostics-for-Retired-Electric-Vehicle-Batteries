"""Prediction-only selection of bootstrap occurrences; measured labels only score packs."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import t

METHODS = ('point', 'conservative', 'lower_bound')


@dataclass(frozen=True)
class Features:
    source_ids: np.ndarray
    pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        n = len(self.source_ids)
        for name in ('source_ids', 'pred', 'lower', 'upper'):
            a = np.array(getattr(self, name), copy=True)
            if a.shape != (n,) or not np.isfinite(a).all():
                raise ValueError('Features must be finite aligned one-dimensional arrays')
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if np.any(self.lower > self.pred) or np.any(self.pred > self.upper):
            raise ValueError('Expected lower <= prediction <= upper')


def draw_indices(n_source, batch_size, repetitions, seed, simulation_seed):
    rng = np.random.default_rng(np.random.SeedSequence([simulation_seed, seed, 3000924]))
    return rng.integers(0, n_source, size=(repetitions, batch_size), dtype=np.int32)


def select(features, method):
    """Return sorted occurrence POSITIONS, never deduplicated original cell IDs."""
    if method not in METHODS:
        raise ValueError(method)
    n = len(features.pred)
    occurrence = np.arange(n)
    high = np.zeros(n, dtype=bool)
    accepted = features.pred >= 14.
    if method == 'conservative':
        accepted &= features.pred > 14.3
    elif method == 'lower_bound':
        accepted &= features.lower >= 14.
    pos = np.flatnonzero(accepted)
    pos = pos[np.lexsort((pos, features.source_ids[pos], features.pred[pos]))]
    return pos, high


def divide(a, b):
    return float(a / b) if b else np.nan


def evaluate(features, truth, method, selection=None):
    pos, high = select(features, method) if selection is None else selection
    truth = np.asarray(truth, dtype=float)
    n = len(features.pred)
    if truth.shape != (n,) or not np.isfinite(truth).all():
        raise ValueError('Measured capacities must be finite and aligned')
    accepted = np.zeros(n, dtype=bool)
    accepted[pos] = True
    rejected = ~accepted
    eligible = features.pred >= 14.
    additional = eligible & rejected
    bad = truth < 14.
    k = len(pos) // 8
    used = 8 * k
    pack_truth = truth[pos[:used]].reshape(k, 8)
    caps = pack_truth.min(axis=1) if k else np.array([])
    sds = pack_truth.std(axis=1, ddof=1) if k else np.array([])
    g = int(np.sum(caps >= 14.))
    tp = int(np.sum(rejected & bad))
    fp = int(np.sum(rejected & ~bad))
    fn = int(np.sum(accepted & bad))
    tn = int(np.sum(accepted & ~bad))
    nr = int(rejected.sum())
    # These zero-valued columns retain the frozen result schema.
    hu = np.zeros(n, dtype=bool)
    out = dict(n_input=n, n_unique_sources=len(np.unique(features.source_ids)),
        n_accepted=len(pos), n_rejected=nr, n_rejected_total=nr,
        n_rejected_capacity=int((~eligible).sum()), n_rejected_strategy=int(additional.sum()),
        n_rejected_uncertainty_global=int(hu.sum()), n_rejected_overlap=int((hu & ~eligible).sum()),
        n_leftover=len(pos)-used, n_used=used, n_packs=k, n_good_packs=g,
        n_failed_packs=k-g, n_possible_packs=n//8, n_eligible=int(eligible.sum()),
        rejection_rate=divide(nr,n), assembly_rate=divide(k,n//8), utilization_rate=divide(used,n),
        pass_rate=divide(g,k), yield_rate=divide(g,n//8),
        cap_mean=float(caps.mean()) if k else np.nan, sd_mean=float(sds.mean()) if k else np.nan,
        cap_sum=float(caps.sum()), sd_sum=float(sds.sum()),
        global_uncertainty_rejection_rate=divide(hu.sum(),n),
        additional_rejection_rate_full_pool=divide(additional.sum(),n),
        strategy_rejection_rate=divide(additional.sum(),eligible.sum()),
        n_true_bad=int(bad.sum()), n_true_good=int((~bad).sum()),
        n_rejected_true_bad=tp, n_rejected_true_good=fp,
        n_retained_true_bad=fn, n_retained_true_good=tn,
        rejection_precision=divide(tp,nr), bad_cell_recall=divide(tp,bad.sum()),
        good_cell_false_rejection_rate=divide(fp,(~bad).sum()),
        n_additional_rejected_true_bad=int((additional & bad).sum()),
        n_additional_rejected_true_good=int((additional & ~bad).sum()))
    return out


def summarize_values(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if not n:
        return dict(n_valid=0, mean=np.nan, median=np.nan, sd=np.nan,
                    batch_q025=np.nan, batch_q975=np.nan, mc_mean_ci_low=np.nan,
                    mc_mean_ci_high=np.nan, degenerate=False)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    half = float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n > 1 else np.nan
    return dict(n_valid=n, mean=mean, median=float(np.median(a)), sd=sd,
                batch_q025=float(np.quantile(a,.025)), batch_q975=float(np.quantile(a,.975)),
                mc_mean_ci_low=mean-half, mc_mean_ci_high=mean+half,
                degenerate=bool(np.ptp(a)==0))
