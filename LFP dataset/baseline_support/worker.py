"""One method per fresh interpreter, preserving the copied scientific modules."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def json_config(cfg):
    result = {}
    for name, value in vars(cfg).items():
        if name.startswith('_'):
            continue
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        result[name] = value
    return result


def load_matrix(path):
    import numpy as np
    from scipy.io import loadmat
    contents = loadmat(path)
    for key in ('processed_Qd_mat', 'processed_Vd_mat'):
        if key not in contents:
            raise ValueError(f'LFP MAT schema requires {key}')
    q, voltage = (np.asarray(contents[key], dtype=float) for key in ('processed_Qd_mat', 'processed_Vd_mat'))
    if q.ndim != 2 or q.shape[0] != 201 or voltage.shape != q.shape:
        raise ValueError('LFP MAT requires matching processed_Qd_mat/processed_Vd_mat matrices of shape (201, cells)')
    if not np.isfinite(q).all() or not np.isfinite(voltage).all() or np.any(q[-1] <= 0):
        raise ValueError('LFP MAT contains nonfinite data or nonpositive final capacities')
    if not np.allclose(voltage, np.linspace(2.5, 3.5, 201)[:, None], rtol=0, atol=1e-6):
        raise ValueError('LFP voltage grid must ascend from 2.50 to 3.50 V with 0.005 V spacing')
    return q


def budgets_for(data, cfg, n_cells, seed):
    import numpy as np
    split = data.active_split_indices(n_cells, seed=seed)
    joined = np.concatenate(list(split.values()))
    if len(joined) != n_cells or len(np.unique(joined)) != n_cells or any(not len(split[k]) for k in ('train', 'val', 'test')):
        raise ValueError('LFP dataset is too small for the original nonempty train/validation/test protocol')
    return split, list(range(len(split['train']), len(split['train']) + len(split['pool']) + 1, cfg.num_tr_per_iter))


def advance(split, rng, step):
    import numpy as np
    chosen = rng.choice(len(split['pool']), size=min(step, len(split['pool'])), replace=False)
    result = {key: value.copy() for key, value in split.items()}
    result['train'] = np.sort(np.concatenate((split['train'], split['pool'][chosen])))
    keep = np.ones(len(split['pool']), dtype=bool)
    keep[chosen] = False
    result['pool'] = split['pool'][keep]
    return result


def metrics(actual, predicted):
    import numpy as np
    error = predicted - actual
    denominator = actual + (actual == 0) * 1e-12
    return {'MAPE': float(np.mean(np.abs(error / denominator)) * 100),
            'RMSPE': float(np.sqrt(np.mean((error / denominator)**2)) * 100),
            'R2': float(1 - np.sum(error**2) / (np.sum((actual - actual.mean())**2) + 1e-12)),
            'STD': float(np.std(error)), 'MAE_Ah': float(np.mean(np.abs(error))),
            'RMSE_Ah': float(np.sqrt(np.mean(error**2)))}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run_round(request, cfg, data, utils, q, split, original_split, number, output, v_init):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    group, canonical = request['group'], request['model']
    model_name = canonical.lower()
    length = cfg.segment_length if group == 4 else cfg.TAIL_LEN
    arrays = {key: data.build_XY_from_global(q, ids, length) for key, ids in split.items()}
    xtr, ytr = arrays['train']
    xval, yval = arrays['val']
    xtest, ytest = arrays['test']
    xpool, _ = arrays['pool']
    norm = data.fit_normalizer(xtr, ytr, cfg.normalization_mode)
    extra_training_args = {}
    if group == 4:
        voltage_cfg = data.get_voltage_config()
        rng = np.random.RandomState(request['seed'] + number)
        xpool = data.apply_masking(xpool, v_init[split['pool']], 'pool_test', 0, None, rng, voltage_cfg)
        xtest = data.apply_masking(xtest, v_init[split['test']], 'pool_test', 0, None, rng, voltage_cfg)
        validation_full_n, validation_y_n = data.apply_normalizer(xval, yval, norm)
        xval = data.apply_masking(xval, None, 'val', 0, None, rng, voltage_cfg)
        if model_name in ('mlp', 'cnn'):
            extra_training_args = dict(cl_config=cfg.missing_sim_curriculum,
                Xval_n_full=validation_full_n, Yval_n_full=validation_y_n, voltage_cfg=voltage_cfg)
        else:
            xtr = data.apply_masking(xtr, None, 'train_ml', 0, cfg.missing_sim_random, rng, voltage_cfg)
    xn, yn = data.apply_normalizer(xtr, ytr, norm)
    xvn, yvn = data.apply_normalizer(xval, yval, norm)
    xtn, ytn = data.apply_normalizer(xtest, ytest, norm)
    xpn, _ = data.apply_normalizer(xpool, np.zeros((len(xpool), 1)), norm)
    training_metadata = {}
    output.mkdir(parents=True, exist_ok=False)
    if group == 3:
        from train.train_da import train_da_baseline, predict_q50
        trained = train_da_baseline(canonical, xn, yn, xpn, xvn, yvn, cfg, request['seed'])
        handle = trained.model
        prediction = utils.denorm_y(predict_q50(handle, xtn, cfg).reshape(-1, 1), norm).reshape(-1)
        training_metadata = trained.metadata
        if trained.auxiliary_states:
            torch.save(trained.auxiliary_states, output / 'best_aux.pt')
        torch.save(handle.state_dict(), output / 'best.pt')
        family = 'DL'
    elif group == 2 or model_name in ('mlp', 'cnn'):
        from train.train_torch import train_val_one_model, eval_model_full
        handle = train_val_one_model(model_name, xn, yn, xvn, yvn, cfg, **extra_training_args)
        evaluation = eval_model_full(handle, xn, yn, xvn, yvn, xtn, ytn, cfg, norm=norm, clip01=False)
        values = np.asarray(evaluation['test']['y_pred_den'])
        prediction = values[:, 1] if values.ndim == 2 and values.shape[1] == 3 else values.reshape(-1)
        torch.save(handle.state_dict(), output / 'best.pt')
        family = 'DL'
    else:
        from train.train_sklearn import build_ml_models, eval_ml_model
        handle = build_ml_models(cfg)[model_name]()
        evaluation = eval_ml_model(handle, xn, yn, xvn, yvn, xtn, ytn, norm=norm, quantiles=cfg.QUANTILES)
        values = np.asarray(evaluation['test']['y_pred_den'])
        prediction = values[:, 1] if values.ndim == 2 and values.shape[1] == 3 else values.reshape(-1)
        joblib.dump(handle, output / 'best.pt')
        family = 'ML'
    actual = ytest.reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError('Training produced nonfinite test predictions')
    scores = metrics(actual, prediction)
    pd.DataFrame({'cell_index_zero_based': split['test'], 'true_capacity_Ah': actual,
                  'pred_capacity_Ah': prediction, 'error_Ah': prediction - actual}).to_csv(output / 'predictions.csv', index=False)
    np.savez(output / 'normalizers.npz', **{k: norm[k] for k in ('xmin', 'xmax', 'ymin', 'ymax')})
    write_json(output / 'splits.json', {key: ids.tolist() for key, ids in split.items()})
    write_json(output / 'budget_splits.json', {key: ids.tolist() for key, ids in original_split.items()})
    write_json(output / 'metrics.json', scores)
    metadata = dict(dataset='LFP', dataset_sha256=sha256(cfg.DATA_MAT_PATH), config_sha256=sha256(cfg.__file__), group=group, model=canonical,
        family=family, method=canonical if group == 3 else 'SL', seed=request['seed'], round=number,
        original_label_budget=len(original_split['train']),
        effective_split_sizes={key: len(ids) for key, ids in split.items()}, effective_config=json_config(cfg),
        prediction_behavior='normalized predictions clipped [0,1] (v5 group3)' if group == 3 else 'v5 predictions without output clipping',
        normalizers='normalizers.npz; fitted on labelled training cells only; X/Y transforms clipped [0,1] as in v5',
        checkpoint='best.pt', checkpoint_sha256=sha256(output / 'best.pt'),
        split_identity='RandomState(seed), sorted subsets; nested random acquisitions with RandomState(seed+123)',
        index_convention='zero-based column in processed MAT', target='processed_Qd_mat[-1, cell]',
        training=training_metadata, metrics=scores, worker_pid=os.getpid(),
        notice='New LFP baseline results from the saved configuration; not bundled paper results.')
    write_json(output / 'run_metadata.json', metadata)
    return {'round': number, 'output': str(output), 'label_budget': len(original_split['train']), 'metrics': scores}


def execute(request):
    import numpy as np
    import torch
    config_path = Path(request['config_path']).resolve()
    if sha256(config_path) != request['config_sha256']:
        raise ValueError('config.py changed during this run; start a new campaign')
    sys.path.insert(0, str(config_path.parent))
    import config as cfg
    import data
    import utils
    torch.set_num_threads(cfg.THREADS)
    if request['group'] != 1 and cfg.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("config.py device='cuda' requested but CUDA is not available")
    q = load_matrix(cfg.DATA_MAT_PATH)
    dataset_hash = sha256(cfg.DATA_MAT_PATH)
    if request.get('dataset_sha256', dataset_hash) != dataset_hash:
        raise ValueError('Dataset changed during this run; start a new campaign')
    split, budgets = budgets_for(data, cfg, q.shape[1], request['seed'])
    rounds = list(cfg.ROUND_NUMBERS) if cfg.ROUND_NUMBERS is not None else list(range(1, len(budgets) + 1))
    if any(number > len(budgets) for number in rounds):
        raise ValueError(f'ROUND_NUMBERS must lie between 1 and {len(budgets)} for this dataset')
    length = cfg.segment_length if request['group'] == 4 else cfg.TAIL_LEN
    if not 1 <= length <= q.shape[0]:
        raise ValueError('Configured input window must contain between 1 and 201 points')
    modules = {'XGB': 'xgboost', 'CGB': 'catboost', 'LGBM': 'lightgbm'}
    dependencies = {modules[name]: importlib.util.find_spec(modules[name]) is not None for name in cfg.SELECTED_MODELS if name in modules}
    if request['command'] == 'check':
        return dict(group=request['group'], models=list(cfg.SELECTED_MODELS), seed=request['seed'],
                    data_shape=list(q.shape), dataset_sha256=dataset_hash,
                    config_sha256=request['config_sha256'], input_length=length,
                    tail_voltage=[2.5, 2.5 + 0.005 * (cfg.TAIL_LEN - 1)],
                    target='processed_Qd_mat[-1, cell]',
                    split_sizes={key: len(value) for key, value in split.items()}, label_budgets=budgets,
                    round_numbers=rounds, effective_config=json_config(cfg),
                    dependencies=dependencies, worker_pid=os.getpid())
    utils.set_seed(request['seed'])
    rng = np.random.RandomState(request['seed'] + 123)
    v_init = data.simulate_initial_voltage(q.shape[1], seed=request['seed']) if request['group'] == 4 else None
    results = []
    case_root = Path(request['output'])
    for number in range(1, max(rounds) + 1):
        if number in rounds:
            results.append(run_round(request, cfg, data, utils, q, split, split, number, case_root / f'round_{number}', v_init))
        if number < max(rounds):
            split = advance(split, rng, cfg.num_tr_per_iter)
    if v_init is not None:
        np.save(case_root / 'V_init.npy', v_init)
    return {'group': request['group'], 'model': request['model'], 'seed': request['seed'],
            'worker_pid': os.getpid(), 'rounds': results}


if __name__ == '__main__':
    request = json.load(sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            result = execute(request)
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(2)
