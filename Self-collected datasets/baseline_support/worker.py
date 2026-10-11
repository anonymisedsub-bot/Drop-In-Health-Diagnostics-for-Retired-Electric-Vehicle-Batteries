"""Internal isolated execution of one model/seed from an unchanged config.py."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def execute(group, cfg, model, seed, output):
    from baseline_support.runner import validate_config, validate_output, write_json
    from baseline_support.rebatt_separate import load_separate_batch
    from baseline_support.rebatt_protocol import (
        make_batch_split, make_round_roles, round_numbers, write_split_manifest,
        write_budget_manifest, write_config_snapshot,
    )
    import numpy as np
    import torch

    input_length = validate_config(group, cfg)
    if model not in cfg.SELECTED_MODELS or seed not in cfg.random_seeds:
        raise ValueError('Worker case must be selected in config.py')
    output = validate_output(output)
    # Atomic ownership applies to check and train alike; never reuse a case.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.THREADS)
    if group != 1 and cfg.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; edit device in config.py')
    data = load_separate_batch(cfg.DATA_MAT_PATH, cfg.SOURCE_BATCH)
    split = make_batch_split(data, batch=cfg.MODEL_BATCH, seed=seed,
                             pool_ratio=cfg.POOL_TARGET_RATIO, pool_divisor=cfg.POOL_DIVISOR,
                             initial_labelled_count=cfg.INITIAL_LABELLED_COUNT,
                             acquisition_size=cfg.ACQUISITION_SIZE, vseg_low=cfg.VSEG_LOW)
    available_rounds = list(round_numbers(split))
    rounds = list(cfg.ROUND_NUMBERS) if cfg.ROUND_NUMBERS is not None else available_rounds
    if any(number not in available_rounds for number in rounds):
        raise ValueError(f'ROUND_NUMBERS must select available rounds for this MAT: {available_rounds}')
    adapter = importlib.import_module('data')
    roles = make_round_roles(split, rounds[0])
    builder = adapter.build_padding_xy if group == 4 else adapter.build_tail_xy
    raw = {}
    for role, indices in [('labelled', roles.labelled_indices), ('unlabelled', roles.unlabelled_indices),
                          ('validation', roles.validation_indices), ('test', roles.test_indices)]:
        x, y = builder(data, indices, role, cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW)
        if x.shape != (len(indices), 1, input_length):
            raise ValueError(f'Unexpected {role} input shape: {x.shape}')
        if role == 'unlabelled' and y is not None:
            raise ValueError('Unlabelled capacities must remain hidden')
        raw[role] = (x, y)
    normalizer = adapter.fit_normalizer(*raw['labelled'], cfg.normalization_mode)
    for role, pair in raw.items():
        x, y = adapter.apply_normalizer(*pair, normalizer)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite normalized inputs for {role}')
    dependency = {'XGB': 'xgboost', 'CGB': 'catboost', 'LGBM': 'lightgbm'}.get(model)
    dependency_available = dependency is None or importlib.util.find_spec(dependency) is not None
    training_settings = {name: getattr(cfg, name) for name in
                         ('num_epochs', 'batch_size', 'lr', 'es_enable', 'es_patience',
                          'THREADS', 'device', 'ML_CONFIG') if hasattr(cfg, name)}
    report = {'dataset': cfg.DATASET_NAME, 'group': group, 'model': model, 'seed': seed,
              'source_batch': cfg.SOURCE_BATCH, 'model_batch': cfg.MODEL_BATCH,
              'source_row_count': len(data.qmax), 'tail_voltage_range': [cfg.V_MIN, cfg.VSEG_LOW],
              'upstream_voltage_range': [cfg.VSEG_LOW, cfg.VSEG_UPP], 'tail_length': cfg.TAIL_LEN,
              'input_length': input_length, 'voltage_removal_window': cfg.VOLTAGE_REMOVAL_WINDOW,
              'labelled_count': len(roles.labelled_indices), 'unlabelled_count': len(roles.unlabelled_indices),
              'validation_count': len(split.validation_indices), 'test_count': len(split.test_indices),
              'eligible_count': len(data.qmax) - len(split.excluded_indices),
              'unlabelled_targets_hidden': raw['unlabelled'][1] is None,
              'prediction_clipping': False, 'residual_std_ddof': 0,
              'mat_sha256': hashlib.sha256(data.mat_path.read_bytes()).hexdigest(),
              'data_path': str(data.mat_path), 'rounds': rounds, 'available_rounds': available_rounds,
              'normalization_fit_role': 'labelled only', 'process_id': os.getpid(),
              'dependency': dependency, 'dependency_available': dependency_available,
              'training_settings': training_settings,
              'group4_lgbm': cfg.ML_CONFIG['lgbm'] if group == 4 else None}
    write_json(output / 'check.json', report)
    write_config_snapshot(output / 'config_snapshot.json', cfg)
    if cfg.RUN_MODE == 'check':
        print(json.dumps(report), flush=True)
        return
    if not dependency_available:
        raise ImportError(f'{dependency} is required for {model}; install requirements-baselines.txt')
    scientific = importlib.import_module('scientific')
    method, model_name = (model, 'mlp') if group == 3 else ('SL', model.lower())
    write_split_manifest(output / 'protocol_split_manifest.json', data, split)
    for round_number in rounds:
        roles = make_round_roles(split, round_number)
        destination = output / f'iter_{round_number}'
        destination.mkdir()
        write_budget_manifest(destination / 'protocol_budget_manifest.json', data, roles)
        execution = {**report, 'round': round_number, 'status': 'running',
                     'checkpoint': 'best.pt', 'normalization': 'normalization.json',
                     'split_identity': '../protocol_split_manifest.json',
                     'budget_identity': 'protocol_budget_manifest.json',
                     'effective_counts': {'labelled': len(roles.labelled_indices),
                                          'unlabelled': len(roles.unlabelled_indices),
                                          'validation': len(roles.validation_indices),
                                          'test': len(roles.test_indices)}}
        write_json(destination / 'execution.json', execution)
        try:
            scientific.run_one_case(data, split, roles, seed, method, model_name, destination)
            execution['status'] = 'completed'
        except BaseException:
            execution['status'] = 'failed'
            raise
        finally:
            write_json(destination / 'execution.json', execution)
    print(json.dumps({'dataset': cfg.DATASET_NAME, 'model': model, 'status': 'completed',
                      'output': str(output)}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit('Internal worker; run the group main.py using config.py.')
    group_text, config_text, selected_model, seed_text, output_text, expected_hash = sys.argv[1:]
    config_path = Path(config_text).resolve()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('config.py changed during the campaign; start a new run')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(config_path.parent))
    cfg = importlib.import_module('config')
    if Path(cfg.__file__).resolve() != config_path:
        raise ValueError('The worker imported a different config.py')
    execute(int(group_text), cfg, selected_model, int(seed_text), Path(output_text))
