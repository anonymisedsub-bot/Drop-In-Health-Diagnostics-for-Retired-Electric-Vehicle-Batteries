"""Internal process orchestration for config-driven NMC entry points."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {1: ('XGB', 'RF', 'CGB', 'LGBM'),
          2: ('MLP', 'CNN', 'LSTM', 'Transformer'),
          3: ('DA-AL', 'DA-MMD', 'DA-CORAL', 'SSL-PL'),
          4: ('MLP', 'CNN', 'CGB', 'LGBM')}


def _positive_integer(value):
    return type(value) is int and value > 0


def validate_config(group, cfg):
    """Reject inconsistent settings; never rewrite the imported configuration."""
    if group not in GROUPS:
        raise ValueError(f'Unknown baseline group: {group}')
    if cfg.RUN_MODE not in ('train', 'check'):
        raise ValueError('RUN_MODE must be train or check')
    selected = cfg.SELECTED_MODELS
    if (not isinstance(selected, tuple) or not selected
            or any(name not in GROUPS[group] for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError(f'SELECTED_MODELS must be a nonempty unique tuple from {GROUPS[group]}')
    if (not isinstance(cfg.random_seeds, tuple) or not cfg.random_seeds
            or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in cfg.random_seeds)
            or len(set(cfg.random_seeds)) != len(cfg.random_seeds)):
        raise ValueError('random_seeds must be a nonempty unique tuple of integers in 0..2**32-1')
    rounds = cfg.ROUND_NUMBERS
    if rounds is not None and (not isinstance(rounds, tuple) or not rounds
            or any(not _positive_integer(number) for number in rounds)
            or len(set(rounds)) != len(rounds)):
        raise ValueError('ROUND_NUMBERS must be None or a nonempty tuple of unique positive indices')
    if not _positive_integer(cfg.THREADS):
        raise ValueError('THREADS must be a positive integer')
    if group != 1 and cfg.device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    identities = {'NMC_A': (4, 4, 3.0), 'NMC_B': (6, 3, 2.75)}
    if cfg.DATASET_NAME not in identities:
        raise ValueError('DATASET_NAME must be NMC_A or NMC_B')
    source, batch, minimum = identities[cfg.DATASET_NAME]
    for name, expected in [('SOURCE_BATCH', source), ('MODEL_BATCH', batch),
                           ('V_MIN', minimum), ('V_MAX', 4.2), ('DV', .005)]:
        if not math.isclose(getattr(cfg, name), expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'{name} disagrees with the selected {cfg.DATASET_NAME} MAT schema: expected {expected}')
    if not cfg.V_MIN <= cfg.VSEG_LOW < cfg.VSEG_UPP <= cfg.V_MAX:
        raise ValueError('VSEG_LOW/VSEG_UPP must lie in the stored voltage grid')
    if not math.isclose(cfg.VSEG_UPP - cfg.VSEG_LOW, .30, rel_tol=0, abs_tol=1e-10):
        raise ValueError('VSEG_UPP must be VSEG_LOW + 0.30 for this input protocol')
    tail_steps = (cfg.VSEG_LOW - cfg.V_MIN) / cfg.DV
    if not math.isclose(tail_steps, round(tail_steps), abs_tol=1e-8) or cfg.TAIL_LEN != round(tail_steps) + 1:
        raise ValueError('TAIL_LEN and VSEG_LOW must align with the configured voltage grid')
    if not math.isfinite(cfg.VOLTAGE_REMOVAL_WINDOW) or cfg.VOLTAGE_REMOVAL_WINDOW < 0:
        raise ValueError('VOLTAGE_REMOVAL_WINDOW must be finite and nonnegative')
    if cfg.normalization_mode not in ('per_feature', 'per_channel'):
        raise ValueError('normalization_mode must be per_feature or per_channel')
    length = cfg.TAIL_LEN
    if group == 4:
        if (cfg.UPSTREAM_LEN != 60 or cfg.MAX_SEGS != 6 or cfg.SUBSEG_LEN != 10
                or cfg.UPSTREAM_START_IDX != cfg.TAIL_LEN or cfg.segment_length != cfg.TAIL_LEN + 60):
            raise ValueError('Group-4 padding dimensions must derive from the tail and six 0.05 V fragments')
        length = cfg.segment_length
    if group != 1:
        if not _positive_integer(cfg.num_epochs) or not _positive_integer(cfg.batch_size):
            raise ValueError('num_epochs and batch_size must be positive integers')
        for name in ('MLP_CONFIG', 'CNN1D_CONFIG', 'LSTM_CONFIG', 'TRANSFORMER_CONFIG'):
            values = getattr(cfg, name, None)
            if values is None:
                continue
            for key in ('in_ch', 'output_dim'):
                if values.get(key) != 1:
                    raise ValueError(f'{name}[{key!r}] must be 1 for the scalar capacity protocol')
            if list(values.get('quantiles', ())) != list(cfg.QUANTILES):
                raise ValueError(f'{name}["quantiles"] must match QUANTILES')
            for key in ('seq_len', 'max_seq_len'):
                if key in values and values[key] != length:
                    raise ValueError(f'{name}[{key!r}] must equal the derived input length {length}')
    if not Path(cfg.DATA_MAT_PATH).expanduser().resolve().is_file():
        raise FileNotFoundError(f'DATA_MAT_PATH does not exist: {cfg.DATA_MAT_PATH}')
    return length


def validate_output(path):
    """Keep newly generated artifacts out of code, data and archived results."""
    path = Path(path).expanduser().resolve()
    release = ROOT.parent
    allowed = (ROOT / 'runs', release / 'runs')
    if release.is_relative_to(path):
        raise ValueError('OUTPUT_DIR cannot be the release directory or its ancestor')
    if path.is_relative_to(release) and not any(path.is_relative_to(base) for base in allowed):
        raise ValueError('OUTPUT_DIR inside the release must be under runs; code/data/history are protected')
    if path.is_file():
        raise ValueError('OUTPUT_DIR must be a directory')
    return path


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def run_group(group, cfg):
    """Launch each configured model/seed in its own fresh Python process."""
    validate_config(group, cfg)
    base = validate_output(cfg.OUTPUT_DIR)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = base / f'{cfg.RUN_MODE}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(cfg.__file__).resolve()
    config_source = config_path.read_bytes()
    config_hash = hashlib.sha256(config_source).hexdigest()
    (output / 'config_source.py').write_bytes(config_source)
    manifest = {'run_mode': cfg.RUN_MODE, 'dataset': cfg.DATASET_NAME, 'group': group,
                'models': list(cfg.SELECTED_MODELS), 'random_seeds': list(cfg.random_seeds),
                'round_numbers': cfg.ROUND_NUMBERS, 'config_path': str(config_path),
                'config_sha256': config_hash, 'status': 'running', 'cases': []}
    manifest_path = output / ('run_manifest.json' if cfg.RUN_MODE == 'train' else 'check_manifest.json')
    write_json(manifest_path, manifest)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': str(cfg.THREADS),
           'MKL_NUM_THREADS': str(cfg.THREADS), 'OPENBLAS_NUM_THREADS': str(cfg.THREADS)}
    try:
        for model in cfg.SELECTED_MODELS:
            for seed in cfg.random_seeds:
                destination = output / cfg.DATASET_NAME / model / f'seed_{seed}'
                command = [sys.executable, '-B', '-u', str(Path(__file__).with_name('worker.py')),
                           str(group), str(config_path), model, str(seed), str(destination), config_hash]
                completed = subprocess.run(command, cwd=config_path.parent, env=env)
                manifest['cases'].append({'model': model, 'seed': seed, 'output': str(destination),
                                          'returncode': completed.returncode})
                write_json(manifest_path, manifest)
                if completed.returncode:
                    raise SystemExit(completed.returncode)
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        write_json(manifest_path, manifest)
    print(json.dumps({'status': 'completed', 'group': group, 'cases': len(manifest['cases']),
                      'output': str(output)}), flush=True)
    return output
