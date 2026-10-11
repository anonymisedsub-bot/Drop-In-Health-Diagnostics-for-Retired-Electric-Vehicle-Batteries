"""Internal orchestration for a single group's config-driven main.py."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from baseline_support.worker import json_config, sha256

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    1: ('XGB', 'RF', 'CGB', 'LGBM'),
    2: ('MLP', 'CNN', 'LSTM', 'Transformer'),
    3: ('DA-AL', 'DA-MMD', 'DA-CORAL', 'SSL-PL'),
    4: ('MLP', 'CNN', 'CGB', 'LGBM'),
}


def validate_config(group, cfg):
    if cfg.RUN_MODE not in ('train', 'check'):
        raise ValueError("RUN_MODE must be 'train' or 'check'")
    if (not isinstance(cfg.SELECTED_MODELS, tuple) or not cfg.SELECTED_MODELS
            or any(name not in MODELS[group] for name in cfg.SELECTED_MODELS)
            or len(set(cfg.SELECTED_MODELS)) != len(cfg.SELECTED_MODELS)):
        raise ValueError(f'SELECTED_MODELS must be a nonempty tuple drawn from {MODELS[group]}')
    if (not isinstance(cfg.random_seeds, tuple) or not cfg.random_seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 - 123 for seed in cfg.random_seeds)
            or len(set(cfg.random_seeds)) != len(cfg.random_seeds)):
        raise ValueError('random_seeds must contain unique integers between 0 and 2**32-124')
    if type(cfg.THREADS) is not int or cfg.THREADS < 1:
        raise ValueError('THREADS must be a positive integer')
    if cfg.ROUND_NUMBERS is not None and (
            not isinstance(cfg.ROUND_NUMBERS, tuple) or not cfg.ROUND_NUMBERS
            or any(type(number) is not int or number < 1 for number in cfg.ROUND_NUMBERS)
            or len(set(cfg.ROUND_NUMBERS)) != len(cfg.ROUND_NUMBERS)):
        raise ValueError('ROUND_NUMBERS must be None or a tuple of unique positive integers')
    if group != 1 and cfg.device not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu' or 'cuda'")
    for name in ('num_tr_initial', 'num_tr_per_iter', 'TAIL_LEN'):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    if group != 1:
        for name in ('num_epochs', 'batch_size'):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
    if not 0 <= cfg.al_pool_ratio < 1 or not 0 < cfg.al_val_ratio < 1:
        raise ValueError('al_pool_ratio and al_val_ratio must define valid split proportions')


def validate_output_base(path):
    """Only run directories within the release, or a directory outside it."""
    result = Path(path).resolve()
    repository = ROOT.parent
    allowed = ((ROOT / 'runs').resolve(), (repository / 'runs').resolve())
    if (result == repository or result in repository.parents
            or (repository in result.parents
                and not any(base == result or base in result.parents for base in allowed))):
        raise ValueError('OUTPUT_DIR targets protected bundled artifacts; use runs/ or an external directory')
    if result.exists() and not result.is_dir():
        raise ValueError('OUTPUT_DIR must be a directory')
    return result


def worker(request, threads):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(threads)
    process = subprocess.run(
        [sys.executable, '-B', str(ROOT / 'baseline_support/worker.py')],
        input=json.dumps(request), text=True, capture_output=True,
        cwd=ROOT, env=env, encoding='utf-8', errors='replace',
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    try:
        return json.loads(process.stdout), process.stderr
    except json.JSONDecodeError as error:
        raise RuntimeError(f'Invalid baseline worker response: {process.stdout}') from error


def run_group(group, cfg):
    """Run the configured models, seeds and budgets without changing cfg."""
    output = None
    manifest = None
    owns_output = False
    try:
        validate_config(group, cfg)
        output_base = validate_output_base(cfg.OUTPUT_DIR) if cfg.RUN_MODE == 'train' else None
        request_base = dict(group=group, config_path=str(Path(cfg.__file__).resolve()),
                            config_sha256=sha256(cfg.__file__))
        checks = []
        for seed in cfg.random_seeds:
            check, _ = worker(dict(request_base, command='check', seed=seed), cfg.THREADS)
            checks.append(check)
        if cfg.RUN_MODE == 'check':
            print(json.dumps({'command': 'check', 'dataset': 'LFP', 'checks': checks}, indent=2))
            return 0
        missing = sorted({module for check in checks for module, present in check['dependencies'].items() if not present})
        if missing:
            raise ValueError(f'Missing optional dependencies: {", ".join(missing)}. Install requirements-baselines.txt.')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = output_base / f'{stamp}_{uuid.uuid4().hex[:8]}'
        # Claim before writing: a colliding process can never have its manifest overwritten.
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        manifest = dict(command='train', dataset='LFP', group=group, status='running',
                        config_sha256=request_base['config_sha256'], effective_config=json_config(cfg),
                        checks=checks, cases=[])
        manifest_path = output / 'run_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (output / 'config_snapshot.py').write_bytes(Path(cfg.__file__).read_bytes())
        for model in cfg.SELECTED_MODELS:
            for check in checks:
                seed = check['seed']
                case_root = output / model / f'seed_{seed}'
                request = dict(request_base, command='train', seed=seed, model=model,
                               dataset_sha256=check['dataset_sha256'], output=str(case_root))
                print(f'LFP group {group}: {model}, seed {seed}', file=sys.stderr, flush=True)
                result, log = worker(request, cfg.THREADS)
                (case_root / 'worker.log').write_text(log, encoding='utf-8')
                manifest['cases'].append(result)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        manifest['status'] = 'completed'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'command': 'train', 'status': 'completed', 'output': str(output),
                          'cases': len(manifest['cases'])}, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        if owns_output and manifest is not None:
            manifest.update(status='failed', error=str(error))
            (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'baseline error: {error}', file=sys.stderr)
        return 2
