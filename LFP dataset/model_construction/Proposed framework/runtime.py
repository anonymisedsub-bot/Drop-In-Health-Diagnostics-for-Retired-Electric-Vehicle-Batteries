"""Output protection, effective settings, and immutable checkpoint settings."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_output(path):
    output = Path(path).resolve()
    release = ROOT.parent.resolve()
    allowed = (ROOT / 'runs', release / 'runs')
    if output == release or (release in output.parents and not any(output == item.resolve() or item.resolve() in output.parents for item in allowed)):
        raise ValueError('Output targets a protected bundled artifact; choose this dataset runs directory, release runs, or an external directory')
    return output


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not a configuration value: {type(value).__name__}')


def config_values(config):
    values = {}
    for name, value in vars(config).items():
        if name.startswith('_') or callable(value):
            continue
        try:
            values[name] = json.loads(json.dumps(value, default=_json_default))
        except TypeError:
            continue
    return values


def save_config(path, config):
    Path(path).write_text(json.dumps(config_values(config), indent=2), encoding='utf-8')


def new_campaign(config):
    output = validate_output(config.OUTPUT_DIR) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    save_config(output / 'config_snapshot.json', config)
    return output


def selected_rounds(requested, available):
    available = tuple(available)
    if requested is None:
        return available
    selected = tuple(requested)
    if not selected or len(selected) != len(set(selected)) or any(type(number) is not int or number not in available for number in selected):
        raise ValueError(f'ROUND_NUMBERS must contain unique integers from {available}')
    return tuple(sorted(selected))


def validate_settings(config):
    if config.RUN_MODE not in ('train', 'infer', 'check'):
        raise ValueError('RUN_MODE must be train, infer, or check')
    for name in ('THREADS', 'batch_size', 'num_epochs', 'patience_epochs'):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    seeds = tuple(config.random_seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('random_seeds must contain unique nonnegative integers')
    if config.scheduler['policy'] not in ('plateau', 'cosine_epoch', 'none'):
        raise ValueError('scheduler.policy must be plateau, cosine_epoch, or none')


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class FrozenConfig:
    """Separate read-only checkpoint namespace, including nested model settings."""
    def __init__(self, values):
        object.__setattr__(self, '_values', _freeze(values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        raise AttributeError('Checkpoint configuration is immutable')
