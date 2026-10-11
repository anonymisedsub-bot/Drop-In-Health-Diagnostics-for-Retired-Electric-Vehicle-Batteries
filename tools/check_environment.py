"""Inspect required packages without changing the current Python environment."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform

REQUIRED = ('numpy', 'scipy', 'pandas', 'torch', 'matplotlib', 'h5py', 'openpyxl', 'scikit-learn', 'Pillow')
BASELINE_REQUIRED = ('xgboost', 'lightgbm', 'catboost', 'joblib', 'tqdm')

def inspect_environment(include_baselines=False):
    packages = {}
    missing = []
    required = REQUIRED + (BASELINE_REQUIRED if include_baselines else ())
    for name in required:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            if name in required:
                missing.append(name)
    device = {'cuda_available': False}
    if importlib.util.find_spec('torch'):
        import torch
        device = {'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'device': device, 'missing_required': missing}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', action='store_true',
                        help='Also require the optional baseline packages.')
    result = inspect_environment(include_baselines=parser.parse_args().baselines)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['missing_required']))
