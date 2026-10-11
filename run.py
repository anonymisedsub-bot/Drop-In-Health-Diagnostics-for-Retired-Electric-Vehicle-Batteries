"""Release checks and simulation shortcuts.

Run LFP/NMC models through their model_construction/Proposed framework/main.py or each Baseline
group's main.py. Their adjacent config.py controls the complete experiment.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULES = {'regrouping': 'Sorting and regrouping simulation',
           'tea': 'Techno-economic-environmental assessment'}

def execute(arguments: list[str], workdir: Path = ROOT) -> None:
    print('Running: ' + subprocess.list2cmdline([sys.executable, *arguments]), flush=True)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    subprocess.run([sys.executable, *arguments], cwd=workdir, env=env, check=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'verify', *MODULES])
    args, extra = parser.parse_known_args(argv)
    if args.command == 'verify' and extra:
        parser.error(f'{args.command} accepts no additional arguments: ' + ' '.join(extra))
    if args.command in MODULES:
        execute([str(ROOT / MODULES[args.command] / 'run.py'), *extra])
    elif args.command == 'check':
        execute([str(ROOT / 'tools' / 'check_environment.py'), *extra])
    elif args.command == 'verify':
        execute([str(ROOT / 'tools' / 'release_integrity.py')])
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
