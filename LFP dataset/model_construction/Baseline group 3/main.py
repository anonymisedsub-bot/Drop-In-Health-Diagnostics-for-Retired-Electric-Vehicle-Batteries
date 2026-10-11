"""Run LFP Baseline group 3 using only the adjacent config.py."""
from pathlib import Path
import sys

# Preserve the group directory first for its scientific modules.
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
import config
from baseline_support.runner import run_group


def main():
    if len(sys.argv) != 1:
        raise SystemExit('Edit config.py; main.py accepts no command-line settings.')
    return run_group(3, config)


if __name__ == '__main__':
    raise SystemExit(main())
