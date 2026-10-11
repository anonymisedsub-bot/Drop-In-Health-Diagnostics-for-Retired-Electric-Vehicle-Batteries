"""Run NMC Baseline group 3 using the adjacent config.py."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import config as cfg
from baseline_support.runner import run_group


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Edit config.py to select data, models and training settings; main.py accepts no arguments.")
    return run_group(3, cfg)


if __name__ == "__main__":
    main()
