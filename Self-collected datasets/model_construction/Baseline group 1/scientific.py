"""Run supervised tail-only baselines under the shared ReBATT within-batch protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_CONSTRUCTION_ROOT = HERE.parent
if str(MODEL_CONSTRUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_CONSTRUCTION_ROOT))

from baseline_support.rebatt_protocol import (  # noqa: E402
    build_voltage_windows,
    complete_partial_tail_mask,
    make_round_roles,
    make_batch_split,
    round_numbers,
    write_budget_manifest,
    write_config_snapshot,
    write_input_protocol_snapshot,
    write_normalization_snapshot,
    write_split_manifest,
)

import config as cfg  # noqa: E402
from data import apply_normalizer, build_tail_xy, fit_normalizer  # noqa: E402
from train.train_sklearn import build_ml_models, eval_ml_model  # noqa: E402
from utils import set_seed  # noqa: E402


BASELINE_GROUP = "Baseline group 1"
FIXED_COMBOS = [
    ("SL", "xgb"),
    ("SL", "rf"),
    ("SL", "cgb"),
    ("SL", "lgbm"),
]
ML_MODELS = {"xgb", "rf", "cgb", "lgbm"}


def partial_tail_eligible_mask(data, vseg_low: float) -> np.ndarray:
    """Return the shared pre-split completeness mask for the configured tail."""

    return complete_partial_tail_mask(data, vseg_low)


def write_cohort_manifest(path: Path, data, batch: int, eligible_mask: np.ndarray) -> None:
    """Record which cells were excluded before the seeded split and why."""

    batch_mask = data.experiment_batch == batch
    excluded = np.flatnonzero(batch_mask & ~eligible_mask)
    payload = {
        "split_policy": "eligible_partial_tail_resplit",
        "criterion": "partial Q/V finite through tail endpoint and IC finite through preceding interval",
        "tail_voltage_min": float(data.voltage_grid[0]),
        "tail_voltage_max": float(data.voltage_grid[int(build_voltage_windows(
            data.voltage_grid, cfg.VSEG_LOW).tail_indices[-1])]),
        "dataset_batch": int(batch),
        "source_count": int(batch_mask.sum()),
        "eligible_count": int(np.count_nonzero(batch_mask & eligible_mask)),
        "excluded_count": int(len(excluded)),
        "excluded_samples": [
            {
                "global_index": int(index),
                "source_batch": int(data.source_batch[index]),
                "source_matrix_column": int(data.source_column[index]),
                "battery_id": int(data.battery_id[index]),
            }
            for index in excluded
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _point_prediction(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.ndim == 2 and prediction.shape[1] == 3:
        return prediction[:, 1]
    return prediction.reshape(-1)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = y_pred - y_true
    safe = np.maximum(np.abs(y_true), np.finfo(float).eps)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MAPE": float(np.mean(np.abs(error) / safe) * 100.0),
        "RMSPE": float(np.sqrt(np.mean((error / safe) ** 2)) * 100.0),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "STD": float(np.std(error, ddof=0)) if error.size > 1 else 0.0,
    }


def _save_model(handle, model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name not in ML_MODELS:
        raise ValueError(f"Unsupported ML baseline: {model_name}")
    joblib.dump(handle, output_dir / "best.pt")
    (output_dir / "model_meta.json").write_text(
        json.dumps({"baseline_group": BASELINE_GROUP, "method": "SL", "model": model_name}, indent=2),
        encoding="utf-8",
    )


def _fit_and_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    normalizer: dict[str, float],
    seed: int,
):
    if model_name not in ML_MODELS:
        raise ValueError(f"Unsupported ML baseline: {model_name}")
    estimator = build_ml_models(cfg)[model_name]()
    evaluation = eval_ml_model(
        estimator,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        normalizer,
        quantiles=cfg.QUANTILES,
    )
    return estimator, _point_prediction(evaluation["test"]["y_pred_den"])


def run_one_case(data, split, roles, seed: int, method: str, model_name: str, output_dir: Path) -> None:
    if method != "SL":
        raise ValueError("Tail-only comparative baselines use supervised learning only.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir.parent / "split_manifest.json", data, split)
    write_config_snapshot(output_dir.parent / "config_snapshot.json", cfg)
    write_input_protocol_snapshot(
        output_dir.parent / "input_protocol.json",
        data,
        split.test_indices,
        cfg.VSEG_LOW,
        cfg.VOLTAGE_REMOVAL_WINDOW,
    )
    write_budget_manifest(output_dir / "budget_manifest.json", data, roles)

    x_train_raw, y_train_raw = build_tail_xy(
        data, roles.labelled_indices, "labelled", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_val_raw, y_val_raw = build_tail_xy(
        data, split.validation_indices, "validation", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_test_raw, y_test_raw = build_tail_xy(
        data, split.test_indices, "test", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )

    normalizer = fit_normalizer(x_train_raw, y_train_raw, cfg.normalization_mode)
    write_normalization_snapshot(output_dir / "normalization.json", minmax=normalizer)
    x_train, y_train = apply_normalizer(x_train_raw, y_train_raw, normalizer)
    x_val, y_val = apply_normalizer(x_val_raw, y_val_raw, normalizer)
    x_test, y_test = apply_normalizer(x_test_raw, y_test_raw, normalizer)

    set_seed(seed)
    handle, y_pred = _fit_and_predict(
        model_name,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        normalizer,
        seed,
    )
    y_true = np.asarray(y_test_raw).reshape(-1)
    metrics = _regression_metrics(y_true, y_pred)
    (output_dir / "training_log.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "method": method,
                "model": model_name,
                "seed": int(seed),
                "labelled_count": int(len(roles.labelled_indices)),
                "validation_count": int(len(split.validation_indices)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_model(handle, model_name, output_dir)
    test_idx = np.asarray(split.test_indices, dtype=int)
    performance = pd.DataFrame(
        {
            "global_index": test_idx,
            "experiment_batch": data.experiment_batch[test_idx],
            "source_batch": data.source_batch[test_idx],
            "source_matrix_column": data.source_column[test_idx],
            "filter_original_mat_column": data.filter_original_column[test_idx],
            "battery_id": data.battery_id[test_idx],
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_pred - y_true,
        }
    )
    performance.to_excel(output_dir / "summary_performance.xlsx", index=False)

    summary = {
        "baseline_group": BASELINE_GROUP,
        "method": method,
        "model": model_name,
        "dataset_batch": int(split.batch),
        "seed": int(seed),
        "round_number": int(roles.round_number),
        "labelled_fraction": float(roles.labelled_fraction),
        "pool_count": int(len(split.pool_order)),
        "validation_count": int(len(split.validation_indices)),
        "test_count": int(len(split.test_indices)),
        "labelled_count": int(len(roles.labelled_indices)),
        "unlabelled_count": int(len(roles.unlabelled_indices)),
        **metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Batch {split.batch} seed {seed} iter {roles.round_number} {model_name}: "
        f"test n={len(split.test_indices)}, MAPE={metrics['MAPE']:.3f}%, "
        f"RMSPE={metrics['RMSPE']:.3f}%, R2={metrics['R2']:.4f}",
        flush=True,
    )

