"""Run group-3 DA/SSL-PL baselines under the shared ReBATT within-batch protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
MODEL_CONSTRUCTION_ROOT = HERE.parent
if str(MODEL_CONSTRUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_CONSTRUCTION_ROOT))

from baseline_support.rebatt_protocol import (  # noqa: E402
    make_round_roles,
    make_batch_split,
    round_numbers,
    write_budget_manifest,
    write_config_snapshot,
    write_input_protocol_snapshot,
    write_normalization_snapshot,
    write_split_manifest,
)
from baseline_support.baseline_group3 import (  # noqa: E402
    GROUP3_METHODS, GROUP3_COMBOS, select_group3_methods,
)

import config as cfg  # noqa: E402
from data import apply_normalizer, build_tail_xy, fit_normalizer  # noqa: E402
from train.train_da import DATrainResult, predict_q50, train_da_baseline  # noqa: E402
from utils import set_seed  # noqa: E402


BASELINE_GROUP = "Baseline group 3"
FIXED_COMBOS = GROUP3_COMBOS
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


def _save_model(
    result: DATrainResult,
    method: str,
    model_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in result.model.state_dict().items()
    }
    torch.save(state, output_dir / "best.pt")
    for state_name, auxiliary_state in result.auxiliary_states.items():
        torch.save(auxiliary_state, output_dir / f"{state_name}.pt")
    (output_dir / "model_meta.json").write_text(json.dumps({"baseline_group": BASELINE_GROUP, "method": method, "model": model_name, **result.metadata}, indent=2), encoding="utf-8")


def _fit_and_predict(
    method: str,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
):
    if model_name != "mlp":
        raise ValueError(f"Group-3 methods require the MLP backbone, got {model_name}")
    result = train_da_baseline(
        method,
        x_train,
        y_train,
        x_target,
        x_val,
        y_val,
        cfg,
        seed,
    )
    return result, predict_q50(result.model, x_test, cfg)


def _denormalize_y(values: np.ndarray, normalizer) -> np.ndarray:
    ymin = float(np.asarray(normalizer["ymin"]).reshape(-1)[0])
    ymax = float(np.asarray(normalizer["ymax"]).reshape(-1)[0])
    return np.asarray(values, dtype=float).reshape(-1) * (ymax - ymin) + ymin


def run_one_case(data, split, roles, seed: int, method: str, model_name: str, output_dir: Path) -> None:
    if (method, model_name) not in GROUP3_COMBOS:
        raise ValueError(f"Unsupported group-3 combination: {(method, model_name)}")
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
    x_target_raw, y_target_raw = build_tail_xy(
        data, roles.unlabelled_indices, "unlabelled", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_val_raw, y_val_raw = build_tail_xy(
        data, split.validation_indices, "validation", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_test_raw, y_test_raw = build_tail_xy(
        data, split.test_indices, "test", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    if y_target_raw is not None:
        raise ValueError("Unlabelled target-domain samples must not expose capacity labels")

    normalizer = fit_normalizer(x_train_raw, y_train_raw, cfg.normalization_mode)
    write_normalization_snapshot(output_dir / "normalization.json", minmax=normalizer)
    x_train, y_train = apply_normalizer(x_train_raw, y_train_raw, normalizer)
    x_target, _ = apply_normalizer(x_target_raw, None, normalizer)
    x_val, y_val = apply_normalizer(x_val_raw, y_val_raw, normalizer)
    x_test, _ = apply_normalizer(x_test_raw, None, normalizer)

    set_seed(seed)
    result, y_pred_normalized = _fit_and_predict(
        method,
        model_name,
        x_train,
        y_train,
        x_target,
        x_val,
        y_val,
        x_test,
        seed,
    )
    y_pred = _denormalize_y(y_pred_normalized, normalizer)
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
                "unlabelled_count": int(len(roles.unlabelled_indices)),
                "validation_count": int(len(split.validation_indices)),
                "training": result.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_model(result, method, model_name, output_dir)
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

