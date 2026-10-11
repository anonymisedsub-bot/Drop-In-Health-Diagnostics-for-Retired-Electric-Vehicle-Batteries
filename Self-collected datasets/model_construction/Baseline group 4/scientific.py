"""Run fixed-length real-partial padding baselines under within-batch ReBATT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
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

import config as cfg  # noqa: E402
from data import (  # noqa: E402
    apply_normalizer_with_padding,
    apply_training_prefix_mask,
    build_padding_xy,
    count_observed_upper_slots,
    fit_normalizer,
    get_voltage_config,
)
from train.train_sklearn import build_ml_models, eval_ml_model  # noqa: E402
from train.train_torch import eval_model, train_val_one_model  # noqa: E402
from utils import set_seed  # noqa: E402


BASELINE_GROUP = "Baseline group 4"
FIXED_COMBOS = [
    ("SL", "mlp"),
    ("SL", "cnn"),
    ("SL", "cgb"),
    ("SL", "lgbm"),
]
ML_MODELS = {"xgb", "gpr", "rf", "cgb", "lgbm"}
DL_MODELS = {"mlp", "cnn", "lstm", "gru", "transformer"}


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
    model_path = output_dir / "best.pt"
    if model_name in ML_MODELS:
        joblib.dump(handle, model_path)
    else:
        state = {key: value.detach().cpu().clone() for key, value in handle.state_dict().items()}
        torch.save(state, model_path)
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
    available_counts: np.ndarray,
):
    if model_name in ML_MODELS:
        if model_name in {"cgb", "lgbm"}:
            x_train = apply_training_prefix_mask(
                x_train,
                cfg.missing_sim_random,
                np.random.RandomState(seed),
                get_voltage_config(),
                available_counts,
            )
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

    if model_name in DL_MODELS:
        handle = train_val_one_model(
            model_name,
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            cl_config=(cfg.missing_sim_curriculum if model_name in {"mlp", "cnn"} else None),
            available_counts=available_counts,
        )
        evaluation = eval_model(handle, x_test, y_test, cfg, normalizer, clip01=False, name_tag="test")
        return handle, _point_prediction(evaluation["test"]["y_pred_den"])

    raise ValueError(f"Unsupported baseline model: {model_name}")


def run_one_case(data, split, roles, seed: int, method: str, model_name: str, output_dir: Path) -> None:
    if method != "SL":
        raise ValueError("The padding-input baseline uses supervised learning only.")
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

    x_train_raw, y_train_raw = build_padding_xy(
        data, roles.labelled_indices, "labelled", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_val_raw, y_val_raw = build_padding_xy(
        data, split.validation_indices, "validation", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    x_test_raw, y_test_raw = build_padding_xy(
        data, split.test_indices, "test", cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW
    )
    available_counts = count_observed_upper_slots(x_train_raw, get_voltage_config())

    normalizer = fit_normalizer(x_train_raw, y_train_raw, cfg.normalization_mode)
    write_normalization_snapshot(output_dir / "normalization.json", minmax=normalizer)
    x_train, y_train = apply_normalizer_with_padding(x_train_raw, y_train_raw, normalizer)
    x_val, y_val = apply_normalizer_with_padding(x_val_raw, y_val_raw, normalizer)
    x_test, y_test = apply_normalizer_with_padding(x_test_raw, y_test_raw, normalizer)

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
        available_counts,
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

