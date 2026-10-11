"""Config-driven Proposed training and inference for NMC_A or NMC_B."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

import config as cfg
from data import (
    apply_normalizer_cap,
    apply_normalizer_tail,
    build_ordered_windows,
    build_subset,
    fit_normalizer_cap,
    fit_normalizer_tail,
    inverse_cap,
)
from losses import mape, r2_score, rmspe
from model import ConditionalDecoder
from train_model import eval_observed, train_epoch


MODEL_ROOT = Path(__file__).resolve().parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.rebatt_results import iteration_name
from common.rebatt_protocol import (
    make_round_roles,
    make_batch_split,
    round_numbers,
    write_budget_manifest,
    write_config_snapshot,
    write_normalization_snapshot,
    write_split_manifest,
    write_input_protocol_snapshot,
)


def set_global_seed(seed: int) -> None:
    """Set all random-number generators used by one model fit."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BatterySet(torch.utils.data.Dataset):
    """Role-aware dataset which keeps unlabelled capacity targets absent."""

    def __init__(self, pack):
        self.X_tail = torch.from_numpy(pack["X_tail"]).float()
        self.Z_meta = torch.from_numpy(pack["z_meta"]).float()
        self.y_cap = (
            None
            if pack["y_cap"] is None
            else torch.from_numpy(pack["y_cap"][:, 0]).float()
        )
        self.S_list = [
            [torch.from_numpy(segment).float() for segment in segments]
            for segments in pack["S_list"]
        ]

    def __len__(self):
        return self.X_tail.shape[0]

    def __getitem__(self, index):
        target = None if self.y_cap is None else self.y_cap[index]
        return self.X_tail[index], self.S_list[index], self.Z_meta[index], target


def collate_varlen(batch):
    X_tail, S_list, Z_meta, targets = zip(*batch)
    if all(target is None for target in targets):
        target_batch = None
    elif any(target is None for target in targets):
        raise ValueError("A batch cannot mix labelled and unlabelled samples")
    else:
        target_batch = torch.stack(targets)
    return (
        torch.stack(X_tail),
        list(S_list),
        torch.stack(Z_meta),
        target_batch,
    )


class InfiniteIterator:
    """Cycle through a loader so labelled and unlabelled batches can be paired."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iterator = iter(dataloader)

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def __len__(self):
        return len(self._dataloader)


def _apply_norm_to_pack(pack, tail_norm, cap_norm):
    pack = dict(pack)
    if pack["X_tail"].shape[0] == 0:
        return pack
    pack["X_tail"] = apply_normalizer_tail(pack["X_tail"], tail_norm)
    mins = tail_norm["mins"].min(axis=1, keepdims=True)
    maxs = tail_norm["maxs"].max(axis=1, keepdims=True)
    denominator = np.clip(maxs - mins, 1e-8, None)
    pack["S_list"] = [
        [((segment - mins) / denominator).astype(np.float32) for segment in segments]
        for segments in pack["S_list"]
    ]
    pack["y_cap"] = apply_normalizer_cap(pack["y_cap"], cap_norm)
    return pack


def _apply_step_decay_if_needed(optimizer, epoch_index):
    schedule = cfg.scheduler
    if schedule.get("policy", "plateau") != "plateau":
        return
    step_every = int(schedule.get("epoch_step", 10))
    if step_every <= 0 or (epoch_index + 1) % step_every:
        return
    gamma = float(schedule.get("step_gamma", 0.95))
    minimum = float(schedule.get("min_lr", 0.0))
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * gamma, minimum)


def format_epoch_progress(epoch, train_metrics, validation_metrics, learning_rate):
    """Match the reference Proposed joint-training console summary."""

    return (
        f"[AL][JT] epoch {epoch} "
        f"Tr_S(cap_mse={train_metrics['cap_mse']:.4f}) "
        f"Tr_T(tail_mse_t={train_metrics['tail_mse_t']:.4f}, "
        f"DA={train_metrics.get('da_loss', 0.0):.4f}) "
        f"Val(total={validation_metrics['total']:.4f}) | lr={learning_rate:.3e}"
    )


def _make_loader(pack, seed, shuffle):
    return DataLoader(
        BatterySet(pack),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_varlen,
        num_workers=0,
        pin_memory=cfg.device.startswith("cuda"),
        generator=make_torch_generator(seed) if shuffle else None,
        worker_init_fn=seed_worker if shuffle else None,
    )


def train_one_round(data, roles, windows, idx_tail_hi, seed, out_dir: Path):
    """Fit and evaluate one fresh model at one nested label budget."""

    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    sup_pack = build_subset(
        data,
        roles.labelled_indices,
        cfg,
        windows,
        idx_tail_hi,
        role="labelled",
    )
    val_pack = build_subset(
        data,
        roles.validation_indices,
        cfg,
        windows,
        idx_tail_hi,
        role="validation",
    )
    test_pack = build_subset(
        data,
        roles.test_indices,
        cfg,
        windows,
        idx_tail_hi,
        role="test",
    )
    ssl_pack = (
        build_subset(
            data,
            roles.unlabelled_indices,
            cfg,
            windows,
            idx_tail_hi,
            role="unlabelled",
        )
        if len(roles.unlabelled_indices)
        else None
    )

    tail_norm = fit_normalizer_tail(
        sup_pack["X_tail"],
        cfg.normalization_mode,
    )
    cap_norm = fit_normalizer_cap(sup_pack["y_cap"])
    write_normalization_snapshot(
        out_dir / "normalization.json",
        tail=tail_norm,
        capacity=cap_norm,
    )
    sup_normalized = _apply_norm_to_pack(sup_pack, tail_norm, cap_norm)
    val_normalized = _apply_norm_to_pack(val_pack, tail_norm, cap_norm)
    test_normalized = _apply_norm_to_pack(test_pack, tail_norm, cap_norm)
    ssl_normalized = (
        _apply_norm_to_pack(ssl_pack, tail_norm, cap_norm)
        if ssl_pack is not None
        else None
    )

    supervised_iterator = InfiniteIterator(
        _make_loader(sup_normalized, seed + 1000, shuffle=True)
    )
    validation_loader = _make_loader(val_normalized, seed, shuffle=False)
    ssl_iterator = (
        InfiniteIterator(_make_loader(ssl_normalized, seed + 2000, shuffle=True))
        if ssl_normalized is not None
        else None
    )

    network = ConditionalDecoder(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    policy = cfg.scheduler.get("policy", "plateau")
    if policy == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler.get("factor", 0.5),
            patience=cfg.scheduler.get("patience", 5),
            min_lr=cfg.scheduler["min_lr"],
        )
    elif policy == "cosine_epoch":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.scheduler["min_lr"],
        )
    else:
        scheduler = None

    best_loss = float("inf")
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }
    bad_epochs = 0
    log_path = out_dir / "log.jsonl"
    log_path.write_text("", encoding="utf-8")
    print(f"[AL] Joint-train ({cfg.num_epochs} epochs)", flush=True)

    for epoch in range(cfg.num_epochs):
        train_metrics, weights = train_epoch(
            network,
            supervised_iterator,
            ssl_iterator,
            optimizer,
            current_global_epoch=epoch,
        )
        validation_metrics = eval_observed(
            network,
            validation_loader,
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation_metrics["total"])
            _apply_step_decay_if_needed(optimizer, epoch)
        elif scheduler is not None:
            scheduler.step()

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": validation_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                        "weights": weights,
                    }
                )
                + "\n"
            )
        print(
            format_epoch_progress(
                epoch, train_metrics, validation_metrics,
                optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )

        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience_epochs:
                print(f"[AL] Early stop joint-train at epoch {epoch}", flush=True)
                break

    torch.save(best_state, out_dir / "best.pt")
    network.load_state_dict(best_state)
    network.eval().to(cfg.device)

    @torch.no_grad()
    def predict(pack):
        loader = _make_loader(pack, seed, shuffle=False)
        quantiles = cfg.model["quantiles"]
        median_index = quantiles.index(0.5)
        lower_values, median_values, upper_values = [], [], []
        for X_tail, segment_lists, Z_meta, _ in loader:
            X_tail = X_tail.to(cfg.device, non_blocking=True)
            Z_meta = Z_meta.to(cfg.device, non_blocking=True)
            stacked, owners = [], []
            for sample_index, segments in enumerate(segment_lists):
                stacked.extend(segments)
                owners.extend([sample_index] * len(segments))
            if stacked:
                segments = torch.stack(stacked).to(cfg.device, non_blocking=True)
                segment_indices = torch.tensor(
                    owners,
                    dtype=torch.long,
                    device=cfg.device,
                )
            else:
                segments = torch.empty(0, 2, cfg.SEG_LEN, device=cfg.device)
                segment_indices = torch.empty(0, dtype=torch.long, device=cfg.device)
            _, capacity, _ = network(X_tail, segments, segment_indices, Z_meta)
            lower_values.append(capacity[:, 0].cpu().numpy())
            median_values.append(capacity[:, median_index].cpu().numpy())
            upper_values.append(capacity[:, -1].cpu().numpy())

        lower = np.concatenate(lower_values)[:, None]
        median = np.concatenate(median_values)[:, None]
        upper = np.concatenate(upper_values)[:, None]
        y_true = inverse_cap(pack["y_cap"], cap_norm).squeeze(-1)
        y_pred = inverse_cap(median, cap_norm).squeeze(-1)
        lower = inverse_cap(lower, cap_norm).squeeze(-1)
        upper = inverse_cap(upper, cap_norm).squeeze(-1)
        z_distance = stats.norm.ppf(quantiles[-1]) - stats.norm.ppf(quantiles[0])
        sigma = (upper - lower) / (z_distance + 1e-12)
        return y_true, y_pred, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma

    y_true, y_pred, lower_95, upper_95 = predict(test_normalized)
    metrics = {
        "MAPE": mape(y_true, y_pred),
        "RMSPE": rmspe(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    performance = pd.DataFrame(
        {
            "dataset": cfg.DATASET_NAME,
            "global_index": roles.test_indices,
            "experiment_batch": data.experiment_batch[roles.test_indices],
            "source_batch": data.source_batch[roles.test_indices],
            "source_matrix_column": data.source_column[roles.test_indices],
            "battery_id": data.battery_id[roles.test_indices],
            "filter_original_mat_column": data.filter_original_column[roles.test_indices],
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_lo95": lower_95,
            "y_pred_hi95": upper_95,
        }
    )
    performance["error"] = performance["y_pred"] - performance["y_true"]
    performance.to_excel(out_dir / "summary_performance.xlsx", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def load_training_data():
    """Read the configured processed MAT directly; no historical settings are loaded."""
    from common.rebatt_separate import load_separate_batch
    if cfg.DATASET_NAME not in cfg.DATASET_PRESETS:
        raise ValueError('DATASET_NAME must be NMC_A or NMC_B')
    return load_separate_batch(cfg.DATA_MAT_PATH, cfg.SOURCE_BATCH)


def make_training_split(data, seed):
    return make_batch_split(
        data, batch=cfg.MODEL_BATCH, seed=seed,
        pool_ratio=cfg.POOL_TARGET_RATIO, pool_divisor=cfg.POOL_DIVISOR,
        initial_labelled_count=cfg.INITIAL_LABELLED_COUNT,
        acquisition_size=cfg.ACQUISITION_SIZE, vseg_low=cfg.VSEG_LOW,
    )


def run_one_seed(data, seed: int, results_root: Path):
    """Fit a fresh model at every configured nested label budget."""
    from runtime import selected_rounds
    batch = cfg.MODEL_BATCH
    split = make_training_split(data, seed)
    rounds = selected_rounds(cfg.ROUND_NUMBERS, round_numbers(split))
    write_split_manifest(results_root / "split_manifest.json", data, split)
    write_config_snapshot(results_root / "config_snapshot.json", cfg)
    write_input_protocol_snapshot(
        results_root / "input_protocol.json", data, split.test_indices,
        cfg.VSEG_LOW, cfg.VOLTAGE_REMOVAL_WINDOW,
    )
    windows, idx_tail_hi, _ = build_ordered_windows(cfg, data.voltage_grid)
    all_budgets = []
    for round_number in rounds:
        roles = make_round_roles(split, round_number)
        output_dir = results_root / iteration_name(round_number)
        print(
            f"\n[AL] === Batch {batch} Seed {seed} Iteration {round_number} "
            f"(train={len(roles.labelled_indices)}, pool={len(roles.unlabelled_indices)}) ===",
            flush=True,
        )
        write_budget_manifest(output_dir / "budget_manifest.json", data, roles)
        metrics = train_one_round(
            data,
            roles,
            windows,
            idx_tail_hi,
            seed,
            output_dir,
        )
        summary = {
            "dataset_batch": int(batch),
            "seed": int(seed),
            "round_number": int(round_number),
            "labelled_fraction": float(roles.labelled_fraction),
            "labelled_count": int(len(roles.labelled_indices)),
            "unlabelled_count": int(len(roles.unlabelled_indices)),
            **metrics,
        }
        (output_dir / "summary_iter.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[AL][TEST] Batch {batch} Seed {seed} Iteration {round_number} | "
            f"train={summary['labelled_count']} | "
            f"MAPE={summary['MAPE']:.3f}% RMSPE={summary['RMSPE']:.3f}% "
            f"R2={summary['R2']:.4f}",
            flush=True,
        )
        all_budgets.append(summary)
    (results_root / "budget_summary.json").write_text(
        json.dumps(all_budgets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[AL] Batch {batch} Seed {seed} complete: test results by iteration", flush=True)
    for summary in all_budgets:
        print(
            f"  iter {summary['round_number']} train={summary['labelled_count']} "
            f"MAPE={summary['MAPE']:.3f}% RMSPE={summary['RMSPE']:.3f}% "
            f"R2={summary['R2']:.4f}",
            flush=True,
        )


def main():
    """Execute RUN_MODE using only config.py settings."""
    from runtime import new_campaign, selected_rounds, validate_settings
    validate_settings(cfg)
    torch.set_num_threads(cfg.THREADS)
    if cfg.RUN_MODE in ('train', 'check'):
        data = load_training_data()
        windows, _, _ = build_ordered_windows(cfg, data.voltage_grid)
        splits = [make_training_split(data, seed) for seed in cfg.random_seeds]
        for split in splits:
            selected_rounds(cfg.ROUND_NUMBERS, round_numbers(split))
        if cfg.RUN_MODE == 'check':
            result = dict(dataset=cfg.DATASET_NAME, cells=len(data.qmax), segments=len(windows), raw_tail_points=cfg.RAW_TAIL_POINTS, device=cfg.device)
            print(json.dumps(result, indent=2))
            return result
    output = new_campaign(cfg)
    if cfg.RUN_MODE == 'train':
        for seed in cfg.random_seeds:
            run_one_seed(data, seed, output / f'{cfg.DATASET_NAME}_seed{seed}')
    else:
        from inference import infer, prepare
        if cfg.CHECKPOINT is not None:
            infer(output=output, checkpoint=cfg.CHECKPOINT)
        else:
            for seed in cfg.random_seeds:
                _, split, _ = prepare(cfg.DATASET_NAME, seed)
                for round_number in selected_rounds(cfg.ROUND_NUMBERS, round_numbers(split)):
                    infer(cfg.DATASET_NAME, seed, round_number, output)
    print(f'[OUTPUT] {output}', flush=True)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Edit config.py and run main.py without command-line arguments.')
    sys.modules['main'] = sys.modules[__name__]
    main()
