"""Domain-adaptation and transfer-learning trainers for Baseline group 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from train.train_ssl_knn import prepare_ssl_pl_training


SUPPORTED_METHODS = ("DA-AL", "DA-MMD", "DA-CORAL", "SSL-PL")


@dataclass
class DATrainResult:
    """A fitted predictor plus reproducibility metadata and auxiliary weights."""

    model: nn.Module
    metadata: dict[str, Any]
    auxiliary_states: dict[str, dict[str, torch.Tensor]]


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(features: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Return an identity view whose backward gradient is negated and scaled."""

    return _GradientReversal.apply(features, float(scale))


def _validate_feature_pair(source: torch.Tensor, target: torch.Tensor) -> None:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target features must both be two-dimensional")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")


def _squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cdist(left, right, p=2).square()


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Biased multi-kernel RBF maximum mean discrepancy."""

    _validate_feature_pair(source, target)
    if source.shape[0] == 0 or target.shape[0] == 0:
        return source.sum() * 0.0
    scales = tuple(float(value) for value in bandwidths)
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("bandwidths must contain positive values")

    dist_ss = _squared_distance(source, source)
    dist_tt = _squared_distance(target, target)
    dist_st = _squared_distance(source, target)
    kernel_ss = sum(torch.exp(-dist_ss / (2.0 * sigma**2)) for sigma in scales)
    kernel_tt = sum(torch.exp(-dist_tt / (2.0 * sigma**2)) for sigma in scales)
    kernel_st = sum(torch.exp(-dist_st / (2.0 * sigma**2)) for sigma in scales)
    discrepancy = kernel_ss.mean() + kernel_tt.mean() - 2.0 * kernel_st.mean()
    return discrepancy.clamp_min(0.0)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard squared CORAL distance between feature covariances."""

    _validate_feature_pair(source, target)
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.sum() * 0.0
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    covariance_source = source_centered.T @ source_centered / (source.shape[0] - 1)
    covariance_target = target_centered.T @ target_centered / (target.shape[0] - 1)
    feature_dim = source.shape[1]
    return (covariance_source - covariance_target).square().sum() / (
        4.0 * feature_dim**2
    )


class DomainDiscriminator(nn.Module):
    """Binary source/target classifier used by adversarial DA."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _device_from_config(config) -> torch.device:
    requested = str(getattr(config, "device", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _build_predictor(config, sequence_length: int, device: torch.device) -> MLP:
    model_config = dict(config.MLP_CONFIG)
    for key, expected in (("in_ch", 1), ("seq_len", sequence_length), ("output_dim", 1)):
        if model_config[key] != expected:
            raise ValueError(f"MLP_CONFIG {key} must equal {expected} for the configured input and target")
    if list(model_config["quantiles"]) != list(config.QUANTILES):
        raise ValueError("MLP_CONFIG quantiles must match QUANTILES")
    return MLP(config=model_config).to(device)


def _encode(model: MLP, inputs: torch.Tensor) -> torch.Tensor:
    return model.backbone(inputs.flatten(1))


def _regress(model: MLP, features: torch.Tensor) -> torch.Tensor:
    return torch.cat([head(features) for head in model.quantile_heads], dim=1)


def weighted_capacity_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: Sequence[float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Weighted per-head MSE; zero-weight heads receive zero gradient."""

    if predictions.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("predictions and targets must have shapes (N, Q) and (N, 1)")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("predictions and targets must have the same sample count")
    values = torch.as_tensor(
        tuple(float(value) for value in weights),
        dtype=predictions.dtype,
        device=predictions.device,
    )
    if values.ndim != 1 or values.numel() != predictions.shape[1]:
        raise ValueError("weights must contain one value per prediction head")
    if torch.any(values < 0) or float(values.sum().item()) <= 0.0:
        raise ValueError("weights must be non-negative and contain a positive value")
    loss_per_head = (predictions - targets).square().mean(dim=0)
    return torch.sum(loss_per_head * values) / values.sum()


def _median_index(config) -> int:
    quantiles = list(getattr(config, "QUANTILES", [0.1, 0.5, 0.9]))
    return quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2


def _median_prediction(model: MLP, inputs: torch.Tensor, config) -> torch.Tensor:
    outputs = _regress(model, _encode(model, inputs))
    index = _median_index(config)
    return outputs[:, index : index + 1]


def _to_tensor(array: np.ndarray, name: str) -> torch.Tensor:
    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return torch.from_numpy(values)


def _make_loader(
    *arrays: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*arrays),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def _validation_mse(
    model: MLP,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    config,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        prediction = _median_prediction(model, validation_x.to(device), config)
        return float(nn.functional.mse_loss(prediction, validation_y.to(device)).item())


def _is_improved(value: float, best: float | None, config) -> bool:
    if best is None:
        return True
    delta = float(getattr(config, "es_min_delta", 0.0))
    mode = str(getattr(config, "es_mode", "min"))
    return value < best - delta if mode == "min" else value > best + delta


def _make_scheduler(optimizer: torch.optim.Optimizer, config):
    scheduler_config = dict(getattr(config, "scheduler", {}))
    policy = scheduler_config.get("policy", "none")
    if policy == "plateau":
        return policy, torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
            min_lr=float(scheduler_config.get("min_lr", getattr(config, "lr_min", 0.0))),
        )
    if policy == "cosine_epoch":
        return policy, torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(config, "cosine_T_max", config.num_epochs)),
            eta_min=float(getattr(config, "lr_min", 0.0)),
        )
    if policy == "exp":
        return policy, torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(scheduler_config.get("gamma", 0.98)),
        )
    return "none", None


def _step_scheduler(policy: str, scheduler, validation_loss: float) -> None:
    if scheduler is None:
        return
    if policy == "plateau":
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def apply_step_decay_if_needed(
    optimizer: torch.optim.Optimizer,
    config,
    epoch_index: int,
) -> None:
    """Apply the configured extra epoch-step decay for the plateau policy."""

    scheduler_config = dict(getattr(config, "scheduler", {}))
    if scheduler_config.get("policy", "plateau") != "plateau":
        return
    step_every = int(scheduler_config.get("epoch_step", 0))
    if step_every <= 0 or (int(epoch_index) + 1) % step_every != 0:
        return
    gamma = float(scheduler_config.get("step_gamma", 1.0))
    minimum = float(scheduler_config.get("min_lr", 0.0))
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = max(
            float(parameter_group["lr"]) * gamma,
            minimum,
        )


def _validate_training_arrays(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> None:
    if source_x.ndim != 3 or target_x.ndim != 3 or validation_x.ndim != 3:
        raise ValueError("All input arrays must have shape (samples, channels, sequence)")
    if source_y.ndim != 2 or validation_y.ndim != 2:
        raise ValueError("Source and validation targets must have shape (samples, 1)")
    if source_x.shape[0] == 0 or validation_x.shape[0] == 0:
        raise ValueError("Source and validation sets cannot be empty")
    if source_x.shape[0] != source_y.shape[0]:
        raise ValueError("Source inputs and targets have different sample counts")
    if validation_x.shape[0] != validation_y.shape[0]:
        raise ValueError("Validation inputs and targets have different sample counts")
    expected_shape = source_x.shape[1:]
    if target_x.shape[1:] != expected_shape or validation_x.shape[1:] != expected_shape:
        raise ValueError("Source, target, and validation inputs must share a feature shape")


def _train_alignment_method(
    method: str,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    config,
    seed: int,
    device: torch.device,
) -> DATrainResult:
    model = _build_predictor(config, source_x.shape[-1], device)
    feature_dim = int(model.config["hidden_dims"][-1])
    has_target = target_x.shape[0] > 0
    discriminator = None
    parameters: list[nn.Parameter] = list(model.parameters())
    if method == "DA-AL" and has_target:
        discriminator = DomainDiscriminator(
            feature_dim,
            int(getattr(config, "DA_DISCRIMINATOR_HIDDEN_DIM", 64)),
        ).to(device)
        parameters.extend(discriminator.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scheduler_policy, scheduler = _make_scheduler(optimizer, config)
    source_loader = _make_loader(
        source_x,
        source_y,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    target_loader = (
        _make_loader(
            target_x,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed + 1,
        )
        if has_target
        else None
    )
    domain_loss = nn.BCEWithLogitsLoss()
    best_metric: float | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_discriminator_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improvement = 0
    alignment_steps = 0

    for epoch in range(1, int(config.num_epochs) + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        source_batches = cycle(source_loader)
        target_batches = cycle(target_loader) if target_loader is not None else None
        steps_this_epoch = (
            max(len(source_loader), len(target_loader))
            if target_loader is not None
            else len(source_loader)
        )
        for _ in range(steps_this_epoch):
            source_batch_x, source_batch_y = next(source_batches)
            source_batch_x = source_batch_x.to(device)
            source_batch_y = source_batch_y.to(device)
            source_features = _encode(model, source_batch_x)
            source_prediction = _regress(model, source_features)
            loss = weighted_capacity_loss(
                source_prediction,
                source_batch_y,
                getattr(config, "QUANTILE_LOSS_WEIGHTS", (0.0, 1.0, 0.0)),
            )

            if target_batches is not None:
                (target_batch_x,) = next(target_batches)
                target_features = _encode(model, target_batch_x.to(device))
                if method == "DA-AL":
                    source_logits = discriminator(gradient_reverse(source_features))
                    target_logits = discriminator(gradient_reverse(target_features))
                    domain_logits = torch.cat((source_logits, target_logits), dim=0)
                    domain_targets = torch.cat(
                        (torch.zeros_like(source_logits), torch.ones_like(target_logits)),
                        dim=0,
                    )
                    adversarial = domain_loss(domain_logits, domain_targets)
                    loss = loss + float(config.DA_ADV_WEIGHT) * adversarial
                elif method == "DA-MMD":
                    loss = loss + float(config.DA_MMD_WEIGHT) * mmd_loss(
                        source_features,
                        target_features,
                        getattr(config, "DA_MMD_BANDWIDTHS", (0.5, 1.0, 2.0)),
                    )
                elif method == "DA-CORAL":
                    loss = loss + float(config.DA_CORAL_WEIGHT) * coral_loss(
                        source_features, target_features
                    )
                alignment_steps += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_metric = _validation_mse(
            model, validation_x, validation_y, config, device
        )
        apply_step_decay_if_needed(optimizer, config, epoch - 1)
        _step_scheduler(scheduler_policy, scheduler, validation_metric)
        if _is_improved(validation_metric, best_metric, config):
            best_metric = validation_metric
            best_model_state = _clone_state_dict(model)
            best_discriminator_state = (
                _clone_state_dict(discriminator) if discriminator is not None else None
            )
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if bool(getattr(config, "es_enable", True)) and no_improvement >= int(
                getattr(config, "es_patience", 20)
            ):
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    auxiliary_states = {}
    if best_discriminator_state is not None:
        auxiliary_states["domain_discriminator"] = best_discriminator_state
    metadata = {
        "method": method,
        "source_count": int(source_x.shape[0]),
        "target_count": int(target_x.shape[0]),
        "alignment_steps": int(alignment_steps),
        "best_epoch": int(best_epoch),
        "best_validation_mse_normalized": float(best_metric),
    }
    return DATrainResult(model, metadata, auxiliary_states)


def _train_ssl_pl(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    config,
    seed: int,
    device: torch.device,
) -> DATrainResult:
    """Train the same MLP using source labels and KNN-labelled pool inputs."""

    knn_k = int(config.knn_k)
    knn_temperature = float(config.knn_temperature)
    training_x, training_y = prepare_ssl_pl_training(
        source_x,
        source_y,
        target_x,
        k=knn_k,
        temperature=knn_temperature,
    )
    model = _build_predictor(config, source_x.shape[-1], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scheduler_policy, scheduler = _make_scheduler(optimizer, config)
    loader = _make_loader(
        _to_tensor(training_x, "training_x"),
        _to_tensor(training_y, "training_y"),
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    best_metric: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, int(config.num_epochs) + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = weighted_capacity_loss(
                _regress(model, _encode(model, batch_x)),
                batch_y,
                getattr(config, "QUANTILE_LOSS_WEIGHTS", (0.0, 1.0, 0.0)),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_metric = _validation_mse(
            model, validation_x, validation_y, config, device
        )
        apply_step_decay_if_needed(optimizer, config, epoch - 1)
        _step_scheduler(scheduler_policy, scheduler, validation_metric)
        if _is_improved(validation_metric, best_metric, config):
            best_metric = validation_metric
            best_state = _clone_state_dict(model)
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if bool(getattr(config, "es_enable", True)) and no_improvement >= int(
                getattr(config, "es_patience", 20)
            ):
                break

    if best_state is None or best_metric is None:
        raise RuntimeError("SSL-PL training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "method": "SSL-PL",
        "source_count": int(len(source_x)),
        "target_count": int(len(target_x)),
        "pseudo_label_count": int(len(target_x)),
        "knn_k": knn_k,
        "knn_temperature": knn_temperature,
        "alignment_steps": 0,
        "best_epoch": int(best_epoch),
        "best_validation_mse_normalized": float(best_metric),
    }
    return DATrainResult(model, metadata, {})


def train_da_baseline(
    method: str,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config,
    seed: int,
) -> DATrainResult:
    """Fit one manuscript group-3 method without using target-domain labels."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported group-3 method: {method}")
    source_x = np.asarray(source_x, dtype=np.float32)
    source_y = np.asarray(source_y, dtype=np.float32)
    target_x = np.asarray(target_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    validation_y = np.asarray(validation_y, dtype=np.float32)
    _validate_training_arrays(
        source_x, source_y, target_x, validation_x, validation_y
    )

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _device_from_config(config)
    if method == "SSL-PL":
        return _train_ssl_pl(
            source_x,
            source_y,
            target_x,
            _to_tensor(validation_x, "validation_x"),
            _to_tensor(validation_y, "validation_y"),
            config,
            seed,
            device,
        )
    tensors = (
        _to_tensor(source_x, "source_x"),
        _to_tensor(source_y, "source_y"),
        _to_tensor(target_x, "target_x"),
        _to_tensor(validation_x, "validation_x"),
        _to_tensor(validation_y, "validation_y"),
    )
    return _train_alignment_method(method, *tensors, config, seed, device)


def predict_q50(model: MLP, inputs: np.ndarray, config) -> np.ndarray:
    """Return clipped normalized median predictions as a one-dimensional array."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("inputs must have shape (samples, channels, sequence)")
    device = _device_from_config(config)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = _median_prediction(
            model, torch.from_numpy(values).to(device), config
        )
    return np.clip(prediction.cpu().numpy().reshape(-1), 0.0, 1.0)
