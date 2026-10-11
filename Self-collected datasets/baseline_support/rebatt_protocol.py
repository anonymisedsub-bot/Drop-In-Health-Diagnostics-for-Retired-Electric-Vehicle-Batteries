"""Shared, leakage-safe protocol for the ReBATT model experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
EXPERIMENT_BATCHES = (1, 2, 3)
INITIAL_LABELLED_COUNT = 30
ACQUISITION_SIZE = 30
INPUT_PROTOCOL_VERSION = "real_partial_batchwise_v3"


@dataclass(frozen=True)
class ReBATTData:
    """Validated arrays and identities from the processed ReBATT MAT file."""

    q_partial: np.ndarray
    v_partial: np.ndarray
    ic_partial: np.ndarray
    q_full: np.ndarray
    v_full: np.ndarray
    ic_full: np.ndarray
    qmax: np.ndarray
    experiment_batch: np.ndarray
    source_batch: np.ndarray
    source_column: np.ndarray
    battery_id: np.ndarray
    filter_original_column: np.ndarray
    voltage_grid: np.ndarray
    mat_path: Path
    identity_provenance: str | None = None


@dataclass(frozen=True)
class BatchSplit:
    """One seeded, within-batch split and random Pool acquisition order."""

    batch: int
    seed: int
    initial_labelled_indices: np.ndarray
    pool_order: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    acquisition_size: int
    tail_screening_voltage: float | None = None
    excluded_indices: np.ndarray | None = None


@dataclass(frozen=True)
class BudgetRoles:
    """Role indices after one or more 30-sample acquisition rounds."""

    labelled_indices: np.ndarray
    unlabelled_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    labelled_fraction: float
    round_number: int


@dataclass(frozen=True)
class RoleView:
    """Role-specific raw inputs with capacity targets hidden when unlabelled."""

    role: str
    q: np.ndarray
    v: np.ndarray
    ic: np.ndarray
    target: np.ndarray | None
    global_indices: np.ndarray


@dataclass(frozen=True)
class VoltageWindows:
    """Tail indices and six ordered upper-fragment half-open index windows."""

    vseg_low: float
    vseg_upp: float
    tail_indices: np.ndarray
    upstream_windows: tuple[tuple[int, int], ...]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def nearest_pool_size(
    sample_count: int,
    pool_ratio: float = 0.6,
    pool_divisor: int = 30,
) -> int:
    """Return the nearest valid Pool size, resolving exact ties downward."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < pool_ratio < 1.0:
        raise ValueError("pool_ratio must lie strictly between 0 and 1")
    if pool_divisor <= 0:
        raise ValueError("pool_divisor must be positive")

    target = float(sample_count) * float(pool_ratio)
    lower = int(np.floor(target / pool_divisor)) * pool_divisor
    upper = int(np.ceil(target / pool_divisor)) * pool_divisor
    candidates = [size for size in {lower, upper} if 0 < size < sample_count]
    if not candidates:
        raise ValueError(
            "No positive Pool size below sample_count is divisible by pool_divisor"
        )
    return min(candidates, key=lambda size: (abs(size - target), size))


def complete_partial_tail_mask(data: ReBATTData, vseg_low: float) -> np.ndarray:
    """Identify cells with finite partial Q/V points and aligned IC through the tail."""

    tail_end = int(build_voltage_windows(data.voltage_grid, vseg_low).tail_indices[-1])
    return (
        np.isfinite(data.q_partial[:tail_end + 1, :]).all(axis=0)
        & np.isfinite(data.v_partial[:tail_end + 1, :]).all(axis=0)
        & np.isfinite(data.ic_partial[:tail_end, :]).all(axis=0)
    )


def make_batch_split(
    data: ReBATTData,
    batch: int,
    seed: int,
    pool_ratio: float = 0.6,
    pool_divisor: int = 30,
    initial_labelled_count: int = INITIAL_LABELLED_COUNT,
    acquisition_size: int = ACQUISITION_SIZE,
    vseg_low: float | None = None,
) -> BatchSplit:
    """Screen complete partial tails, then split one Batch into four roles."""

    batch = int(batch)
    if batch not in EXPERIMENT_BATCHES + (4, 5):
        raise ValueError(
            f"batch must be one of {EXPERIMENT_BATCHES + (4, 5)}, got {batch}"
        )
    if initial_labelled_count <= 0 or acquisition_size <= 0:
        raise ValueError("initial_labelled_count and acquisition_size must be positive")
    if pool_divisor % acquisition_size != 0:
        raise ValueError("pool_divisor must be divisible by acquisition_size")

    batch_mask = data.experiment_batch == batch
    if vseg_low is None:
        batch_indices = np.flatnonzero(batch_mask)
        excluded_indices = None
    else:
        tail_complete = complete_partial_tail_mask(data, vseg_low)
        batch_indices = np.flatnonzero(batch_mask & tail_complete)
        excluded_indices = _readonly(np.flatnonzero(batch_mask & ~tail_complete).astype(np.int64))
    remaining_count = len(batch_indices) - initial_labelled_count
    if remaining_count < 3:
        raise ValueError("Batch is too small for initial labels, Pool, validation and test")
    pool_size = nearest_pool_size(
        remaining_count,
        pool_ratio=pool_ratio,
        pool_divisor=pool_divisor,
    )
    held_out_count = remaining_count - pool_size
    validation_count = held_out_count // 2
    if validation_count == 0 or held_out_count - validation_count == 0:
        raise ValueError("Validation and test must each contain at least one sample")
    ordered = np.random.default_rng(int(seed)).permutation(batch_indices)
    initial_end = initial_labelled_count
    pool_end = initial_end + pool_size
    validation_end = pool_end + validation_count

    return BatchSplit(
        batch=batch,
        seed=int(seed),
        initial_labelled_indices=_readonly(ordered[:initial_end].astype(np.int64, copy=False)),
        pool_order=_readonly(ordered[initial_end:pool_end].astype(np.int64, copy=False)),
        validation_indices=_readonly(
            np.sort(ordered[pool_end:validation_end]).astype(np.int64, copy=False)
        ),
        test_indices=_readonly(
            np.sort(ordered[validation_end:]).astype(np.int64, copy=False)
        ),
        acquisition_size=int(acquisition_size),
        tail_screening_voltage=None if vseg_low is None else float(vseg_low),
        excluded_indices=excluded_indices,
    )


def round_numbers(split: BatchSplit) -> range:
    """Return all training rounds, including the initial 30-label round."""

    if len(split.pool_order) % split.acquisition_size:
        raise ValueError("Pool size must be divisible by acquisition_size")
    return range(1, len(split.pool_order) // split.acquisition_size + 2)


def make_round_roles(
    split: BatchSplit,
    round_number: int,
) -> BudgetRoles:
    """Expose the initial labels plus acquired Pool samples at one round."""

    if not isinstance(round_number, (int, np.integer)) or round_number not in round_numbers(split):
        raise ValueError(f"round_number must be in {list(round_numbers(split))}")
    acquired_count = (int(round_number) - 1) * split.acquisition_size
    labelled = np.concatenate((
        split.initial_labelled_indices,
        split.pool_order[:acquired_count],
    ))
    total_trainable = len(split.initial_labelled_indices) + len(split.pool_order)

    return BudgetRoles(
        labelled_indices=_readonly(labelled),
        unlabelled_indices=_readonly(split.pool_order[acquired_count:].copy()),
        validation_indices=_readonly(split.validation_indices.copy()),
        test_indices=_readonly(split.test_indices.copy()),
        labelled_fraction=len(labelled) / total_trainable,
        round_number=int(round_number),
    )


def role_view(
    data: ReBATTData,
    role: str,
    indices: np.ndarray,
    *,
    vseg_low: float | None = None,
    voltage_removal_window: float = 0.20,
) -> RoleView:
    """Use stored partial Q/V/IC for every role and trim its initial voltage.

    The maximum finite partial voltage is the cell's observed Vinit. Retain
    voltages through max(Vtail, Vinit - window), so the complete tail survives.
    Unlabelled samples never expose their capacity targets.
    """

    valid_roles = ("labelled", "unlabelled", "validation", "test")
    if role not in valid_roles:
        raise ValueError(f"role must be one of {valid_roles}, got {role!r}")
    global_indices = np.asarray(indices, dtype=np.int64)
    if global_indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if global_indices.size and (
        global_indices.min() < 0
        or global_indices.max() >= data.q_partial.shape[1]
    ):
        raise ValueError("indices contain an out-of-range global column")

    q = data.q_partial[:, global_indices]
    v = data.v_partial[:, global_indices]
    ic = data.ic_partial[:, global_indices]
    if vseg_low is None:
        raise ValueError("vseg_low is required for the voltage-removal window")
    if global_indices.size:
        tail_end = _aligned_grid_index(data.voltage_grid, float(vseg_low), "vseg_low")
        width = float(voltage_removal_window)
        if not np.isfinite(width) or width < 0:
            raise ValueError("voltage_removal_window must be finite and nonnegative")
        tail_observed = (
            np.isfinite(q[:tail_end + 1]).all(axis=0)
            & np.isfinite(v[:tail_end + 1]).all(axis=0)
            & np.isfinite(ic[:tail_end]).all(axis=0)
        )
        if not tail_observed.all():
            bad = int(global_indices[np.flatnonzero(~tail_observed)[0]])
            raise ValueError(f"The configured partial tail is incomplete at global column {bad}")
        # Vinit is the maximum finite voltage in the measured partial curve.
        initial_voltage = np.max(np.where(np.isfinite(v), v, -np.inf), axis=0)
        limit = np.maximum(float(vseg_low), initial_voltage - width)
        keep = np.isfinite(v) & (v <= limit[None, :] + 1e-10)
        q = np.where(keep, q, np.nan)
        v = np.where(keep, v, np.nan)
        # IC[i] belongs to the interval ending at voltage_grid[i + 1].
        ic = np.where(keep[:-1] & keep[1:], ic, np.nan)
    else:
        width = float(voltage_removal_window)
        if not np.isfinite(width) or width < 0:
            raise ValueError("voltage_removal_window must be finite and nonnegative")
    target = None if role == "unlabelled" else data.qmax[global_indices]

    return RoleView(
        role=role,
        q=_readonly(q),
        v=_readonly(v),
        ic=_readonly(ic),
        target=None if target is None else _readonly(target),
        global_indices=_readonly(global_indices.copy()),
    )


def _aligned_grid_index(
    voltage_grid: np.ndarray,
    voltage: float,
    name: str,
) -> int:
    matches = np.flatnonzero(
        np.isclose(voltage_grid, voltage, rtol=0.0, atol=1e-10)
    )
    if matches.size != 1:
        raise ValueError(f"{name}={voltage} is not aligned to voltage_grid")
    return int(matches[0])


def build_voltage_windows(
    voltage_grid: np.ndarray,
    vseg_low: float,
    upstream_span: float = 0.30,
    max_segments: int = 6,
) -> VoltageWindows:
    """Build the inclusive tail and six left-open/right-closed upper windows."""

    grid = np.asarray(voltage_grid, dtype=np.float64).squeeze()
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("voltage_grid must be a one-dimensional grid")
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if upstream_span <= 0:
        raise ValueError("upstream_span must be positive")

    idx_tail_hi = _aligned_grid_index(grid, float(vseg_low), "vseg_low")
    vseg_upp = float(vseg_low) + float(upstream_span)
    idx_upp_hi = _aligned_grid_index(grid, vseg_upp, "vseg_upp")
    upper_point_count = idx_upp_hi - idx_tail_hi
    if upper_point_count <= 0 or upper_point_count % max_segments != 0:
        raise ValueError(
            "The upper-region grid-point count must be positive and divisible "
            "by max_segments"
        )

    width = upper_point_count // max_segments
    first = idx_tail_hi + 1
    upper_windows = tuple(
        (first + segment * width, first + (segment + 1) * width)
        for segment in range(max_segments)
    )
    return VoltageWindows(
        vseg_low=float(vseg_low),
        vseg_upp=vseg_upp,
        tail_indices=_readonly(np.arange(idx_tail_hi + 1, dtype=np.int64)),
        upstream_windows=upper_windows,
    )


def complete_upstream_fragment_count(
    q_column: np.ndarray,
    v_column: np.ndarray,
    windows: VoltageWindows,
    ic_column: np.ndarray | None = None,
) -> int:
    """Count contiguous fragments observed in a real partial Q/IC curve."""

    q = np.asarray(q_column)
    v = np.asarray(v_column)
    if q.ndim != 1 or v.ndim != 1 or q.shape != v.shape:
        raise ValueError("q_column and v_column must be equal-length vectors")
    required_length = windows.upstream_windows[-1][1]
    if q.size < required_length:
        raise ValueError("Partial columns do not cover the configured upper region")

    q_mask = np.isfinite(q)
    v_mask = np.isfinite(v)
    if not np.array_equal(q_mask, v_mask):
        raise ValueError("Partial Q/V finite masks must match")
    if not q_mask[windows.tail_indices].all() or not v_mask[windows.tail_indices].all():
        raise ValueError("The configured voltage tail is not fully observed")
    if np.any((~q_mask[:-1]) & q_mask[1:]):
        raise ValueError("Partial finite values must form one contiguous prefix")

    ic_mask = None
    if ic_column is not None:
        ic = np.asarray(ic_column)
        if ic.ndim != 1 or ic.shape[0] != q.shape[0] - 1:
            raise ValueError("ic_column must have one fewer value than q_column")
        ic_mask = np.isfinite(ic)
        expected_ic_mask = q_mask[:-1] & q_mask[1:]
        if not np.array_equal(ic_mask, expected_ic_mask):
            raise ValueError("Partial Q/IC finite masks are not right-endpoint aligned")
        tail_end = int(windows.tail_indices[-1])
        if not ic_mask[:tail_end].all():
            raise ValueError("The configured IC tail is not fully observed")

    count = 0
    for start, end in windows.upstream_windows:
        qv_observed = q_mask[start:end].all() and v_mask[start:end].all()
        ic_observed = ic_mask is None or ic_mask[start - 1 : end - 1].all()
        if qv_observed and ic_observed:
            count += 1
        else:
            break
    return count


@lru_cache(maxsize=None)
def _file_sha256(path_text: str) -> str:
    """Hash a dataset once per resolved path during a process."""

    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_rows(data: ReBATTData, indices: np.ndarray) -> list[dict]:
    """Convert global columns into JSON-safe, auditable sample identities."""

    rows = []
    for index in np.asarray(indices, dtype=np.int64):
        global_index = int(index)
        rows.append(
            {
                "global_index": global_index,
                "experiment_batch": int(data.experiment_batch[global_index]),
                "source_batch": int(data.source_batch[global_index]),
                "source_matrix_column": int(data.source_column[global_index]),
                "battery_id": int(data.battery_id[global_index]),
                "filter_original_mat_column": int(
                    data.filter_original_column[global_index]
                ),
            }
        )
    return rows


def _write_json(path: str | Path, payload: dict) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def write_split_manifest(
    path: str | Path,
    data: ReBATTData,
    split: BatchSplit,
) -> dict:
    """Write one within-batch split and complete sample provenance."""

    mat_path = data.mat_path.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "dataset": {
            "path": str(mat_path),
            "sha256": _file_sha256(str(mat_path)),
        },
        "batch_mapping": {
            str(int(batch)): int(source)
            for batch, source in zip(np.unique(data.experiment_batch),
                                     np.unique(data.source_batch))
        },
        "dataset_batch": int(split.batch),
        "seed": int(split.seed),
        "initial_labelled_count": int(len(split.initial_labelled_indices)),
        "acquisition_size": int(split.acquisition_size),
        "pool_count": int(len(split.pool_order)),
        "validation_count": int(len(split.validation_indices)),
        "test_count": int(len(split.test_indices)),
        "initial_labelled": _identity_rows(data, split.initial_labelled_indices),
        "pool_order": _identity_rows(data, split.pool_order),
        "validation": _identity_rows(data, split.validation_indices),
        "test": _identity_rows(data, split.test_indices),
    }
    if data.identity_provenance is not None:
        payload["identity_provenance"] = data.identity_provenance
    if split.tail_screening_voltage is not None:
        payload["schema_version"] = 2
        excluded = np.asarray(split.excluded_indices, dtype=np.int64)
        eligible_count = (
            len(split.initial_labelled_indices) + len(split.pool_order)
            + len(split.validation_indices) + len(split.test_indices)
        )
        payload["tail_screening"] = {
            "rule": "finite_partial_Q_V_points_and_IC_intervals_through_inclusive_tail_endpoint",
            "tail_range": [float(data.voltage_grid[0]), float(split.tail_screening_voltage)],
            "source_count": int(np.count_nonzero(data.experiment_batch == split.batch)),
            "eligible_count": int(eligible_count),
            "excluded_count": int(len(excluded)),
            "excluded": _identity_rows(data, excluded),
        }
    return _write_json(path, payload)


def write_input_protocol_snapshot(
    path: str | Path,
    data: ReBATTData,
    test_indices: np.ndarray,
    vseg_low: float,
    voltage_removal_window: float = 0.20,
) -> dict:
    """Persist the experiment contract and each test cell's actual visibility."""

    view = role_view(data, "test", test_indices, vseg_low=vseg_low,
                     voltage_removal_window=voltage_removal_window)
    windows = build_voltage_windows(data.voltage_grid, vseg_low)
    records = _identity_rows(data, view.global_indices)
    for column, (record, index) in enumerate(zip(records, view.global_indices)):
        initial = float(np.nanmax(data.v_partial[:, index]))
        limit = max(float(vseg_low), initial - float(voltage_removal_window))
        retained = float(np.nanmax(view.v[:, column]))
        record.update(
            initial_voltage_source="partial_max_finite",
            initial_voltage=initial,
            retained_voltage_limit=limit,
            retained_max_voltage=retained,
            voltage_removal_applied=initial - limit,
            removed_grid_points=int(
                np.isfinite(data.v_partial[:, index]).sum()
                - np.isfinite(view.v[:, column]).sum()
            ),
            complete_fragments=complete_upstream_fragment_count(
                view.q[:, column], view.v[:, column], windows, view.ic[:, column]
            ),
        )
    return _write_json(path, {
        "schema_version": 1,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "dataset_sha256": _file_sha256(str(data.mat_path.resolve())),
        "q_definition": "remaining_discharge_capacity",
        "q_transform": "none",
        "ic_source": "stored_mat_direct",
        "ic_alignment": "Q[1:h+1]/IC[:h]; Q[a:b]/IC[a-1:b-1]",
        "role_inputs": {
            "labelled": "partial_with_voltage_removal",
            "unlabelled": "partial_with_voltage_removal",
            "validation": "partial_with_voltage_removal",
            "test": "partial_with_voltage_removal",
        },
        "capacity_target": "processed_Qmax_mat",
        "normalization_fit_role": "labelled",
        "vseg_low": float(vseg_low),
        "upstream_span": float(windows.vseg_upp - windows.vseg_low),
        "upstream_fragments": 6,
        "voltage_removal_window": float(voltage_removal_window),
        "test_inputs": records,
    })


def write_budget_manifest(
    path: str | Path,
    data: ReBATTData,
    roles: BudgetRoles,
) -> dict:
    """Write membership after one random-acquisition round."""

    payload = {
        "schema_version": 1,
        "round_number": int(roles.round_number),
        "labelled_fraction": float(roles.labelled_fraction),
        "labelled_count": int(len(roles.labelled_indices)),
        "unlabelled_count": int(len(roles.unlabelled_indices)),
        "labelled": _identity_rows(data, roles.labelled_indices),
        "unlabelled": _identity_rows(data, roles.unlabelled_indices),
    }
    return _write_json(path, payload)


def _json_safe(value):
    """Convert common configuration and NumPy values to plain JSON values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def write_config_snapshot(path: str | Path, config) -> dict:
    """Write all public JSON-compatible values from a config module/object."""

    payload = {}
    for name, value in vars(config).items():
        if name.startswith("_") or callable(value):
            continue
        try:
            payload[name] = _json_safe(value)
        except TypeError:
            continue
    return _write_json(path, payload)


def write_normalization_snapshot(path: str | Path, **normalizers) -> dict:
    """Write named normalization parameters without losing NumPy values."""

    payload = {
        str(name): _json_safe(normalizer)
        for name, normalizer in normalizers.items()
    }
    return _write_json(path, payload)
