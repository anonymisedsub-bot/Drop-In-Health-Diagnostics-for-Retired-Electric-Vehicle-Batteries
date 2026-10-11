"""CETL-compatible curve processing for the ReBATT 1C-cycle rebuild."""

from __future__ import annotations

import numbers

import numpy as np
from scipy.interpolate import PchipInterpolator

from .models import (
    Curve,
    DataContractError,
    ProcessedCurve,
    ProtocolMetrics,
    _as_real_float64_array,
)


_GRID_LOWER_V = 2.75
_GRID_UPPER_V = 4.20
_GRID_STEP_V = 0.005
_TAIL_MARGIN_V = 0.05
_MONOTONE_TOLERANCE_V = 1e-6
_MAX_MONOTONE_RUNS = 100


def _as_real_array(values: object, label: str) -> np.ndarray:
    return _as_real_float64_array(values, label)


def _finite_real_scalar(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_, np.ndarray)) or not isinstance(
        value, numbers.Real
    ):
        raise DataContractError(f"{label} must be one finite real scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"{label} must be one finite real scalar") from exc
    if not np.isfinite(converted):
        raise DataContractError(f"{label} must be one finite real scalar")
    return converted


def voltage_grid() -> np.ndarray:
    """Return the exact 291-row, inclusive 2.75:0.005:4.20 V grid."""
    return _GRID_LOWER_V + _GRID_STEP_V * np.arange(291, dtype=np.float64)


def _validated_grid(values: object) -> np.ndarray:
    grid = _as_real_array(values, "voltage grid")
    if grid.ndim != 1:
        raise DataContractError("voltage grid must be 1D")
    expected = voltage_grid()
    if grid.shape != expected.shape or not np.allclose(
        grid, expected, rtol=0.0, atol=1e-12
    ):
        raise DataContractError(
            "voltage grid must equal the ascending 291-point 2.75:0.005:4.20 grid"
        )
    return expected


def _finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _smooth_finite_vector(samples: np.ndarray, window: int) -> np.ndarray:
    count = samples.size
    if count < 2:
        return samples.copy()
    span = min(window, count)
    degree = min(2, span - 1)
    left_count = span // 2
    smoothed = np.empty(count, dtype=np.float64)
    for index in range(count):
        start = max(0, min(index - left_count, count - span))
        offsets = np.arange(start, start + span, dtype=np.float64) - index
        design = np.vander(offsets, N=degree + 1, increasing=True)
        coefficients = np.linalg.lstsq(
            design,
            samples[start : start + span],
            rcond=None,
        )[0]
        smoothed[index] = coefficients[0]
    return smoothed


def smooth_columns_matlab_sgolay(
    values: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Match MATLAB ``smoothdata(..., 'sgolay', window)`` per finite segment.

    MATLAB's even span is asymmetric around an interior sample.  For span 20,
    ten samples lie to the left and nine to the right.  Endpoints use the same
    full span with a shifted evaluation position.  NaNs delimit independent
    segments and remain NaN.
    """
    samples = _as_real_array(values, "SG input")
    if samples.ndim not in (1, 2) or 0 in samples.shape:
        raise DataContractError("SG input must be a nonempty 1D or 2D array")
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 1
    ):
        raise DataContractError("SG window must be a positive integer")
    if np.isinf(samples).any():
        raise DataContractError("SG input must not contain infinity")

    was_1d = samples.ndim == 1
    columns = samples[:, None] if was_1d else samples
    output = np.full(columns.shape, np.nan, dtype=np.float64)
    for column in range(columns.shape[1]):
        finite = np.isfinite(columns[:, column])
        for start, stop in _finite_segments(finite):
            output[start:stop, column] = _smooth_finite_vector(
                columns[start:stop, column], int(window)
            )
    return output[:, 0] if was_1d else output


def _stable_unique_sorted(
    voltage: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable unique by voltage, then stable ascending sort for PCHIP."""
    voltage_values = _as_real_array(voltage, "run voltage")
    capacity_values = _as_real_array(capacity, "run capacity")
    if voltage_values.ndim != 1 or capacity_values.ndim != 1:
        raise DataContractError("run voltage and capacity must be 1D")
    if voltage_values.size != capacity_values.size:
        raise DataContractError("run voltage and capacity lengths differ")
    if not np.isfinite(voltage_values).all() or not np.isfinite(capacity_values).all():
        raise DataContractError("run voltage and capacity must be finite")
    _, first_indices = np.unique(voltage_values, return_index=True)
    stable_indices = np.sort(first_indices)
    stable_voltage = voltage_values[stable_indices]
    stable_capacity = capacity_values[stable_indices]
    order = np.argsort(stable_voltage, kind="stable")
    return stable_voltage[order], stable_capacity[order]


def _split_into_monotone_runs(
    voltage: np.ndarray,
    tolerance_v: float = _MONOTONE_TOLERANCE_V,
) -> list[tuple[int, int]]:
    values = _as_real_array(voltage, "voltage")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise DataContractError("voltage must be one finite vector")
    if values.size < 3:
        return []

    delta = np.diff(values)
    direction = np.zeros(delta.shape, dtype=np.int8)
    direction[delta > tolerance_v] = 1
    direction[delta < -tolerance_v] = -1

    nonzero = np.flatnonzero(direction)
    if nonzero.size:
        direction[: nonzero[0]] = direction[nonzero[0]]
        for index in range(int(nonzero[0]) + 1, direction.size):
            if direction[index] == 0:
                direction[index] = direction[index - 1]

    change_indices = np.flatnonzero(np.diff(direction) != 0) + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices - 1, [values.size - 1]))
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start + 1 >= 3
    ]


def _fill_interpolation_gaps(grid: np.ndarray, interpolated: np.ndarray) -> None:
    finite_indices = np.flatnonzero(np.isfinite(interpolated))
    if finite_indices.size >= 2:
        first, second = finite_indices[:2]
        lower_slope = (interpolated[second] - interpolated[first]) / (
            grid[second] - grid[first]
        )
        below = np.arange(first)
        interpolated[below] = interpolated[first] + lower_slope * (
            grid[below] - grid[first]
        )

        penultimate, last = finite_indices[-2:]
        upper_slope = (interpolated[last] - interpolated[penultimate]) / (
            grid[last] - grid[penultimate]
        )
        above = np.arange(last + 1, grid.size)
        interpolated[above] = interpolated[last] + upper_slope * (
            grid[above] - grid[last]
        )

    missing_indices = np.flatnonzero(~np.isfinite(interpolated))
    finite_indices = np.flatnonzero(np.isfinite(interpolated))
    if missing_indices.size and not finite_indices.size:
        raise DataContractError("curve produced no interpolatable grid values")
    for index in missing_indices:
        nearest = finite_indices[np.argmin(np.abs(grid[finite_indices] - grid[index]))]
        interpolated[index] = interpolated[nearest]


def _interpolate_runs(
    voltage: np.ndarray,
    capacity: np.ndarray,
    grid: np.ndarray,
    runs: list[tuple[int, int]],
) -> np.ndarray:
    voltage_values = _as_real_array(voltage, "voltage")
    capacity_values = _as_real_array(capacity, "capacity")
    grid_values = _as_real_array(grid, "interpolation grid")
    if (
        voltage_values.ndim != 1
        or capacity_values.ndim != 1
        or grid_values.ndim != 1
        or voltage_values.size != capacity_values.size
    ):
        raise DataContractError("interpolation inputs have incompatible shapes")
    if (
        not np.isfinite(voltage_values).all()
        or not np.isfinite(capacity_values).all()
        or not np.isfinite(grid_values).all()
        or grid_values.size < 2
        or not np.all(np.diff(grid_values) > 0.0)
    ):
        raise DataContractError("interpolation inputs must be finite on an ascending grid")

    interpolated = np.full(grid_values.shape, np.nan, dtype=np.float64)
    for start, end in runs:
        if start < 0 or end >= voltage_values.size or end < start:
            raise DataContractError("monotone run bounds are invalid")
        run_voltage, run_capacity = _stable_unique_sorted(
            voltage_values[start : end + 1], capacity_values[start : end + 1]
        )
        if run_voltage.size < 2:
            continue
        covered = (grid_values >= run_voltage[0]) & (grid_values <= run_voltage[-1])
        if np.any(covered):
            interpolated[covered] = PchipInterpolator(
                run_voltage, run_capacity, extrapolate=False
            )(grid_values[covered])

    _fill_interpolation_gaps(grid_values, interpolated)
    if not np.isfinite(interpolated).all():
        raise DataContractError("curve interpolation did not produce finite values")
    return interpolated


def process_discharge(curve: Curve, grid: np.ndarray) -> ProcessedCurve:
    """Process a discharge with the original CETL final capacity row flip."""
    if not isinstance(curve, Curve):
        raise DataContractError("discharge curve must be a Curve")
    target = _validated_grid(grid)
    smoothed_voltage = smooth_columns_matlab_sgolay(curve.voltage, window=20)

    tail_indices = np.flatnonzero(smoothed_voltage < target[0] + _TAIL_MARGIN_V)
    if not tail_indices.size:
        raise DataContractError("discharge curve has no points below cutoff + 0.05 V")
    tail_minimum = int(tail_indices[np.argmin(smoothed_voltage[tail_indices])])
    if tail_minimum + 1 < 3:
        raise DataContractError("discharge curve leaves fewer than three points after trimming")

    reversed_voltage = smoothed_voltage[: tail_minimum + 1][::-1]
    reversed_capacity = curve.capacity[: tail_minimum + 1][::-1]
    runs = _split_into_monotone_runs(reversed_voltage)
    if not runs:
        raise DataContractError("discharge curve has no interpolatable monotone run")
    if len(runs) > _MAX_MONOTONE_RUNS:
        raise DataContractError("discharge curve contains more than 100 monotone runs")

    interpolated = _interpolate_runs(reversed_voltage, reversed_capacity, target, runs)
    processed_capacity = interpolated[::-1].copy()
    return ProcessedCurve(voltage=target.copy(), capacity=processed_capacity)


def process_charge(curve: Curve, grid: np.ndarray) -> ProcessedCurve:
    """Align an observed CC charge curve without reversing its capacity semantics."""
    if not isinstance(curve, Curve):
        raise DataContractError("charge curve must be a Curve")
    target = _validated_grid(grid)
    smoothed_voltage = smooth_columns_matlab_sgolay(curve.voltage, window=20)
    processed_capacity = _interpolate_runs(
        smoothed_voltage,
        curve.capacity,
        target,
        [(0, smoothed_voltage.size - 1)],
    )
    return ProcessedCurve(voltage=target.copy(), capacity=processed_capacity)


def apply_partial_mask(
    q: np.ndarray,
    v: np.ndarray,
    v_init: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the HANDOFF remaining-capacity transform and measured-region mask."""
    capacity = _as_real_array(q, "partial capacity")
    voltage = _as_real_array(v, "partial voltage")
    if capacity.ndim not in (1, 2) or capacity.shape != voltage.shape:
        raise DataContractError("partial Q and V must have the same 1D or 2D shape")
    if capacity.size == 0 or capacity.shape[0] < 2:
        raise DataContractError("partial Q and V must contain at least two rows")
    if not np.isfinite(capacity).all() or not np.isfinite(voltage).all():
        raise DataContractError("source partial Q and V must be finite")
    initial_voltage = _finite_real_scalar(v_init, "initial voltage")

    capacity_columns = capacity[:, None] if capacity.ndim == 1 else capacity
    voltage_columns = voltage[:, None] if voltage.ndim == 1 else voltage
    if not np.all(np.diff(voltage_columns, axis=0) > 0.0):
        raise DataContractError("partial voltage must be strictly ascending")

    cumulative = np.flip(capacity_columns, axis=0).copy()
    remaining = cumulative[0:1, :] - cumulative
    remaining[0, :] = 0.0
    masked_q = np.full(capacity_columns.shape, np.nan, dtype=np.float64)
    masked_v = np.full(voltage_columns.shape, np.nan, dtype=np.float64)
    for column in range(capacity_columns.shape[1]):
        valid = voltage_columns[:, column] <= initial_voltage
        indices = np.flatnonzero(valid)
        if indices.size < 2 or not np.array_equal(indices, np.arange(indices.size)):
            raise DataContractError(
                "initial voltage must leave a finite prefix of at least two points"
            )
        count = int(indices.size)
        masked_q[:count, column] = remaining[:count, column]
        masked_v[:count, column] = voltage_columns[:count, column]

    if capacity.ndim == 1:
        return masked_q[:, 0], masked_v[:, 0]
    return masked_q, masked_v


def sg20_incremental_capacity(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate dQ/dV, then apply SG20 within each finite IC prefix."""
    capacity = _as_real_array(q, "IC capacity")
    voltage = _as_real_array(v, "IC voltage")
    if capacity.ndim not in (1, 2) or capacity.shape != voltage.shape:
        raise DataContractError("IC Q and V must have the same 1D or 2D shape")
    if capacity.size == 0 or capacity.shape[0] < 2:
        raise DataContractError("IC Q and V must contain at least two rows")
    if np.isinf(capacity).any() or np.isinf(voltage).any():
        raise DataContractError("IC Q and V must not contain infinity")

    was_1d = capacity.ndim == 1
    capacity_columns = capacity[:, None] if was_1d else capacity
    voltage_columns = voltage[:, None] if was_1d else voltage
    result = np.full(
        (capacity_columns.shape[0] - 1, capacity_columns.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    for column in range(capacity_columns.shape[1]):
        q_finite = np.isfinite(capacity_columns[:, column])
        v_finite = np.isfinite(voltage_columns[:, column])
        if not np.array_equal(q_finite, v_finite):
            raise DataContractError(f"IC column {column + 1} Q/V finite masks differ")
        indices = np.flatnonzero(q_finite)
        if indices.size < 2:
            raise DataContractError(
                f"IC column {column + 1} must have at least two finite prefix rows"
            )
        if not np.array_equal(indices, np.arange(indices.size)):
            raise DataContractError(
                f"IC column {column + 1} must have one contiguous finite prefix"
            )
        count = int(indices.size)
        finite_v = voltage_columns[:count, column]
        if not np.all(np.diff(finite_v) > 0.0):
            raise DataContractError(f"IC column {column + 1} voltage must ascend")
        raw_ic = np.diff(capacity_columns[:count, column]) / np.diff(finite_v)
        result[: count - 1, column] = smooth_columns_matlab_sgolay(
            raw_ic, window=20
        )

    return result[:, 0] if was_1d else result


def validate_cycle_protocol(cc: Curve, cclast: Curve) -> ProtocolMetrics:
    """Validate only observed CC/CClast values and return their measured metrics."""
    if not isinstance(cc, Curve) or not isinstance(cclast, Curve):
        raise DataContractError("protocol inputs must both be Curve instances")
    metrics = ProtocolMetrics.from_curves(cc, cclast)
    if metrics.cc_median_current <= 0.0:
        raise DataContractError("CC median current must be positive")
    if metrics.cclast_median_current >= 0.0:
        raise DataContractError("CClast median current must be negative")
    if not 1.8 <= abs(metrics.cclast_median_current) <= 3.0:
        raise DataContractError("CClast absolute median current is outside the observed 1C range")
    if not 0.8 <= abs(metrics.cc_median_current) <= 1.6:
        raise DataContractError("CC absolute median current is outside the observed 0.5C range")
    if metrics.cc_max_voltage < 4.15:
        raise DataContractError("CC maximum voltage must reach at least 4.15 V")
    if metrics.cclast_max_voltage < 4.0:
        raise DataContractError("CClast maximum voltage must reach at least 4.0 V")
    if metrics.cclast_min_voltage > 2.80:
        raise DataContractError("CClast minimum voltage must reach 2.80 V or lower")
    if metrics.cc_capacity_span <= 0.0:
        raise DataContractError("CC capacity span must be positive")
    if metrics.cclast_capacity_span <= 0.0:
        raise DataContractError("CClast capacity span must be positive")
    return metrics
