"""Pure, batch-aware processing of the three observed discharge sheets."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import Curve, DataContractError, ProcessedCurve
from .processing import (
    _interpolate_runs,
    _split_into_monotone_runs,
    _stable_unique_sorted,
    smooth_columns_matlab_sgolay,
    voltage_grid,
)


class RejectedSample(DataContractError):
    """A known, auditable reason that a sample cannot be processed."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        self.reason = reason
        self.details = details
        super().__init__(f"{reason}: {details}")


def grid_for_batch(batch: int) -> np.ndarray:
    """Return the approved ascending voltage grid for one source batch."""
    if isinstance(batch, (bool, np.bool_)) or batch not in (1, 2, 3, 4, 5, 6):
        raise RejectedSample("invalid_batch", {"batch": batch})
    if batch in (4, 5):
        return 3.0 + 0.005 * np.arange(241, dtype=np.float64)
    return voltage_grid()


def _curve_for_sheet(sheets: dict, sheet: str) -> tuple[Curve, dict]:
    try:
        payload = sheets[sheet]
        raw = np.asarray(payload["data"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RejectedSample("unusable_discharge", {"sheet": sheet, "problem": "missing_or_invalid_data"}) from exc
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] < 3 or not np.isfinite(raw).all():
        raise RejectedSample("unusable_discharge", {"sheet": sheet, "problem": "fewer_than_three_finite_points"})
    try:
        curve = Curve(raw[:, 0], raw[:, 1], raw[:, 2])
    except DataContractError as exc:
        raise RejectedSample("unusable_discharge", {"sheet": sheet, "problem": str(exc)}) from exc
    if float(np.median(curve.current)) >= 0.0 or float(np.ptp(curve.capacity)) <= 0.0:
        raise RejectedSample("unusable_discharge", {"sheet": sheet, "problem": "current_or_capacity_span"})
    return curve, payload


def _process_raw_grid(curve: Curve, grid: np.ndarray, *, allow_short: bool = False) -> tuple[ProcessedCurve, np.ndarray, dict]:
    """Map raw monotone V/Q runs to the batch grid; IC smoothing happens later."""
    tails = np.flatnonzero(curve.voltage < grid[0] + 0.05)
    fallback = not bool(tails.size)
    if fallback and not allow_short:
        raise DataContractError("discharge curve has no points below cutoff + 0.05 V")
    end = int(curve.voltage.size - 1 if fallback else tails[np.argmin(curve.voltage[tails])])
    if end + 1 < 3:
        raise DataContractError("discharge curve leaves fewer than three points after trimming")
    reversed_v = curve.voltage[: end + 1][::-1]
    reversed_q = curve.capacity[: end + 1][::-1]
    runs = _split_into_monotone_runs(reversed_v)
    if not runs or len(runs) > 100:
        raise DataContractError("discharge needs 1 to 100 usable monotone runs")
    covered = np.zeros(grid.size, dtype=bool)
    endpoints = []
    for start, stop in runs:
        run_v, _ = _stable_unique_sorted(reversed_v[start : stop + 1], reversed_q[start : stop + 1])
        if run_v.size >= 2:
            covered |= (grid >= run_v[0]) & (grid <= run_v[-1])
            endpoints.append(float(run_v[-1]))
    interpolated = _interpolate_runs(reversed_v, reversed_q, grid, runs)
    return ProcessedCurve(grid.copy(), interpolated[::-1].copy()), covered, {
        "tail_fallback_used": fallback,
        "raw_support_endpoint_V": max(endpoints),
        "Q_rows_interpolated": int(covered.sum()),
        "Q_rows_filled_or_extrapolated": int((~covered).sum()),
    }


def _process_curve(curve: Curve, batch: int, sheet: str, grid: np.ndarray):
    try:
        # B4/5 historically required a cutoff-adjacent CClast trace; keep
        # that usability rule even though CClast is no longer the minimum gate.
        allow_short = sheet == "CClast" and batch not in (4, 5)
        return _process_raw_grid(curve, grid, allow_short=allow_short)
    except DataContractError as exc:
        raise RejectedSample("unusable_discharge", {"sheet": sheet, "problem": str(exc)}) from exc


def physical_ic(q: np.ndarray) -> np.ndarray:
    """Return physical-grid dQ/dV with SG20 smoothing inside finite segments."""
    try:
        capacity = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError("physical IC needs one finite-or-NaN vector") from exc
    if capacity.ndim != 1 or np.isinf(capacity).any() or np.count_nonzero(np.isfinite(capacity)) < 2:
        raise DataContractError("physical IC needs one finite-or-NaN vector with at least two finite rows")
    if not np.isfinite(capacity[0]):
        raise DataContractError("physical IC needs a finite cutoff capacity")
    return smooth_columns_matlab_sgolay(np.diff(capacity) / 0.005, window=20)


def _internal_nan_positions(values: np.ndarray, grid: np.ndarray, *, signal: str) -> dict:
    """Describe NaNs strictly inside a signal's first/last finite rows."""
    array = np.asarray(values)
    expected = len(grid) - 1 if signal == "IC" else len(grid)
    if array.ndim != 1 or array.size != expected:
        return {}
    finite_rows = np.flatnonzero(np.isfinite(array))
    if finite_rows.size < 2:
        return {}
    inside = np.arange(finite_rows[0], finite_rows[-1] + 1)
    nan_rows = inside[np.isnan(array[inside])]
    if not nan_rows.size:
        return {}
    result = {
        "signal": signal,
        "rows_1based": [int(row) + 1 for row in nan_rows],
    }
    if signal == "IC":
        result.update(
            lower_voltage_V=[float(grid[row]) for row in nan_rows],
            upper_voltage_V=[float(grid[row + 1]) for row in nan_rows],
        )
    else:
        result["voltage_V"] = [float(grid[row]) for row in nan_rows]
    return result


def _reject_internal_nan_in_discharge(processed: dict[str, np.ndarray], grid: np.ndarray) -> None:
    """Reject internal gaps in DisC/DisCC2 only; CClast gaps remain allowed."""
    phase_order = (("DisC", "partial"), ("DisCC2", "RPT"))
    affected_phases: list[str] = []
    positions: list[dict] = []
    for sheet, prefix in phase_order:
        phase_positions = []
        for signal in ("Q", "V", "IC"):
            position = _internal_nan_positions(
                processed[prefix + "_" + signal], grid, signal=signal
            )
            if position:
                phase_positions.append({
                    "phase": sheet,
                    "processed_prefix": prefix,
                    **position,
                })
        if phase_positions:
            affected_phases.append(sheet)
            positions.extend(phase_positions)
    if affected_phases:
        raise RejectedSample("internal_nan_in_discharge", {
            "affected_phases": affected_phases,
            "positions": positions,
        })


def _physical_remaining(
    result: ProcessedCurve,
    physical_support: np.ndarray,
    grid: np.ndarray,
    *,
    maximum_voltage: float | None = None,
    upper_limit_voltage: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Keep local support, low extension and at most one raw-end ceiling row."""
    cumulative = result.capacity[::-1].copy()
    support = np.asarray(physical_support, dtype=bool).copy()
    if support.shape != grid.shape or cumulative.shape != grid.shape:
        raise DataContractError("processed curve and support must match the physical grid")
    local = np.flatnonzero(support & np.isfinite(cumulative))
    if local.size < 2:
        raise DataContractError("curve has fewer than two locally supported finite grid rows")
    if not np.isfinite(cumulative[0]):
        raise DataContractError("curve has no finite capacity at the lower cutoff")

    first = int(local[0])
    keep = support | (np.arange(grid.size) < first)
    upper = np.zeros(grid.shape, dtype=bool)
    if upper_limit_voltage is not None:
        # Subtract tolerance so an endpoint numerically equal to a grid row
        # does not acquire a spurious next row.
        ceiling = int(np.searchsorted(grid, upper_limit_voltage - 1e-12, side="left"))
        if ceiling < grid.size and ceiling == int(local[-1]) + 1:
            if grid[ceiling] - upper_limit_voltage <= 0.005 + 1e-12:
                upper[ceiling] = True
                keep[ceiling] = True
    if maximum_voltage is not None:
        ceiling = min(int(np.searchsorted(grid, maximum_voltage - 1e-12, side="left")), grid.size - 1)
        keep &= np.arange(grid.size) <= ceiling
        upper &= keep
    keep &= np.isfinite(cumulative)
    if np.count_nonzero(keep) < 2 or not keep[0]:
        raise DataContractError("curve has fewer than two finite rows from the lower cutoff")

    remaining = np.full(grid.shape, np.nan, dtype=np.float64)
    remaining[keep] = cumulative[0] - cumulative[keep]
    remaining[0] = 0.0
    voltage = np.where(np.isfinite(remaining), grid, np.nan)
    exported_support = support & np.isfinite(remaining)
    finite_indices = np.flatnonzero(np.isfinite(remaining))
    if exported_support.sum() < 2:
        raise DataContractError("curve has fewer than two locally supported exported grid rows")
    first_finite, last_finite = int(finite_indices[0]), int(finite_indices[-1])
    counts = {
        "Q_rows_interpolated": int(exported_support.sum()),
        "Q_rows_filled_or_extrapolated": int(np.count_nonzero(keep & ~support)),
        "Q_rows_internal_nan": int(np.count_nonzero(~np.isfinite(remaining[first_finite:last_finite + 1]))),
        "Q_rows_upper_nan": int(np.count_nonzero(~np.isfinite(remaining[last_finite + 1:]))),
    }
    if upper_limit_voltage is not None:
        counts["Q_rows_upper_extrapolated"] = int(upper.sum())
    return remaining, voltage, exported_support, counts
