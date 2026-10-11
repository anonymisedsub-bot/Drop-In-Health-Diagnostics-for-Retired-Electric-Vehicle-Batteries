from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DataContractError(ValueError):
    """An immutable input does not satisfy the approved data contract."""


def _as_real_float64_array(values: object, label: str) -> np.ndarray:
    """Convert numeric input without ever discarding a complex component."""
    try:
        raw_values = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"{label} must be real numeric data") from exc
    has_object_complex = raw_values.dtype.kind == "O" and any(
        isinstance(value, (complex, np.complexfloating)) for value in raw_values.flat
    )
    if np.iscomplexobj(raw_values) or has_object_complex:
        raise DataContractError(f"{label} must be real numeric data")
    try:
        return np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(f"{label} must be convertible to float64") from exc


@dataclass(frozen=True)
class Curve:
    current: np.ndarray
    voltage: np.ndarray
    capacity: np.ndarray

    def __post_init__(self) -> None:
        converted: dict[str, np.ndarray] = {}
        for name in ("current", "voltage", "capacity"):
            values = _as_real_float64_array(getattr(self, name), f"curve {name}")
            if values.ndim != 1:
                raise DataContractError(f"curve {name} must be 1D")
            converted[name] = values

        lengths = {values.size for values in converted.values()}
        if len(lengths) != 1:
            raise DataContractError("curve channels must have equal length")
        if next(iter(lengths)) < 2:
            raise DataContractError("curve must contain at least 2 points")
        if any(not np.isfinite(values).all() for values in converted.values()):
            raise DataContractError("curve values must be finite")

        for name, values in converted.items():
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class SelectedCell:
    output_column: int
    batch: int
    battery_id: int
    source_matrix_column: int
    workbook: Path


@dataclass(frozen=True)
class InventoryRecord:
    batch: int
    battery_id: int
    path: Path
    size_bytes: int
    mtime_utc: str
    sha256: str | None


@dataclass(frozen=True)
class ProcessedCurve:
    """One capacity curve aligned to an ascending voltage grid."""

    voltage: np.ndarray
    capacity: np.ndarray

    def __post_init__(self) -> None:
        converted: dict[str, np.ndarray] = {}
        for name in ("voltage", "capacity"):
            values = _as_real_float64_array(getattr(self, name), f"processed {name}")
            if values.ndim != 1:
                raise DataContractError(f"processed {name} must be 1D")
            converted[name] = values

        if converted["voltage"].size != converted["capacity"].size:
            raise DataContractError("processed voltage and capacity must have equal length")
        if converted["voltage"].size < 2:
            raise DataContractError("processed curve must contain at least 2 points")
        if any(not np.isfinite(values).all() for values in converted.values()):
            raise DataContractError("processed curve values must be finite")
        if not np.all(np.diff(converted["voltage"]) > 0.0):
            raise DataContractError("processed voltage must be strictly ascending")

        for name, values in converted.items():
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ProtocolMetrics:
    """Observed CC/CClast protocol measurements used by hard release gates."""

    n_cc: int
    n_cclast: int
    cc_median_current: float
    cclast_median_current: float
    cc_min_voltage: float
    cc_max_voltage: float
    cclast_min_voltage: float
    cclast_max_voltage: float
    cc_capacity_span: float
    cclast_capacity_span: float
    cutoff_gap: float

    @classmethod
    def from_curves(cls, cc: Curve, cclast: Curve) -> "ProtocolMetrics":
        return cls(
            n_cc=int(cc.voltage.size),
            n_cclast=int(cclast.voltage.size),
            cc_median_current=float(np.median(cc.current)),
            cclast_median_current=float(np.median(cclast.current)),
            cc_min_voltage=float(np.min(cc.voltage)),
            cc_max_voltage=float(np.max(cc.voltage)),
            cclast_min_voltage=float(np.min(cclast.voltage)),
            cclast_max_voltage=float(np.max(cclast.voltage)),
            cc_capacity_span=float(np.ptp(cc.capacity)),
            cclast_capacity_span=float(np.ptp(cclast.capacity)),
            cutoff_gap=float(np.min(cclast.voltage) - 2.75),
        )
