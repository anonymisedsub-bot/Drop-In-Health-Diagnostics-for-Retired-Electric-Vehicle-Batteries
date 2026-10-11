"""Read the current nine-variable per-batch MAT files for model experiments.

The models use measured partial Q/V/IC curves and Qmax. The missing full Q/V
arrays are represented as NaN placeholders, never as fabricated measurements.
Source IDs are unavailable; output identity fields are file-local column numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from baseline_support.rebatt_protocol import ReBATTData


SOURCE_BATCHES = (1, 2, 4, 5, 6)
REQUIRED_KEYS = (
    "processed_Qd_partial_mat", "processed_Vd_partial_mat",
    "processed_IC_partial_mat", "processed_IC_full_mat",
    "processed_RPT_Q_full_mat", "processed_RPT_V_full_mat",
    "processed_RPT_IC_full_mat", "processed_Qmax_mat", "voltage_grid_mat",
)
IDENTITY_PROVENANCE = (
    "The MAT has no original battery/source ID arrays. Output identity fields "
    "are 1-based column positions in this exact MAT file, not original battery "
    "IDs or columns in another file. Source Batch is assigned from the "
    "explicitly selected batch file."
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def load_separate_batch(path: str | Path, source_batch: int) -> ReBATTData:
    if source_batch not in SOURCE_BATCHES:
        raise ValueError(f"Unsupported source Batch: {source_batch}")
    mat_path = Path(path).expanduser().resolve()
    raw = loadmat(mat_path)
    missing = set(REQUIRED_KEYS).difference(raw)
    if missing:
        raise ValueError(f"Batch {source_batch} MAT lacks keys: {sorted(missing)}")

    v_min = 3.0 if source_batch in (4, 5) else 2.75
    grid_count = int(round((4.2 - v_min) / 0.005)) + 1
    grid_raw = np.asarray(raw["voltage_grid_mat"], dtype=np.float64)
    if grid_raw.shape != (grid_count, 1):
        raise ValueError(f"Batch {source_batch} has wrong voltage grid shape")
    grid = grid_raw[:, 0]
    if (not np.isfinite(grid).all()
            or not np.allclose(grid, v_min + np.arange(grid_count) * 0.005,
                               rtol=0, atol=1e-10)):
        raise ValueError(f"Batch {source_batch} voltage grid disagrees with expected 5 mV grid")

    qmax = np.asarray(raw["processed_Qmax_mat"], dtype=np.float64).reshape(-1)
    count = len(qmax)
    if count == 0 or not np.isfinite(qmax).all() or np.any(qmax <= 0):
        raise ValueError(f"Batch {source_batch} has invalid Qmax targets")
    matrices = {}
    for name in ("processed_Qd_partial_mat", "processed_Vd_partial_mat",
                 "processed_RPT_Q_full_mat", "processed_RPT_V_full_mat"):
        value = np.asarray(raw[name], dtype=np.float64)
        if value.shape != (grid_count, count) or np.isinf(value).any():
            raise ValueError(f"Invalid {name} shape or infinite values")
        matrices[name] = value
    for name in ("processed_IC_partial_mat", "processed_IC_full_mat",
                 "processed_RPT_IC_full_mat"):
        value = np.asarray(raw[name], dtype=np.float64)
        if value.shape != (grid_count - 1, count) or np.isinf(value).any():
            raise ValueError(f"Invalid {name} shape or infinite values")
        matrices[name] = value

    for phase, q_key, v_key, ic_key in (
        ("partial", "processed_Qd_partial_mat", "processed_Vd_partial_mat",
         "processed_IC_partial_mat"),
        ("RPT", "processed_RPT_Q_full_mat", "processed_RPT_V_full_mat",
         "processed_RPT_IC_full_mat"),
    ):
        q, v, ic = matrices[q_key], matrices[v_key], matrices[ic_key]
        finite = np.isfinite(q)
        if (not np.array_equal(finite, np.isfinite(v))
                or not np.array_equal(np.isfinite(ic), finite[:-1] & finite[1:])
                or not finite[0].all()
                or not np.allclose(v[finite], np.broadcast_to(grid[:, None], v.shape)[finite],
                                   rtol=0, atol=1e-10)):
            raise ValueError(f"Invalid Batch {source_batch} {phase} Q/V/IC masks or voltages")
    if np.isfinite(matrices["processed_IC_full_mat"]).sum() == 0:
        raise ValueError("Full-discharge IC contains no finite values")

    q_partial = matrices["processed_Qd_partial_mat"].copy()
    v_partial = matrices["processed_Vd_partial_mat"].copy()
    ic_partial = matrices["processed_IC_partial_mat"].copy()
    for column in range(count):
        missing_q = np.flatnonzero(~np.isfinite(q_partial[:, column]))
        missing_ic = np.flatnonzero(~np.isfinite(ic_partial[:, column]))
        first_gap = min(
            int(missing_q[0]) if len(missing_q) else grid_count,
            int(missing_ic[0]) + 1 if len(missing_ic) else grid_count,
        )
        q_partial[first_gap:, column] = np.nan
        v_partial[first_gap:, column] = np.nan
        ic_partial[max(first_gap - 1, 0):, column] = np.nan

    column_position = np.arange(1, count + 1, dtype=np.int64)
    model_batch = 3 if source_batch == 6 else source_batch
    return ReBATTData(
        q_partial=_readonly(q_partial), v_partial=_readonly(v_partial),
        ic_partial=_readonly(ic_partial),
        q_full=_readonly(np.full((grid_count, count), np.nan)),
        v_full=_readonly(np.full((grid_count, count), np.nan)),
        ic_full=_readonly(matrices["processed_IC_full_mat"]),
        qmax=_readonly(qmax),
        experiment_batch=_readonly(np.full(count, model_batch, dtype=np.int64)),
        source_batch=_readonly(np.full(count, source_batch, dtype=np.int64)),
        source_column=_readonly(column_position),
        battery_id=_readonly(column_position.copy()),
        filter_original_column=_readonly(column_position.copy()),
        voltage_grid=_readonly(grid), mat_path=mat_path,
        identity_provenance=IDENTITY_PROVENANCE,
    )
