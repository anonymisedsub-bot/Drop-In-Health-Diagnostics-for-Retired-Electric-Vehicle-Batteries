"""Read and process only the two published discharge sheets of frozen cells.

This adapter reuses the unchanged numerical helpers from the source workflow.
It cannot repeat CClast-dependent cohort screening or reconstruct full IC.
The release mapping binds the inputs to the already selected model cohort.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

from rebatt_preprocessing.xlsx_xml import NS, read_sheet, worksheet_paths
from rebatt_preprocessing.models import DataContractError
from rebatt_preprocessing.batch_processing import (
    RejectedSample,
    _curve_for_sheet,
    _physical_remaining,
    _process_curve,
    _reject_internal_nan_in_discharge,
    grid_for_batch,
    physical_ic,
)

RAW_SHEETS = ('DisC', 'DisCC2')


def read_two_sheets(path):
    """Select by sheet name; reject missing or unexpected public worksheets."""
    source = Path(path)
    sheets = {}
    with ZipFile(source) as archive:
        paths = worksheet_paths(archive)
        missing = set(RAW_SHEETS) - set(paths)
        if missing:
            raise DataContractError(f'{source}: required sheet missing: {sorted(missing)}')
        if set(paths) != set(RAW_SHEETS):
            raise DataContractError(f'{source}: expected exactly DisC and DisCC2; unexpected sheets: {sorted(set(paths)-set(RAW_SHEETS))}')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(part.text or '' for part in item.iter(NS+'t'))
                      for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        for name in RAW_SHEETS:
            try:
                values, rows, headers, metadata = read_sheet(archive, paths[name], name, shared)
            except (ValueError, KeyError, IndexError, ET.ParseError) as exc:
                raise DataContractError(f'{source} sheet {name!r}: {exc}') from exc
            finite = np.isfinite(values).all(axis=1)
            if np.count_nonzero(finite) < 3:
                raise DataContractError(f'{source} sheet {name!r}: fewer than three finite I/V/Q rows')
            sheets[name] = {'data': values[finite], 'excel_rows': rows[finite],
                            'headers': tuple(headers), 'total_rows': len(values),
                            'skipped_rows': len(values)-int(finite.sum()), **metadata}
    return sheets


def process_two_sheets(sheets, batch):
    """Reconstruct DisC/RPT only, without fabricating unavailable CClast data."""
    if batch not in (4, 6):
        raise DataContractError('The two-sheet release contains source batches 4 and 6 only')
    grid = grid_for_batch(batch)
    parsed = {name: _curve_for_sheet(sheets, name)[0] for name in RAW_SHEETS}
    raw_initial = float(parsed['DisC'].voltage[0])
    if batch == 6:
        index = int(np.searchsorted(grid, raw_initial, side='right')) - 1
        if index < 0 or grid[index] < 3.22:
            raise RejectedSample('partial_legacy_initial_grid_voltage', {'raw_initial_voltage_V':raw_initial})
    elif raw_initial <= 3.27:
        raise RejectedSample('partial_raw_initial_voltage', {'initial_voltage_V':raw_initial})
    raw_minimum = float(np.min(parsed['DisC'].voltage))
    if raw_minimum > float(grid[0] + .01):
        raise RejectedSample('disc_raw_min_voltage', {'min_voltage_V':raw_minimum})
    processed = {}
    rpt_cumulative_at_cutoff = None
    for name, prefix in (('DisC', 'partial'), ('DisCC2', 'RPT')):
        result, support, info = _process_curve(parsed[name], batch, name, grid)
        q, v, _, _ = _physical_remaining(
            result, support, grid,
            maximum_voltage=raw_initial if prefix == 'partial' else None,
            upper_limit_voltage=info['raw_support_endpoint_V'],
        )
        if prefix == 'RPT':
            rpt_cumulative_at_cutoff = float(result.capacity[::-1][0])
        if prefix == 'partial' and batch != 6 and float(np.nanmax(v)) < 3.13:
            raise RejectedSample('partial_initial_voltage', {'max_voltage_V':float(np.nanmax(v))})
        processed[prefix+'_Q'], processed[prefix+'_V'] = q, v
        processed[prefix+'_IC'] = physical_ic(q)
    _reject_internal_nan_in_discharge(processed, grid)
    qmax = rpt_cumulative_at_cutoff if batch == 4 else float(np.max(parsed['DisCC2'].capacity))
    return {'processed': processed, 'Qmax': qmax}
