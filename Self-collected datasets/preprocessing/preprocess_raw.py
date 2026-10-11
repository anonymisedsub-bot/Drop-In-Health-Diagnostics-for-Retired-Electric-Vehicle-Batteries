"""Rebuild the eight available fields of the frozen cohort from two raw sheets.

python preprocess_raw.py --dataset-root ../dataset --output-dir ./raw_rebuild
The output directory must not exist. The released nine-field MATs never change.
CClast-dependent processed_IC_full_mat cannot be reconstructed from this release.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.io import loadmat, savemat

DATASETS = {'NMC_A': (4, 353, 'NMC_A_data_processed.mat'),
            'NMC_B': (6, 334, 'NMC_B_data_processed.mat')}
NOT_RECONSTRUCTED = ['processed_IC_full_mat']
RAW_SHEETS = ['DisC', 'DisCC2']
AVAILABLE_FIELDS = (
    'processed_Qd_partial_mat', 'processed_Vd_partial_mat', 'processed_IC_partial_mat',
    'processed_RPT_Q_full_mat', 'processed_RPT_V_full_mat', 'processed_RPT_IC_full_mat',
    'processed_Qmax_mat', 'voltage_grid_mat',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected_hash, expected_bytes):
    path = Path(path)
    if path.stat().st_size != int(expected_bytes) or sha256_file(path) != expected_hash:
        raise ValueError(f'Workbook integrity check failed: {path}')


def validate_mapping(rows, expected_counts=None):
    expected_counts = expected_counts or {name: data[1] for name, data in DATASETS.items()}
    if set(r['dataset'] for r in rows) != set(expected_counts):
        raise ValueError('Mapping datasets differ from the required release cohort')
    for dataset, count in expected_counts.items():
        selected = [r for r in rows if r['dataset'] == dataset]
        ids = [int(r['new_id']) for r in selected]
        columns = [int(r['mat_column_1based']) for r in selected]
        if ids != list(range(1, count + 1)) or columns != ids:
            raise ValueError(f'Noncontiguous, duplicated or unordered identity in {dataset}')
        originals = [int(r['original_battery_id']) for r in selected]
        if len(set(originals)) != count:
            raise ValueError(f'Duplicate original battery identity in {dataset}')
        for row in selected:
            expected = f"raw/{dataset}/{row['new_id']}.xlsx"
            if row['raw_relative_path'] != expected:
                raise ValueError(f'Invalid raw workbook path: {row["raw_relative_path"]}')
            if int(row['original_source_batch']) != DATASETS[dataset][0]:
                raise ValueError(f'Incorrect source batch for {dataset}')


def reconstruct_one(task):
    path, batch = task
    from two_sheet_processing import read_two_sheets, process_two_sheets
    from rebatt_preprocessing.batch_processing import grid_for_batch
    result = process_two_sheets(read_two_sheets(Path(path)), batch)
    values = result['processed']
    return {
        'processed_Qd_partial_mat': values['partial_Q'],
        'processed_Vd_partial_mat': values['partial_V'],
        'processed_IC_partial_mat': values['partial_IC'],
        'processed_RPT_Q_full_mat': values['RPT_Q'],
        'processed_RPT_V_full_mat': values['RPT_V'],
        'processed_RPT_IC_full_mat': values['RPT_IC'],
        'processed_Qmax_mat': np.array([result['Qmax']], dtype=np.float64),
        'voltage_grid_mat': grid_for_batch(batch),
    }


def compare_payload(actual, expected):
    variables = {}
    if set(actual) != set(expected):
        raise ValueError('MAT variable names do not match')
    for name in sorted(expected):
        left, right = np.asarray(actual[name]), np.asarray(expected[name])
        shape_exact = left.shape == right.shape
        mask_exact = shape_exact and np.array_equal(np.isnan(left), np.isnan(right))
        exact = shape_exact and np.array_equal(left, right, equal_nan=True)
        delta = None
        if shape_exact:
            finite = np.isfinite(left) & np.isfinite(right)
            delta = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.
        variables[name] = {'shape': list(left.shape), 'shape_exact': shape_exact,
                           'nan_mask_exact': mask_exact, 'exact': exact,
                           'maximum_absolute_difference': delta}
    return {'all_exact': all(v['exact'] for v in variables.values()), 'variables': variables}


def rebuild(dataset_root, output_dir, reference_dir=None, workers=4):
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    reference_dir = Path(reference_dir).resolve() if reference_dir else dataset_root
    if output_dir.exists():
        raise ValueError('Output directory must not exist; choose a fresh directory')
    mapping = dataset_root / 'raw' / 'source_mapping.csv'
    with mapping.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    validate_mapping(rows)
    for row in rows:
        verify_file(dataset_root / row['raw_relative_path'], row['sha256'], row['byte_size'])
    output_dir.mkdir(parents=True)
    report = {'schema': 'nmc-two-sheet-reconstruction-v2',
              'scope': 'Eight available fields from the frozen selected cohort defined by raw/source_mapping.csv; historical candidate screening is not repeated.',
              'raw_sheets': RAW_SHEETS, 'not_reconstructed': NOT_RECONSTRUCTED,
              'not_reconstructed_reason': 'processed_IC_full_mat requires the unpublished CClast sheet; it remains only in the unchanged reference MAT.',
              'available_field_count': len(AVAILABLE_FIELDS),
              'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'mapping_sha256': sha256_file(mapping), 'workbook_hashes_verified': len(rows), 'datasets': {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dataset, (batch, count, filename) in DATASETS.items():
            cohort = [r for r in rows if r['dataset'] == dataset]
            reference = reference_dir / filename
            expected_hash = {r['processed_mat_sha256'] for r in cohort}
            if expected_hash != {sha256_file(reference)}:
                raise ValueError(f'Reference MAT digest mismatch: {reference}')
            rebuilt = []
            tasks = [(str(dataset_root / r['raw_relative_path']), batch) for r in cohort]
            for i, result in enumerate(pool.map(reconstruct_one, tasks, chunksize=1), 1):
                rebuilt.append(result)
                if i % 25 == 0 or i == count:
                    print(f'{dataset}: reconstructed {i}/{count}', flush=True)
            payload = {name: np.column_stack([r[name] for r in rebuilt]) for name in rebuilt[0]
                       if name != 'voltage_grid_mat'}
            payload['voltage_grid_mat'] = rebuilt[0]['voltage_grid_mat'][:, None]
            reference_payload = {k: v for k, v in loadmat(reference).items() if not k.startswith('__')}
            if set(reference_payload) != set(AVAILABLE_FIELDS) | set(NOT_RECONSTRUCTED):
                raise ValueError(f'Reference MAT schema differs from the canonical nine-field release: {reference}')
            expected = {k: reference_payload[k] for k in AVAILABLE_FIELDS}
            comparison = compare_payload(payload, expected)
            comparison['all_available_fields_exact'] = comparison.pop('all_exact')
            output_filename = filename.replace('_data_processed.mat', '_from_raw.mat')
            comparison.update(sample_count=count, reference_mat_sha256=sha256_file(reference),
                              reconstructed_fields=list(AVAILABLE_FIELDS), not_reconstructed=NOT_RECONSTRUCTED,
                              raw_sheets=RAW_SHEETS, output_mat_filename=output_filename)
            report['datasets'][dataset] = comparison
            savemat(output_dir / output_filename, payload, do_compression=True)
            print(f'{dataset}: all 8 available variables exact = {comparison["all_available_fields_exact"]}; processed_IC_full_mat not reconstructed', flush=True)
    report['all_available_fields_exact'] = all(d['all_available_fields_exact'] for d in report['datasets'].values())
    (output_dir / 'reconstruction_comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not report['all_available_fields_exact']:
        raise ValueError('Available raw fields differ from frozen MAT; inspect reconstruction_comparison.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, help='Defaults to dataset root')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    rebuild(args.dataset_root, args.output_dir, args.reference_dir, args.workers)


if __name__ == '__main__':
    main()
