"""Checkpoint inference with immutable settings and preserved MAT identity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
from runtime import ROOT, sha256, validate_output, FrozenConfig, config_values, save_config
from data import load_all_mats, build_ordered_windows, build_subset, inverse_cap
from main import _apply_norm_to_pack, apply_segment_availability_to_pack, BatterySet, collate_varlen
from model import ConditionalDecoder
from train_model import pack_segments_to_device

PACKAGED_DATA_SHA256 = '7c95928c620bf23c61456225fca8838a21951432b7ffe58ce75bd0d651b82d05'


def validate_indices(indices, count):
    values = np.asarray(indices, dtype=int)
    if (values.ndim != 1 or not len(values) or np.any(values < 0)
            or np.any(values >= count) or len(np.unique(values)) != len(values)):
        raise ValueError('Cell indices must be unique nonempty zero-based MAT columns within range')
    return values


def snapshot(output, seed, kind):
    metadata = dict(
        kind=kind, seed=seed, dataset_sha256=sha256(cfg.DATA_MAT_PATH),
        config=config_values(cfg), capacity_units='Ah',
        index_convention='zero-based processed MAT column', checkpoint='best.pt',
    )
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    save_config(output / 'config_snapshot.json', cfg)


def infer(checkpoint, output, indices=None, k=None):
    output = validate_output(output)
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / 'run_metadata.json').read_text(encoding='utf-8'))
    if sha256(cfg.DATA_MAT_PATH) != metadata['dataset_sha256']:
        raise ValueError('Dataset SHA-256 differs from checkpoint provenance')
    local = FrozenConfig(dict(metadata['config'], device=cfg.device, DATA_MAT_PATH=cfg.DATA_MAT_PATH))
    voltage, capacity = load_all_mats(local.DATA_MAT_PATH)
    split = json.loads((checkpoint.parent / 'splits.json').read_text(encoding='utf-8'))
    indices = validate_indices(split['test'] if indices is None else indices, capacity.shape[1])
    with np.load(checkpoint.parent / 'normalizers.npz') as saved:
        tail_norm = dict(mins=saved['tail_mins'], maxs=saved['tail_maxs'])
        capacity_norm = dict(min=float(saved['cap_min']), max=float(saved['cap_max']))
    windows, high, grid = build_ordered_windows(local)
    pack = build_subset(voltage, capacity, indices, local, windows, high)
    pack = _apply_norm_to_pack(pack, tail_norm, capacity_norm)
    if k is None:
        pack = apply_segment_availability_to_pack(
            pack, indices, np.load(checkpoint.parent / 'V_init.npy'), windows, grid, 'test',
        )
    else:
        if not 0 <= k <= local.MAX_SEGS:
            raise ValueError('k must be between zero and MAX_SEGS')
        pack['S_list'] = [segments[:k] for segments in pack['S_list']]
    network = ConditionalDecoder(local).to(local.device)
    network.load_state_dict(torch.load(checkpoint, map_location=local.device, weights_only=True))
    network.eval()
    loader = torch.utils.data.DataLoader(
        BatterySet(pack), batch_size=cfg.batch_size, collate_fn=collate_varlen, shuffle=False,
    )
    rows = []
    with torch.inference_mode():
        for inputs, fragments, features, _ in loader:
            segments, owners = pack_segments_to_device(fragments, local.device, local.SEG_LEN)
            _, predicted, _ = network(inputs.to(local.device), segments, owners, features.to(local.device))
            rows.append(predicted.cpu().numpy())
    predicted = inverse_cap(np.concatenate(rows), capacity_norm)
    quantiles = local.model['quantiles']
    result = pd.DataFrame({
        'cell_index_zero_based': indices, 'cell_column_one_based': indices + 1,
        'true_capacity_Ah': capacity[-1, indices],
        'pred_capacity_Ah': predicted[:, quantiles.index(0.5)],
        'available_segments': [len(segments) for segments in pack['S_list']],
    })
    for index, quantile in enumerate(quantiles):
        result[f'pred_q{100 * quantile:g}_Ah'] = predicted[:, index]
    mapping_path = ROOT / 'dataset/cell_index_mapping.csv'
    mapping_bound = metadata['dataset_sha256'] == PACKAGED_DATA_SHA256 and mapping_path.exists()
    if mapping_bound:
        mapping = pd.read_csv(mapping_path).set_index('cell_index_zero_based')
        result.insert(2, 'raw_mat_record_one_based', mapping.loc[indices, 'raw_mat_record_one_based'].to_numpy())
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'predictions.csv', index=False)
    report = dict(
        checkpoint_sha256=sha256(checkpoint), dataset_sha256=metadata['dataset_sha256'],
        checkpoint_kind=metadata['kind'], k_override=k, quantile_levels=list(quantiles),
        interval='model quantiles; no coverage guarantee',
        raw_record_mapping='packaged filtering audit' if mapping_bound else 'unavailable: no matching raw record mapping supplied',
    )
    (output / 'prediction_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result
