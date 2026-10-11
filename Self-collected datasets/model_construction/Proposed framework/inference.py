"""Rebuild checkpoint inputs without changing the editable training configuration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from common.rebatt_separate import load_separate_batch
from common.rebatt_protocol import make_batch_split, make_round_roles
from data import build_ordered_windows, build_subset, inverse_cap
from main import BatterySet, collate_varlen, _apply_norm_to_pack
from model import ConditionalDecoder
from runtime import ROOT, FrozenConfig, sha256, validate_output
from train_model import pack_segments_to_device


def historical(dataset, seed):
    if dataset not in ('NMC_A', 'NMC_B'):
        raise ValueError('Dataset must be NMC_A or NMC_B')
    candidates = list((Path(__file__).resolve().parent / 'historical' / dataset).glob(f'**/Proposed_random_seed{seed}'))
    if len(candidates) != 1:
        raise ValueError(f'No unique checkpoint campaign for {dataset}, seed {seed}')
    return candidates[0]


def prepare(dataset, seed=0, campaign=None):
    """Load a separate immutable snapshot namespace for one checkpoint campaign."""
    if dataset not in ('NMC_A', 'NMC_B'):
        raise ValueError('Dataset must be NMC_A or NMC_B')
    campaign = Path(campaign) if campaign is not None else historical(dataset, seed)
    saved = json.loads((campaign / 'config_snapshot.json').read_text(encoding='utf-8'))
    manifest = json.loads((campaign / 'split_manifest.json').read_text(encoding='utf-8'))
    if int(manifest['seed']) != seed:
        raise ValueError('Requested seed differs from checkpoint split manifest')
    model_batch = int(manifest['dataset_batch'])
    source_batch = int(manifest['batch_mapping'][str(model_batch)])
    # Bundled snapshots retain their original machine paths. New snapshots retain
    # the configured data path; each dataset's checkpoint geometry remains local.
    bundled = campaign.resolve().is_relative_to((Path(__file__).resolve().parent / 'historical').resolve())
    path = ROOT / 'dataset' / f'{dataset}_data_processed.mat' if bundled else Path(saved['DATA_MAT_PATH'])
    if sha256(path) != manifest['dataset']['sha256']:
        raise ValueError('Dataset SHA-256 differs from checkpoint split manifest')
    saved.update(PROJECT_ROOT=ROOT, DATASET_NAME=dataset, DATA_MAT_PATH=path, device=cfg.device)
    local = FrozenConfig(saved)
    data = load_separate_batch(path, source_batch)
    split = make_batch_split(
        data, batch=model_batch, seed=seed,
        pool_ratio=local.POOL_TARGET_RATIO, pool_divisor=local.POOL_DIVISOR,
        initial_labelled_count=local.INITIAL_LABELLED_COUNT,
        acquisition_size=local.ACQUISITION_SIZE, vseg_low=local.VSEG_LOW,
    )
    return data, split, local


def infer(dataset=None, seed=0, round_number=1, output=None, checkpoint=None):
    """Infer one saved test split and compare against that run's saved predictions."""
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        run = checkpoint.parent
        campaign = run.parent
        snapshot = json.loads((campaign / 'config_snapshot.json').read_text(encoding='utf-8'))
        manifest = json.loads((campaign / 'split_manifest.json').read_text(encoding='utf-8'))
        model_batch = str(manifest['dataset_batch'])
        source_batch = int(manifest['batch_mapping'][model_batch])
        dataset = snapshot.get('DATASET_NAME') or {4: 'NMC_A', 6: 'NMC_B'}[source_batch]
        seed = int(manifest['seed'])
        round_number = int(run.name.removeprefix('iter_'))
    else:
        dataset = dataset or cfg.DATASET_NAME
        campaign = historical(dataset, seed)
        run = campaign / f'iter_{round_number}'
        checkpoint = run / 'best.pt'
    data, split, local = prepare(dataset, seed, campaign)
    roles = make_round_roles(split, round_number)
    budget = json.loads((run / 'budget_manifest.json').read_text(encoding='utf-8'))
    if int(budget['round_number']) != round_number:
        raise ValueError('Checkpoint directory differs from saved budget round')
    if [item['global_index'] for item in budget['labelled']] != roles.labelled_indices.tolist():
        raise ValueError('Checkpoint labelled membership differs from saved split')
    windows, high, _ = build_ordered_windows(local, data.voltage_grid)
    pack = build_subset(data, roles.test_indices, local, windows, high, role='test')
    normalizers = json.loads((run / 'normalization.json').read_text(encoding='utf-8'))
    tail = normalizers['tail']
    tail['mins'] = np.asarray(tail['mins'], dtype=np.float32)
    tail['maxs'] = np.asarray(tail['maxs'], dtype=np.float32)
    capacity_norm = normalizers['capacity']
    pack = _apply_norm_to_pack(pack, tail, capacity_norm)
    network = ConditionalDecoder(local).to(cfg.device)
    network.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True))
    network.eval()
    loader = DataLoader(BatterySet(pack), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_varlen)
    quantile_predictions = []
    with torch.inference_mode():
        for inputs, fragments, metadata, _ in loader:
            segments, owners = pack_segments_to_device(fragments, cfg.device, local.SEG_LEN)
            _, predictions, _ = network(inputs.to(cfg.device), segments, owners, metadata.to(cfg.device))
            quantile_predictions.append(predictions.cpu().numpy())
    quantiles = inverse_cap(np.concatenate(quantile_predictions), capacity_norm)
    actual = quantiles[:, local.model['quantiles'].index(0.5)]
    saved = pd.read_excel(run / 'summary_performance.xlsx')
    if not np.array_equal(saved['global_index'].to_numpy(), roles.test_indices):
        raise AssertionError('Saved prediction identity mismatch')
    error = float(np.max(np.abs(actual - saved.y_pred.to_numpy())))
    digest = sha256(data.mat_path)
    table = pd.DataFrame({
        'dataset': dataset, 'source_batch': data.source_batch[roles.test_indices],
        'mat_sha256': digest, 'mat_column_1based': data.source_column[roles.test_indices],
        'global_index_0based': roles.test_indices, 'capacity_true_Ah': data.qmax[roles.test_indices],
        'capacity_predicted_Ah': actual,
    })
    for index, quantile in enumerate(local.model['quantiles']):
        table[f'capacity_q{quantile:g}_Ah'] = quantiles[:, index]
    result = dict(dataset=dataset, seed=seed, round=round_number,
                  labelled_count=len(roles.labelled_indices), test_count=len(table),
                  max_absolute_error_Ah=error, reference='checkpoint saved test predictions',
                  checkpoint_sha256=sha256(checkpoint), mat_sha256=digest)
    if output is not None:
        output = validate_output(output)
        output.mkdir(parents=True, exist_ok=True)
        stem = f'{dataset}_seed{seed}_round{round_number}'
        table.to_csv(output / f'{stem}_predictions.csv', index=False)
        (output / f'{stem}_inference_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if error > 2e-4:
        raise AssertionError(f'Checkpoint prediction mismatch: {error} Ah')
    return result
