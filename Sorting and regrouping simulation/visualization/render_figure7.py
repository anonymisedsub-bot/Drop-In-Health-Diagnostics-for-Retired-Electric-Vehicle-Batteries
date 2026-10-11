"""Render manuscript Figure 7 a-l from the packaged frozen simulation tables."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

import figure_style as fs
import build_main_figure as old
import build_manuscript_png as mp
import build_merged_main_png as merged
import build_workflow_panel as workflow
import plot_mechanism as mechanism

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0,str(MODULE))
from portable_io import path_reference, validate_output
PANELS = dict(a='Simulation workflow', **{k: v[3] for k, v in merged.HEADINGS.items()})
PANEL_SOURCES = {
    'a': [],
    'b': ['main_b_example_300_records.csv'],
    'c': ['fig01_flow_counts.csv'],
    'd': ['main2_assembled_cells.csv'],
    'e': ['main2_pack_metrics.csv'],
    'f': ['main_f_sorting_quality.csv', 'main_f_residual_risk_records.csv'],
    'g': ['main_c_quality_yield.csv'],
    'h': ['merged_h_distribution_summary.csv'],
    'i': ['merged_i_distribution_summary.csv'],
    'j': ['main_ghi_all_settings.csv'],
    'k': ['main_ghi_all_settings.csv'],
    'l': ['main_ghi_all_settings.csv'],
}


def content_hash(path):
    data = path.read_bytes()
    return hashlib.sha256(gzip.decompress(data) if path.suffix == '.gz' else data).hexdigest()


def overview(output):
    """A labelled navigation preview; individual panels retain their native pixels."""
    mosaic = [['a']*6, ['b']*3+['c']*3, ['d']*3+['e']*3, ['f']*6,
              ['g']*2+['h']*2+['i']*2, ['j']*2+['k']*2+['l']*2]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(11, 18),
        gridspec_kw={'height_ratios': [1.25, 1.1, 1.1, 1, 1, 1.15]})
    for letter, ax in axes.items():
        ax.imshow(plt.imread(output/f'panel_{letter}.png'))
        ax.axis('off')
        ax.set_title(f'{letter}  {PANELS[letter]}', loc='left', fontsize=10, fontweight='bold', pad=7)
    fig.subplots_adjust(left=.025, right=.975, top=.965, bottom=.075, hspace=.30, wspace=.30)
    fig.suptitle('Figure 7 | Sorting and regrouping simulation', fontsize=15, y=.993)
    fig.text(.04, .046, 'S1  Capacity-estimate-only    S2  Conservative    S3  Lower-bound', fontsize=10)
    fig.text(.04, .032, 'S4  Uncertainty-guided    S5  Uncertainty-guided + targeted RPT', fontsize=10)
    fig.text(.04, .015, 'Overview preview; use the separate 600 dpi panels for manuscript assembly.', fontsize=9, color='#555555')
    fig.savefig(output/'Figure_7_overview.png', dpi=180, facecolor='white')
    plt.close(fig)


def render(data, output, source_data=None):
    data, output = Path(data).resolve(), validate_output(output)
    source_data=validate_output(source_data or MODULE/'results/figure_data')
    for folder in [output, source_data, output/'qa/workflow']:
        folder.mkdir(parents=True, exist_ok=True)
    references = json.loads((HERE/'reference_inputs.json').read_text(encoding='utf-8'))
    for name, expected in references.items():
        assert content_hash(data/name) == expected, f'Frozen numerical result changed: {name}'
    fs.DATA, fs.QA = data, output/'qa'
    mp.OUTPUT, merged.OUTPUT, merged.QA = output, output, output
    merged.SOURCE=source_data; fs.SOURCE=source_data
    mp.TABLES.clear(); mp.REPORT.clear()
    raw, summary, mixture, paired, examples = fs.load()
    ref = raw.loc[raw.n_train_total.eq(1000) & raw['tail'].eq(50)].copy()
    assert len(ref) == 15000

    # Draw the established composite in memory to collect its original source tables.
    # Independent panels below are freshly drawn at native dimensions, never cropped.
    with mp.callbacks(fs, write_table=mp.record_table), mp.callbacks(mechanism, write_table=mp.record_table):
        fig = mp.clean_frame(height_mm=merged.HEIGHT)
        for letter in merged.HEADINGS:
            merged.draw_panel(fig, letter, raw, paired, examples, ref)
        plt.close(fig)
        for letter, pixels in merged.PANEL_PIXELS.items():
            fig = mp.clean_frame(height_mm=pixels[1]/mp.DPI*25.4)
            merged.draw_panel(fig, letter, raw, paired, examples, ref, individual=True)
            mp.save_png(fig, output/f'panel_{letter}.png', independent=True, main=True, canvas_pixels=pixels)
    with mp.callbacks(fs, panel=lambda *args, **kwargs: None, write_table=mp.record_table):
        for letter, source in merged.NATIVE_DRAWERS.items():
            fig = mp.clean_frame(height_mm=130 if letter == 'b' else 210)
            if letter == 'b':
                d = examples.loc[examples.condition.eq('t50_i4_s1') & examples.method.eq('quadrant_c')].copy()
                mp.main2_scatter(fig, d)
            else:
                mp.draw_main1_panel(fig, source, raw, paired, examples, ref)
                if letter == 'f':
                    ax = next(a for a in fig.axes if a.get_gid() == 'sorting_heatmap')
                    ax.set_xticks(range(6), [t.get_text()+'\n(%)' for t in ax.get_xticklabels()])
            mp.save_png(fig, output/f'panel_{letter}.png', independent=True, main=True)
    checked = merged.verify_sources()

    workflow.DEST, workflow.QA = output, output/'qa/workflow'
    workflow.TEXT.clear()
    workflow.main()
    shutil.copyfile(HERE/'FIGURE_7_CAPTION.md', output/'FIGURE_7_CAPTION.md')
    overview(output)

    rows = []
    for letter, title in PANELS.items():
        path = output/f'panel_{letter}.png'
        with Image.open(path) as im:
            assert all(abs(v-600) < .1 for v in im.info['dpi'])
            pixels = list(im.size)
        rows.append(dict(panel=letter, title=title, file=f'panel_{letter}.png',
            pixels=pixels, dpi=600, sha256=fs.sha(path),
            source_data=PANEL_SOURCES[letter]))
    assert len(list(output.glob('panel_*.png'))) == 12
    indexed = {r['file']: r for r in mp.REPORT}
    assert indexed['panel_c.png']['pixels'] == indexed['panel_d.png']['pixels']
    for a, b in [('e','g'), ('h','i')]:
        assert indexed[f'panel_{a}.png']['pixels'][1] == indexed[f'panel_{b}.png']['pixels'][1]
    assert all(r['min_font_pt'] >= 8.5 and r['text_bounds_checked'] for r in mp.REPORT)
    manifest = dict(figure='Figure 7', dataset='LFP dataset', backend='Python/matplotlib',
        formats=['PNG'], panel_labels='Filenames identify a-l; independent panels omit outer letters/titles for manuscript assembly.',
        panels=rows, source_data_sha256={p.name:fs.sha(p) for p in sorted(source_data.glob('*.csv'))},
        source_data_directory=path_reference(source_data,output),
        input_content_sha256=references,
        renderer_source_sha256={p.name:fs.sha(p) for p in sorted(HERE.glob('*.py'))},
        model_retraining=False, simulation_rerun=False,
        evaluation_type='simulation_only', physical_pack_validation=False,
        representative_batch=dict(condition='t50_i4_s1', batch_id=0, records=300, rpt_records=13, packs=11),
        uncertainty='Central 95% batch ranges; not population mean confidence intervals.')
    (output/'panel_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    report = dict(status='PASS', panels=12, dpi=600, source_table_comparisons=checked,
        frozen_result_tables_checked=len(references), scientific_data_changed=False,
        all_panels_drawn_from_code=True, png_panels_used_as_drawing_inputs=False,
        overview_uses_exported_panels=True, outputs=mp.REPORT)
    (output/'qa/verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'PASS: Figure 7 a-l, {checked} source-table comparisons, {len(references)} frozen tables.', flush=True)
    return manifest


def verify(data, output):
    """Read-only validation of the exported panels and their frozen numerical sources."""
    data, output = Path(data), Path(output)
    manifest = json.loads((output/'panel_manifest.json').read_text(encoding='utf-8'))
    source_data=(output/manifest['source_data_directory']).resolve()
    reference_inputs = json.loads((HERE/'reference_inputs.json').read_text(encoding='utf-8'))
    assert manifest['input_content_sha256'] == reference_inputs
    for name, expected in reference_inputs.items():
        assert content_hash(data/name) == expected, name
    assert [row['panel'] for row in manifest['panels']] == list('abcdefghijkl')
    assert {p.name for p in output.glob('panel_*.png')} == {f'panel_{c}.png' for c in 'abcdefghijkl'}
    for row in manifest['panels']:
        assert row['file'] == f"panel_{row['panel']}.png"
        path = output/row['file']
        assert fs.sha(path) == row['sha256'], row['file']
        with Image.open(path) as im:
            assert list(im.size) == row['pixels']
            assert all(abs(d-600) < .1 for d in im.info['dpi'])
        for source in row['source_data']:
            assert (source_data/source).is_file(), source
    for name, expected in manifest['source_data_sha256'].items():
        assert fs.sha(source_data/name) == expected, name
    for reference in (HERE.parent/'results/reference/figure_source').glob('*.csv'):
        expected = pd.read_csv(reference)
        if reference.name == 'main_de_distribution_summary.csv':
            actual = pd.concat([pd.read_csv(source_data/f'merged_{p}_distribution_summary.csv') for p in 'hi'], ignore_index=True)
        else:
            actual = pd.read_csv(source_data/reference.name)
        if reference.name == 'main_f_sorting_quality.csv':
            mask = expected.metric.eq('bad_cell_recall') & expected.mean_percent.round(2).eq(100)
            expected.loc[mask, 'display_decimals'] = 4
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-11, atol=1e-11)
    assert (output/'Figure_7_overview.png').is_file()
    return dict(figure='Figure 7', panels=12, dpi=600, source_table_comparisons=9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=MODULE/'results/reference')
    parser.add_argument('--output', type=Path, default=MODULE/'visualization/outputs')
    parser.add_argument('--source-data',type=Path,default=MODULE/'results/figure_data')
    args = parser.parse_args()
    render(args.data, args.output, args.source_data)
