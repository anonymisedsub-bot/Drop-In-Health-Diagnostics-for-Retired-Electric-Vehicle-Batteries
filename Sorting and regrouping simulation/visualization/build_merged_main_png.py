"""Merge the selected mechanism and outcome panels; author supplies workflow a."""


from pathlib import Path


import json


import numpy as np


import pandas as pd


from matplotlib import pyplot as plt


from matplotlib.patches import Rectangle


from PIL import Image, ImageOps, ImageDraw


import figure_style as fs


import build_main_figure as old


import build_manuscript_png as mp


import plot_mechanism as mechanism


HERE=Path(__file__).resolve().parent


OUTPUT=HERE/'outputs'


QA=HERE/'outputs/qa'
SOURCE=HERE.parent/'results/figure_data'


HEIGHT=297


HEADINGS={
    'b':(2,27,45.5,'Uncertainty-guided selection'),
    'c':(91,27,134.5,'Cell sorting and targeted RPT'),
    'd':(2,79,45.5,'Cell capacities within packs'),
    'e':(91,79,134.5,'Pack capacity by pack'),
    'f':(2,134,90,'Sorting quality and residual risk (%)'),
    'g':(2,185,31,'Quality and yield'),
    'h':(64,185,92,'Pack capacity'),
    'i':(124,185,151,'Within-pack consistency'),
    'j':(2,234,28,'RPT: yield gain (pp)'),
    'k':(56,234,86,'RPT: pass change (pp)'),
    'l':(118,234,148,'RPT workload'),
}


NATIVE_DRAWERS={'b':'mechanism_scatter','f':'f','g':'c','h':'d','j':'g','k':'h','l':'i'}


AXIS=fs.axis_mm


PANEL_PIXELS={'c':(2008,1157),'d':(2008,1157),'e':(1287,958),'i':(1247,958)}


def axes_at(bounds):
    return lambda fig,x,y,w,h:AXIS(fig,*bounds)


def sorting_axes(fig,x,y,w,h):
    return AXIS(fig,25+(x-74)*153/103,170 if h==2 else 149,w*153/103,h if h==2 else 19)


def pack_data(d):
    assembled=d.loc[d.pack_number.ge(0)].sort_values('accepted_rank')
    k=assembled.pack_number.nunique()
    values=assembled.y_true.to_numpy().reshape(k,8)
    retested=assembled.rpt_selected.to_numpy(bool).reshape(k,8)
    assert k==11 and len(assembled)==88
    mp.record_table(assembled,'main2_assembled_cells.csv')
    mp.record_table(pd.DataFrame({'pack_number':np.arange(k),'capacity_ah':values.min(axis=1),
        'sd_ah':values.std(axis=1,ddof=1),'rpt_records':retested.sum(axis=1)}),'main2_pack_metrics.csv')
    return values,retested


def pack_heatmap(fig,d,individual=False):
    values,retested=pack_data(d)
    ax=AXIS(fig,*( (11.5,2.5,55,35.5) if individual else (14,90,53,31) ))
    ax.set_gid('pack_cell_capacities')
    im=ax.imshow(values,aspect='auto',cmap=fs.CAP_CMAP,vmin=13.5,vmax=16.5,interpolation='none')
    for yy,xx in np.argwhere(retested):
        ax.add_patch(Rectangle((xx-.45,yy-.43),.90,.86,fill=False,ec='#242424',lw=.7))
    ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(len(values)),range(1,len(values)+1))
    ax.set_xlabel('Position within an eight-cell pack',labelpad=2)
    ax.set_ylabel('Pack',labelpad=3)
    ax.tick_params(axis='both',length=0,pad=2)
    for spine in ax.spines.values():spine.set_visible(False)
    cax=AXIS(fig,*( (68,2.5,2.5,35.5) if individual else (70,90,2.5,31) ))
    cb=fig.colorbar(im,cax=cax,ticks=[14,15,16])
    cb.ax.tick_params(labelsize=8.5,pad=2,length=2)
    cb.set_label('Cell capacity (Ah)',fontsize=8.5,labelpad=3)
    cb.outline.set_visible(False)
    for spine in cax.spines.values():spine.set_visible(False)


def pack_capacity(fig,d,individual=False):
    values,_=pack_data(d)
    ax=AXIS(fig,*( (12.3,2.4,40,28) if individual else (109,90,66,24) ))
    ax.set_gid('pack_minimum_capacity')
    ax.scatter(np.arange(1,len(values)+1),values.min(axis=1),s=20,c=fs.COLORS['quadrant_c'])
    ax.axhline(14,color='#555555',lw=.7,ls='--')
    ax.set(xlim=(.5,len(values)+.5),ylim=(13.8,16.3),xlabel='Pack',ylabel='Pack capacity (Ah)')
    ax.set_xticks(np.arange(1,len(values)+1,2));ax.set_yticks([14,15,16])
    ax.xaxis.labelpad=2;ax.yaxis.labelpad=2


def paired_axes(fig,x,y,w,h):
    return AXIS(fig,x,279 if h==2 else 244,w,2 if h==2 else 23)


def draw_panel(fig,letter,raw,paired,examples,ref,individual=False):
    if not individual:
        x,y,center,title=HEADINGS[letter]
        mp.centered_panel(fig,x,y,letter,title,center)
    with mp.callbacks(fs,panel=lambda *args,**kwargs:None):
        if letter in 'bcde':
            d=examples.loc[examples.condition.eq('t50_i4_s1')&examples.method.eq('quadrant_c')].copy()
            assert len(d)==300 and d.rpt_selected.sum()==13
            if letter=='b':
                with mp.callbacks(fs,axis_mm=axes_at((15,37,64,29))):mp.main2_scatter(fig,d)
                d['interval_width']=d.upper-d.lower
                mp.record_table(d,'main_b_example_300_records.csv')
            elif letter=='c':
                bounds=(2,2,81,45) if individual else (93,35,81,40)
                with mp.callbacks(fs,axis_mm=axes_at(bounds)):mp.main2_flow(fig,d)
            elif letter=='d':pack_heatmap(fig,d,individual)
            else:pack_capacity(fig,d,individual)
        elif letter=='f':
            with mp.callbacks(fs,axis_mm=sorting_axes):old.sorting(fig,ref)
            if individual:
                ax=next(a for a in fig.axes if a.get_gid()=='sorting_heatmap')
                ax.set_xticks(range(6),[t.get_text()+'\n(%)' for t in ax.get_xticklabels()])
        elif letter=='g':
            with mp.callbacks(fs,axis_mm=axes_at((15,196,41,24))):old.quality_yield(fig,ref)
        elif letter in 'hi':
            capacity=letter=='h';metric='cap_mean' if capacity else 'sd_mean'
            low,high=ref[metric].min(),ref[metric].max();pad=.03 if capacity else .008
            rows=[]
            bounds=(7,2,42,28) if individual else (76 if capacity else 138,196,41 if capacity else 39,24)
            old.distribution(fig,ref,metric,*bounds,
                letter,'Pack capacity' if capacity else 'Within-pack consistency',
                'Mean capacity (Ah)' if capacity else 'Mean SD (Ah)',
                (low-pad,high+pad),[14.6,14.8,15.0] if capacity else [.1,.2,.3],rows)
            fig.axes[-1].set_gid('pack_capacity_distribution' if capacity else 'within_pack_sd_distribution')
            mp.record_table(pd.DataFrame(rows),f'merged_{letter}_distribution_summary.csv')
        else:
            source={'j':'g','k':'h','l':'i'}[letter]
            with mp.callbacks(fs,axis_mm=paired_axes):
                old.all_settings(fig,raw,paired,include=(source,),g_labelpad=6)


def verify_sources():
    comparisons={
        'main_b_example_300_records.csv':HERE.parent/'results/reference/figure_source/main_b_example_300_records.csv',
        'main_c_quality_yield.csv':HERE.parent/'results/reference/figure_source/main_c_quality_yield.csv',
        'main_f_sorting_quality.csv':HERE.parent/'results/reference/figure_source/main_f_sorting_quality.csv',
        'main_f_residual_risk_records.csv':HERE.parent/'results/reference/figure_source/main_f_residual_risk_records.csv',
        'main_ghi_all_settings.csv':HERE.parent/'results/reference/figure_source/main_ghi_all_settings.csv',
        'fig01_flow_counts.csv':HERE.parent/'results/reference/figure_source/fig01_flow_counts.csv',
        'main2_pack_metrics.csv':HERE.parent/'results/reference/figure_source/main2_pack_metrics.csv',
        'main2_assembled_cells.csv':HERE.parent/'results/reference/figure_source/main2_assembled_cells.csv',
    }
    for name,path in comparisons.items():
        current=mp.TABLES[name].reset_index(drop=True)
        expected=pd.read_csv(path)
        if name=='main_c_quality_yield.csv':current['seed']=current.seed.astype(str)
        if name=='main_f_sorting_quality.csv':
            # Only display precision changes; retain the frozen numerical comparison.
            near_hundred=expected.metric.eq('bad_cell_recall') & expected.mean_percent.round(2).eq(100)
            expected.loc[near_hundred,'display_decimals']=4
        pd.testing.assert_frame_equal(current,expected,check_dtype=False,rtol=1e-11,atol=1e-11)
    distribution=pd.concat([mp.TABLES[f'merged_{p}_distribution_summary.csv'] for p in 'hi'],ignore_index=True)
    expected=pd.read_csv(HERE.parent/'results/reference/figure_source/main_de_distribution_summary.csv')
    pd.testing.assert_frame_equal(distribution,expected,check_dtype=False,rtol=1e-11,atol=1e-11)
    for name,table in mp.TABLES.items():table.to_csv(SOURCE/name,index=False)
    return len(comparisons)+1

