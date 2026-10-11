"""Shared native-size panel renderers and PNG export helpers.

The frozen numerical evaluation is reused. Drawing callbacks create fresh matplotlib
figures for individual panels; no panel is cut from a rendered composite image.
"""


from contextlib import contextmanager


from pathlib import Path


import hashlib


import json


import re


import numpy as np


import pandas as pd


from PIL import Image


from matplotlib import pyplot as plt


from matplotlib.lines import Line2D


from matplotlib.patches import Rectangle


from matplotlib.collections import PathCollection


from matplotlib.transforms import Bbox


import figure_style as fs


import build_main_figure as main1


import plot_mechanism as mechanism


import plot_outcomes as outcomes


HERE=Path(__file__).resolve().parent


OUTPUT=HERE/'outputs'


QA=HERE/'outputs/qa'


DPI=600


REPORT=[]


TABLES={}


def record_table(table,name):
    TABLES[name]=table.copy()


@contextmanager
def callbacks(module,**replacements):
    previous={name:getattr(module,name) for name in replacements}
    for name,value in replacements.items():setattr(module,name,value)
    try:yield
    finally:
        for name,value in previous.items():setattr(module,name,value)


def clean_frame(title='',subtitle='',height_mm=None):
    return plt.figure(figsize=(180/25.4,(height_mm or 210)/25.4))


def centered_panel(fig,x,y,letter,title,center):
    """Keep the panel ID anchor and center its title within the panel column."""
    label=fs.text_mm(fig,x,y,letter,fontsize=10.5,weight='bold')
    heading=fs.text_mm(fig,center,y+.4,title,fontsize=9.5,weight='bold',ha='center')
    label.set_gid('panel-label-'+letter);heading.set_gid('panel-title-'+letter)
    if not hasattr(fig,'_panel_heading_audit'):fig._panel_heading_audit=[]
    fig._panel_heading_audit.append((letter,x,y,center,label,heading))


def thin_axes(fig):
    for ax in fig.axes:
        for spine in ax.spines.values():spine.set_linewidth(.3)
        ax.tick_params(which='major',width=.25 if getattr(ax,'_colorbar',None) else .3)
        ax.tick_params(which='minor',width=.25)


def visible_text(fig):
    artists=list(fig.texts)
    for legend in fig.legends:artists.extend(legend.get_texts())
    for ax in fig.axes:
        artists.extend(ax.texts)
        if ax.axison:
            artists.extend([ax.xaxis.label,ax.yaxis.label,ax.title])
            for which in ['x','y']:
                lo,hi=sorted(getattr(ax,f'get_{which}lim')())
                ticks=getattr(ax,f'get_{which}ticks')()
                artists.extend(label for val,label in zip(ticks,getattr(ax,f'get_{which}ticklabels')())
                    if lo-1e-8<=val<=hi+1e-8)
    return [a for a in artists if a.get_visible() and a.get_text().strip()]


def save_png(fig,path,independent=False,main=False,canvas_pixels=None):
    if main:thin_axes(fig)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    texts=visible_text(fig)
    assert min(t.get_fontsize() for t in texts)>=8.5
    words=[t.get_text() for t in texts]
    assert not any('13 RPT records' in t or 'RPT: top 20%' in t or t=='Screening' for t in words)
    for word in words:
        for match in re.finditer(r'2\.5[–-]2\.(?:73|75|77)',word):
            assert word[match.end():].startswith(' V'),word
    if independent:assert not fig.texts,'Separate panels must not have panel letters or titles'
    renderer=fig.canvas.get_renderer()
    headings=[]
    for letter,x,y,center,label,heading in getattr(fig,'_panel_heading_audit',[]):
        width,height=fig.get_size_inches()*25.4
        np.testing.assert_allclose(label.get_position(),[x/width,1-y/height])
        assert heading.get_ha()=='center'
        bbox=heading.get_window_extent(renderer)
        expected=fig.transFigure.transform((center/width,0))[0]
        assert abs((bbox.x0+bbox.x1)/2-expected)<.05
        for other in fig.texts:
            if (other.get_gid() or '').startswith('panel-label-'):
                assert not bbox.overlaps(other.get_window_extent(renderer)),(letter,other.get_text())
        headings.append(dict(panel=letter,label_x_mm=x,label_y_mm=y,title_center_x_mm=center))
    # Colorbar tight-layout bounds can shorten their axis-label extent. Include
    # every rendered text bbox explicitly so long standalone labels stay intact.
    text_bounds=[t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()) for t in texts]
    bounds=Bbox.union([fig.get_tightbbox(renderer),*text_bounds]).padded(2/25.4)
    if canvas_pixels is not None:
        # Explicit native-size exports keep matched panels at identical pixel
        # dimensions without rescaling their axes, markers, or typography.
        bounds=Bbox.from_bounds(0,0,*[n/DPI for n in canvas_pixels])
    assert all(bounds.contains(b.x0,b.y0) and bounds.contains(b.x1,b.y1) for b in text_bounds)
    fig.savefig(path,dpi=DPI,bbox_inches=bounds,facecolor='white')
    row=dict(file=path.relative_to(OUTPUT).as_posix(),independently_drawn=independent,text_bounds_checked=True,
        width_mm=bounds.width*25.4,height_mm=bounds.height*25.4,
        min_font_pt=min(t.get_fontsize() for t in texts),text=words,panel_headings=headings,
        visible_spine_widths_pt=[s.get_linewidth() for ax in fig.axes if ax.axison for s in ax.spines.values() if s.get_visible()],
        axes=[dict(gid=ax.get_gid(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
            width_mm=ax.get_position().width*180,
            height_mm=ax.get_position().height*fig.get_size_inches()[1]*25.4,
            scatter_points=sum(len(c.get_offsets()) for c in ax.collections if isinstance(c,PathCollection)))
            for ax in fig.axes],
        colorbars=[dict(label=ax.get_ylabel() or ax.get_xlabel(),outline_visible=ax._colorbar.outline.get_visible())
            for ax in fig.axes if getattr(ax,'_colorbar',None)])
    with Image.open(path) as im:
        row['pixels']=list(im.size);row['dpi']=list(im.info['dpi'])
        if canvas_pixels is not None:assert im.size==tuple(canvas_pixels),(im.size,canvas_pixels)
        assert all(abs(v-DPI)<.1 for v in row['dpi'])
    row['sha256']=fs.sha(path)
    REPORT.append(row)
    plt.close(fig)
    print('PNG:',path.relative_to(OUTPUT),flush=True)


def draw_main1_panel(fig,letter,raw,paired,examples,ref):
    if letter=='b':
        d=examples.loc[examples.condition.eq('t50_i4_s1')&examples.method.eq('quadrant_c')].copy()
        assert len(d)==300 and d.rpt_selected.sum()==13
        d['interval_width']=d.upper-d.lower
        main2_scatter(fig,d,overview=True)
        record_table(d,'main_b_example_300_records.csv')
    elif letter=='c':main1.quality_yield(fig,ref)
    elif letter in 'de':
        metric='cap_mean' if letter=='d' else 'sd_mean'
        lo,hi=ref[metric].min(),ref[metric].max()
        if letter=='d':
            main1.distribution(fig,ref,metric,138,49,39,28,'d','Pack capacity','Mean capacity (Ah)',
                (lo-.03,hi+.03),[14.6,14.8,15.0],[])
        else:
            main1.distribution(fig,ref,metric,16,99,42,24,'e','Within-pack consistency','Mean SD (Ah)',
                (lo-.008,hi+.008),[.1,.2,.3],[])
    elif letter=='f':main1.sorting(fig,ref)
    else:main1.all_settings(fig,raw,paired,include=(letter,),g_labelpad=6)


def main2_scatter(fig,d,overview=False):
    if overview:
        fs.panel(fig,2,38,'b','Targeted sorting')
        ax=fs.axis_mm(fig,14,49,42,28)
    else:
        fs.panel(fig,2,3,'b','Select the high-capacity, uncertain records')
        ax=fs.axis_mm(fig,15,12,67,42)
    ax.set_gid('prediction_uncertainty')
    width=d.upper-d.lower;median=d.median_uncertainty.iloc[0]
    ymax=max(width.max()*1.08,median*1.5);xmin=min(9.,d.y_pred.min()-.1);xmax=max(16.5,d.y_pred.max()+.1)
    if overview:xmin,xmax,ymax=9.,16.5,3.5
    ax.add_patch(Rectangle((14,median),xmax-14,ymax-median,fc='#F4E9ED',ec='none',zorder=0))
    tested=d.rpt_selected.to_numpy(bool);keep=d.accepted_rank.ge(0).to_numpy()
    for mask,color,mark in [(~tested&~keep,'#BDC3C8','o'),(~tested&keep,fs.COLORS['quadrant_a'],'o'),
            (tested,fs.COLORS['quadrant_c'],'D')]:
        ax.scatter(d.y_pred[mask],width[mask],s=27 if mark!='D' else 45,c=color,marker=mark,
            alpha=.65,linewidths=.35,edgecolors='white')
    ax.axvline(14,color='#666666',lw=.7,ls='--');ax.axhline(median,color='#666666',lw=.7,ls=':')
    ax.set(xlim=(xmin,xmax),ylim=(0,ymax),xlabel='Estimated capacity (Ah)',ylabel='Interval width (Ah)')
    ax.set_xticks([10,12,14,16]);ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    if overview:
        ax.set_yticks([0,1,2,3]);ax.xaxis.labelpad=2;ax.yaxis.labelpad=2


def main2_flow(fig,d):
    fs.panel(fig,91,3,'c','Where the 300 records go')
    ax=fs.axis_mm(fig,93,11,85,45);ax.set_gid('record_flow')
    mechanism.alluvial(ax,d)
    for text in ax.texts:
        if text.get_text()=='Screening':text.set_text('Sorting')

