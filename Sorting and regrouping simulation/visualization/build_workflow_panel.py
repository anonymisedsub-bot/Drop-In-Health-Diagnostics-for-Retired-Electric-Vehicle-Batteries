"""Standalone icon-led schematic of the frozen sorting/regrouping simulation.

All glyphs are schematic, not additional experimental observations. The five
strategy colours and the simulation constants match the final evaluation.
"""


from pathlib import Path


import json


import hashlib


import numpy as np


from matplotlib import pyplot as plt


from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch, PathPatch


from matplotlib.path import Path as MplPath


from matplotlib.colors import to_rgb


from PIL import Image


import figure_style as fs


HERE=Path(__file__).resolve().parent


DEST=HERE/'outputs'


QA=HERE/'outputs/qa/workflow'








W,H,DPI=180,66,600


INK='#33434C';MUTED='#687780';BLUE='#839FB6';TEAL=fs.COLORS['quadrant_a'];PINK=fs.COLORS['quadrant_c']


INK='#33434C';MUTED='#687780';BLUE='#839FB6';TEAL=fs.COLORS['quadrant_a'];PINK=fs.COLORS['quadrant_c']


INK='#33434C';MUTED='#687780';BLUE='#839FB6';TEAL=fs.COLORS['quadrant_a'];PINK=fs.COLORS['quadrant_c']


INK='#33434C';MUTED='#687780';BLUE='#839FB6';TEAL=fs.COLORS['quadrant_a'];PINK=fs.COLORS['quadrant_c']


INK='#33434C';MUTED='#687780';BLUE='#839FB6';TEAL=fs.COLORS['quadrant_a'];PINK=fs.COLORS['quadrant_c']


GREY='#A6AFB5'


TEXT=[]


def tint(c,a=.65):
    return tuple((1-a)*np.array(to_rgb(c))+a)


def label(ax,x,y,s,size=8.5,color=INK,weight='normal',ha='center',va='center'):
    t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,weight=weight,
        linespacing=1.13,zorder=20)
    TEXT.append(t)
    return t


def arrow(ax,a,b,color=INK,lw=.75,dashed=True,connection='arc3,rad=0'):
    p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,linewidth=lw,
        linestyle=(0,(3,2)) if dashed else '-',color=color,connectionstyle=connection,
        shrinkA=0,shrinkB=0,zorder=7)
    ax.add_patch(p)


def line(ax,coords,color=INK,lw=.65,**kwargs):
    ax.plot(*np.array(coords).T,color=color,lw=lw,solid_capstyle='round',**kwargs)


def cell(ax,x,y,w=3.4,h=7,color=TEAL,z=5):
    """Small cylindrical cell with restrained shading and a visible terminal."""
    e=w*.32
    ax.add_patch(Ellipse((x+w*.53,y+h+.45),w*1.08,e*.95,fc='#536370',ec='none',alpha=.10,zorder=z))
    ax.add_patch(Rectangle((x,y),w,h,fc=tint(color,.48),ec=INK,lw=.45,zorder=z+1))
    ax.add_patch(Rectangle((x+w*.72,y+.15),w*.22,h-.15,fc=color,alpha=.28,ec='none',zorder=z+2))
    ax.add_patch(Ellipse((x+w/2,y+h),w,e,fc=tint(color,.42),ec=INK,lw=.45,zorder=z+3))
    ax.add_patch(Rectangle((x+.22,y+.1),w*.14,h-.3,fc='white',alpha=.50,ec='none',zorder=z+3))
    ax.add_patch(Ellipse((x+w/2,y),w,e,fc=tint(color,.88),ec=INK,lw=.45,zorder=z+4))
    ax.add_patch(Ellipse((x+w/2,y-.18),w*.34,e*.50,fc='#C7CFD3',ec=INK,lw=.4,zorder=z+5))


def database(ax,x,y,w=11,h=12):
    ax.add_patch(Ellipse((x+w*.54,y+h+1),w+1,3,fc=INK,ec='none',alpha=.08,zorder=1))
    ax.add_patch(Rectangle((x,y),w,h,fc=tint(BLUE,.55),ec=INK,lw=.5,zorder=2))
    ax.add_patch(Rectangle((x+w*.78,y),w*.22,h,fc=BLUE,alpha=.35,ec='none',zorder=3))
    for yy in [y+h,y+h*.66,y+h*.33,y]:
        ax.add_patch(Ellipse((x+w/2,yy),w,3,fc=tint(BLUE,.78 if yy==y else .48),ec=INK,lw=.5,zorder=4))
    for yy in [y+3.1,y+7.2]:
        ax.add_patch(Ellipse((x+2,yy),.7,.7,fc='white',ec='none',zorder=5))


def card(ax,x,y,w=15,h=15,back=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.55',
        fc='#E9EFF3' if back else 'white',ec=BLUE,lw=.6,zorder=3 if back else 4))


def bootstrap(ax):
    card(ax,38.7,16.2,16.2,16.2,True)
    card(ax,37.5,17.6,16.2,16.2,True)
    card(ax,36.3,19,16.2,16.2)
    for r in range(4):
        for c in range(4):
            fc=[tint(BLUE,.56),tint(TEAL,.62),tint(GREY,.58),tint(PINK,.63)][(r+c)%4]
            ax.add_patch(FancyBboxPatch((38+c*3.4,21+r*3.1),2.1,2.0,
                boxstyle='round,pad=0,rounding_size=.25',fc=fc,ec='none',zorder=5))
    # A circular arrow denotes sampling with replacement, rather than distinct cells.
    arrow(ax,(52.7,35.3),(55.5,27.8),color=BLUE,dashed=False,connection='arc3,rad=.7')


def strategies(ax):
    # Five concurrent rules applied to identical sampled batches.
    for j,(method,y) in enumerate(zip(fs.METHODS,[17,21.5,26,30.5,35])):
        col=fs.COLORS[method]
        label(ax,69.8,y+.6,f'S{j+1}',color=col,ha='left')
        ax.add_patch(Polygon([(76,y-1),(84,y-1),(82,y+1.2),(82,y+2.2),(79.2,y+2.2),(79.2,y+1.2)],
            closed=True,fc=tint(col,.64),ec=col,lw=.6,zorder=6))
        line(ax,[(64.8,26),(66.5,26),(66.5,y+.5),(68.5,y+.5)],color=tint(INK,.35),lw=.5)
        line(ax,[(84.6,y+.5),(87.2,y+.5),(87.2,24)],color=tint(col,.2),lw=.55)
    arrow(ax,(87.2,24),(103,24),color=TEAL)
    # The branch belongs to S5 only. Its outputs are explicitly measured-good / rejected.
    arrow(ax,(82,37.2),(82,43),color=PINK,dashed=False)
    ax.add_patch(FancyBboxPatch((74.7,44),15,8.5,boxstyle='round,pad=0,rounding_size=.6',
        fc=tint(PINK,.90),ec=PINK,lw=.65,zorder=5))
    label(ax,82.2,47.3,'RPT',color=PINK,weight='bold')
    # Instrument face + probe lead are schematic icons for the exact-measurement reveal.
    line(ax,[(77,50.1),(80,50.1),(80.8,48.9),(81.7,51.0),(83.0,49.8),(87,49.8)],color=PINK,lw=.65,zorder=7)
    label(ax,65.3,45.7,'High–high:\nwidest 20%',color=PINK)
    line(ax,[(89.7,48),(95,48),(95,24)],color=PINK,lw=.65,zorder=7)
    arrow(ax,(95,24),(103,24),color=PINK,dashed=False)
    label(ax,101.3,44.5,'≥14 Ah',color=PINK)
    # The displayed rejection outlet is the failed-RPT branch of S5.
    arrow(ax,(82.2,53),(82.2,58.3),color=GREY,dashed=False)
    label(ax,92.0,57.7,'Reject',color=MUTED)


def ranking(ax):
    # Icons represent a rank order, not measured capacities or a result chart.
    for i in range(8):
        cell(ax,106.0+i*2.6,24.8-i*.37,w=2.0,h=6.2,color=TEAL)
    arrow(ax,(105.8,35),(126.6,35),color=MUTED,dashed=False,lw=.65)
    label(ax,116.2,41.0,'Capacity order')
    label(ax,116.2,48.0,'Per strategy',color=MUTED)


def pack(ax):
    # Exactly eight cylinders in a schematic module. Electrical interconnects
    # are deliberately abstracted; the series configuration is labelled below.
    ax.add_patch(Ellipse((153.8,37.4),28.5,3,fc=INK,ec='none',alpha=.07,zorder=1))
    ax.add_patch(Polygon([(139,18.7),(163,18.7),(167,21.7),(167,36),(143,36),(139,32.8)],
        closed=True,fc='#E7EDF0',ec=BLUE,lw=.65,zorder=2))
    ax.add_patch(Polygon([(163,18.7),(167,21.7),(167,36),(163,32.8)],
        closed=True,fc='#CBD9E1',ec=BLUE,lw=.55,zorder=3))
    for row in range(2):
        for col in range(4):
            cell(ax,141.2+col*5.35+row*.6,21+row*5.9,w=3.8,h=6.5,color=TEAL,z=4+row*5)
    label(ax,152.4,41.0,'8 cells in series')
    # Capacity qualification and the principal outcome dimensions are compact labels.
    label(ax,152.4,48.8,'Target ≥14 Ah',weight='bold')
    label(ax,152.4,55.2,'Quality · yield',color=MUTED)


def main():
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=DPI)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
    label(ax,16,6.2,'Diagnosed pool',weight='bold')
    label(ax,16,11.8,'Capacity + interval',color=MUTED)
    label(ax,45.5,6.2,'Bootstrap',weight='bold')
    label(ax,81.2,6.2,'Five strategies',weight='bold')
    label(ax,116.8,6.2,'Regrouping',weight='bold')
    label(ax,153.3,6.2,'Second-life pack',weight='bold')
    database(ax,15,18.3,10.5,13)
    for r in range(2):
        for c in range(3):
            cell(ax,5.0+c*4.2+r*1.8,25+r*4.4,w=3.4,h=6.7,color=BLUE,z=5+r*6)
    label(ax,16,42.0,'609 records\nper model')
    arrow(ax,(28.1,25.5),(33.3,25.5))
    bootstrap(ax)
    label(ax,45.5,42,'300 / batch')
    label(ax,45.5,51.0,'With replacement',color=MUTED)
    arrow(ax,(57.0,26),(64,26))
    strategies(ax)
    ranking(ax)
    arrow(ax,(129.7,25.5),(136.0,25.5))
    pack(ax)
    # A repetition annotation applies to the entire paired simulation pipeline.
    line(ax,[(35,58),(35,61.1),(171.7,61.1),(171.7,58)],color='#A6B3BC',lw=.55)
    label(ax,103.2,64,'Per setting: 1,000 paired draws × 3 model seeds',color=MUTED)

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=[]
    for t in TEXT:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
        bounds.append((t,b))
    overlaps=[]
    for i,(a,ba) in enumerate(bounds):
        for b,bb in bounds[i+1:]:
            if ba.overlaps(bb):overlaps.append((a.get_text(),b.get_text()))
    assert not overlaps,overlaps
    path=DEST/'panel_a.png'
    fig.savefig(path,dpi=DPI,facecolor='white')
    plt.close(fig)
    with Image.open(path) as im:
        pixels=list(im.size);dpi=list(im.info['dpi'])
    report=dict(status='PASS',file='panel_a.png',pixels=pixels,dpi=dpi,
        width_mm=W,height_mm=H,min_font_pt=min(t.get_fontsize() for t in TEXT),
        text_bounds_checked=True,text_pair_overlaps=overlaps,
        strategy_ids=['S1','S2','S3','S4','S5'],pack_cells=8,
        records_per_model=609,batch_size=300,draws_per_seed=1000,seeds_per_setting=3,
        rpt_selection='S5 only: highest-uncertainty 20% within the high-capacity/high-uncertainty quadrant',
        scientific_results_changed=False,illustration_only=True,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),labels=[t.get_text() for t in TEXT])
    (QA/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PNG: panel_a.png', flush=True)

