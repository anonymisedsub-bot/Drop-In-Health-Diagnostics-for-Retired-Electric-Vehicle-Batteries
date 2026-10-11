"""Truth-free decision schematic and one predeclared, traceable bootstrap example."""


import numpy as np


import pandas as pd


from matplotlib.patches import Rectangle,FancyBboxPatch,PathPatch


from matplotlib.path import Path as MplPath


from figure_style import *


def ribbon(ax,x0,y0,x1,y1,height,color):
    mid=(x0+x1)/2
    vertices=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+height),
        (mid,y1+height),(mid,y0+height),(x0,y0+height),(x0,y0)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,
        MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices,codes),fc=fade(color),ec='none'))


def alluvial(ax,d):
    tested=d.rpt_selected.to_numpy(bool)
    good=d.rpt_measurement.ge(14).to_numpy()
    pred_good=d.y_pred.ge(14).to_numpy()
    counts={'capacity_rejected':int((~pred_good).sum()),'untested_retained':int((pred_good&~tested).sum()),
        'rpt_good':int((tested&good).sum()),'rpt_bad':int((tested&~good).sum())}
    assert sum(counts.values())==300
    nr=counts['capacity_rejected']; ng=counts['untested_retained']; rg=counts['rpt_good']; rb=counts['rpt_bad']
    # Ribbon and node heights are all proportional to record count, without minimum-height inflation.
    scale=.78/300
    x=[.025,.44,.855]; width=.12
    mid=[.06,.06+nr*scale+.025,.06+(nr+rg+rb)*scale+.05]
    end=[.06,.06+(nr+rb)*scale+.05]
    ribbon(ax,x[0]+width,.06,x[1],mid[0],nr*scale,'#A9B0B8')
    ribbon(ax,x[0]+width,.06+nr*scale,x[1],mid[1],(rg+rb)*scale,COLORS['quadrant_c'])
    ribbon(ax,x[0]+width,.06+(nr+rg+rb)*scale,x[1],mid[2],ng*scale,COLORS['quadrant_a'])
    ribbon(ax,x[1]+width,mid[0],x[2],end[0],nr*scale,'#A9B0B8')
    ribbon(ax,x[1]+width,mid[1],x[2],end[0]+nr*scale,rb*scale,COLORS['quadrant_c'])
    ribbon(ax,x[1]+width,mid[1]+rb*scale,x[2],end[1],rg*scale,COLORS['quadrant_c'])
    ribbon(ax,x[1]+width,mid[2],x[2],end[1]+rg*scale,ng*scale,COLORS['quadrant_a'])
    nodes=[(x[0],.06,300,'#DAE2E7','300'),(x[1],mid[0],nr,'#DCE0E4',str(nr)),
        (x[1],mid[1],rg+rb,fade(COLORS['quadrant_c']),str(rg+rb)),
        (x[1],mid[2],ng,fade(COLORS['quadrant_a']),str(ng)),
        (x[2],end[0],nr+rb,'#DCE0E4',str(nr+rb)),
        (x[2],end[1],ng+rg,fade(COLORS['quadrant_a']),str(ng+rg))]
    for xx,yy,n,color,label in nodes:
        ax.add_patch(Rectangle((xx,yy),width,n*scale,fc=color,ec='#757575',lw=.45))
        ax.text(xx+width/2,yy+n*scale/2,label,ha='center',va='center',fontsize=8.5)
    ax.text(.085,.96,'Input',ha='center',va='top',fontsize=8.5)
    ax.text(.50,.96,'Screening',ha='center',va='top',fontsize=8.5)
    ax.text(.915,.96,'Final',ha='center',va='top',fontsize=8.5)
    ax.text(.40,mid[0]+nr*scale/2,'Capacity\nreject',ha='right',va='center',fontsize=8.5)
    ax.text(.40,mid[1]+(rg+rb)*scale/2,'RPT',ha='right',va='center',fontsize=8.5)
    ax.text(.40,mid[2]+ng*scale/2,'Retain',ha='right',va='center',fontsize=8.5)
    ax.text(.845,end[0]+(nr+rb)*scale/2,'Reject',ha='right',va='center',fontsize=8.5)
    ax.text(.845,end[1]+(ng+rg)*scale/2,'Retain',ha='right',va='center',fontsize=8.5)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    write_table(pd.DataFrame([{'route':k,'records':v} for k,v in counts.items()]),'fig01_flow_counts.csv')

