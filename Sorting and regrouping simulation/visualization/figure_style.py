"""Physical-size-aware plotting helpers for the final scientific figure set."""


from pathlib import Path


import hashlib


import json


import numpy as np


import pandas as pd


import matplotlib


matplotlib.use('Agg')


import matplotlib.pyplot as plt


from matplotlib.text import Text


from matplotlib.lines import Line2D


from matplotlib.colors import LinearSegmentedColormap


HERE=Path(__file__).resolve().parent


FINAL=HERE.parent


DATA=FINAL/'results/reference'


EXPORT=HERE/'outputs'


SOURCE=FINAL/'results/figure_data'


QA=HERE/'outputs/qa'


METHODS=('point','conservative','lower_bound','quadrant_a','quadrant_c')


IDS={m:f'S{i+1}' for i,m in enumerate(METHODS)}


COLORS={'point':'#7D858D','conservative':'#BC9769','lower_bound':'#869A72',
        'quadrant_a':'#548D9C','quadrant_c':'#96566B'}


NAMES=['Capacity-estimate-only','Conservative','Lower-bound',
       'Uncertainty-guided sorting','Uncertainty-guided sorting + targeted RPT']


MARKERS=['o','s','^','D','P']


TAIL_RANGES={46:'2.5–2.73 V',50:'2.5–2.75 V',54:'2.5–2.77 V'}


TAIL_LABEL='Tail segment setting (V)'


POOL_LABEL='Pool usage ratio (%)'


MIN_FONT=8.5


WIDTH=180


HEIGHT=168


EXPORT_LOG=[]


plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,
    'xtick.labelsize':MIN_FONT,'ytick.labelsize':MIN_FONT,'legend.fontsize':MIN_FONT,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.4,
    'xtick.major.width':.4,'ytick.major.width':.4,'xtick.minor.width':.35,'ytick.minor.width':.35,
    'xtick.major.size':2.5,'ytick.major.size':2.5,
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
    'figure.facecolor':'white','text.color':'#1D1D1D','axes.labelcolor':'#1D1D1D',
    'xtick.color':'#1D1D1D','ytick.color':'#1D1D1D'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_mm(fig,x,y,text,**kwargs):
    width,height=fig.get_size_inches()*25.4
    return fig.text(x/width,1-y/height,text,va=kwargs.pop('va','top'),**kwargs)


def axis_mm(fig,x,y,w,h):
    width,height=fig.get_size_inches()*25.4
    return fig.add_axes([x/width,1-(y+h)/height,w/width,h/height])


def panel(fig,x,y,letter,title):
    text_mm(fig,x,y,letter,fontsize=10.5,weight='bold')
    text_mm(fig,x+5,y+.4,title,fontsize=9.5,weight='bold')


def fade(color):
    from matplotlib.colors import to_rgb
    return tuple(.20*np.array(to_rgb(color))+.80)


def write_table(frame,name):
    frame.to_csv(SOURCE/name,index=False)


def load():
    filenames=['batch_metrics.csv.gz','summary_by_model.csv','summary_by_pool_and_vtail.csv',
        'paired_batch_effects.csv.gz','paired_summary_by_pool_and_vtail.csv','example_batch0_assignments.csv.gz']
    inputs={name:sha(DATA/name) for name in filenames}
    (QA/'input_manifest.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    raw=pd.read_csv(DATA/filenames[0])
    assert len(raw)==135000 and set(raw.method)==set(METHODS)
    return raw,pd.read_csv(DATA/filenames[1]),pd.read_csv(DATA/filenames[2]),pd.read_csv(DATA/filenames[4]),pd.read_csv(DATA/filenames[5])


CAP_CMAP=LinearSegmentedColormap.from_list('cell_capacity',['#F3F5F5','#C3D9E1','#84A9BE','#645D80'])


def colorbar_mm(fig,cmap,norm,x,y,w,h,ticks,orientation='horizontal'):
    from matplotlib.cm import ScalarMappable
    ax=axis_mm(fig,x,y,w,h)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation=orientation,ticks=ticks)
    cb.outline.set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb.ax.tick_params(labelsize=8.5,length=2,pad=1,width=.35)
    if orientation=='horizontal':
        labels=cb.ax.get_xticklabels()
        labels[0].set_ha('left');labels[-1].set_ha('right')
    return cb

