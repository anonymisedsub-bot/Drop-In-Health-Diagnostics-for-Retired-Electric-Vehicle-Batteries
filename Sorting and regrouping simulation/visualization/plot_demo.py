"""Plot a small regrouping demonstration from its numerical summary."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from portable_io import validate_output

def render(source, destination):
    destination=validate_output(destination)
    destination.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(source)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,col,label in zip(axes,['yield_rate','rpt_rate'],['Effective yield (%)','Oracle RPT occurrences (%)']):
        ax.bar(np.arange(5),summary[col]*100); ax.set_xticks(np.arange(5),['S1','S2','S3','S4','S5']); ax.set_ylabel(label)
    fig.suptitle('Historical cohort: 20 paired batches; checkpoint inference; simulation only')
    for ext in ['png','pdf','svg']: fig.savefig(destination/f'demo_results.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__':
    render(Path(sys.argv[1]),Path(sys.argv[2]))
