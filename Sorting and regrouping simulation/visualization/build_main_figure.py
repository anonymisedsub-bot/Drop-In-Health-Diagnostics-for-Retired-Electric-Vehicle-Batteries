"""Nine-panel overview of the frozen five-strategy results; no simulation runs."""


import json


import numpy as np


import pandas as pd


from scipy.stats import gaussian_kde


from matplotlib import pyplot as plt


from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


from matplotlib.patches import FancyBboxPatch


import figure_style as fs


from plot_outcomes import place_labels


ROOT=fs.HERE


fs.EXPORT=ROOT/'outputs'


fs.SOURCE=ROOT.parent/'results/figure_data'


fs.QA=ROOT/'outputs/qa'


fs.HEIGHT=210


METHODS=fs.METHODS


CONFIGS=[(n,t) for n in [400,1000,1600] for t in [46,50,54]]


def quality_yield(fig,ref):
    fs.panel(fig,64,38,'c','Quality and yield')
    ax=fs.axis_mm(fig,77,49,40,28)
    ax.set_gid('quality_yield')
    ax.set(xlim=(21,29),ylim=(92,102),xlabel='Effective yield (%)',ylabel='Pack pass rate (%)')
    ax.set_xticks([22,25,28]);ax.set_yticks([92,96,100]);ax.xaxis.labelpad=2;ax.yaxis.labelpad=1
    rows=[];labels=[]
    for method,marker in zip(METHODS,fs.MARKERS):
        d=ref.loc[ref.method.eq(method)]
        xx=d.yield_rate.mean()*100;yy=d.pass_rate.mean()*100
        for seed,sm in enumerate(['o','s','^']):
            z=d.loc[d.seed.eq(seed)]
            rows.append(dict(method=method,seed=seed,n_batches=len(z),yield_percent=z.yield_rate.mean()*100,
                pass_percent=z.pass_rate.mean()*100))
        ax.scatter(xx,yy,s=60,marker=marker,c=fs.COLORS[method],ec='white',lw=.4,alpha=.75,zorder=4)
        labels.append((method,xx,yy))
        rows.append(dict(method=method,seed='all',n_batches=len(d),yield_percent=xx,pass_percent=yy))
    place_labels(ax,labels)
    fs.write_table(pd.DataFrame(rows),'main_c_quality_yield.csv')


def distribution(fig,ref,metric,x,y,w,h,letter,title,xlab,limits,ticks,rows):
    fs.panel(fig,x-14,y-11,letter,title)
    ax=fs.axis_mm(fig,x,y,w,h)
    for i,method in enumerate(METHODS):
        d=ref.loc[ref.method.eq(method)]
        a=d[metric].dropna().to_numpy();assert len(a)==3000
        xx=np.linspace(a.min(),a.max(),220);den=gaussian_kde(a)(xx)
        yy=4-i
        ax.fill_between(xx,yy+.01,yy+.01+.51*den/den.max(),fc=fs.fade(fs.COLORS[method]),ec=fs.COLORS[method],lw=.55)
        q=np.quantile(a,[.025,.975]);mean=a.mean()
        ax.plot(q,[yy-.19]*2,c=fs.COLORS[method],lw=1)
        ax.scatter([mean],[yy-.19],s=13,fc='white',ec=fs.COLORS[method],lw=.8,zorder=4)
        seedmeans=[]
        for seed,marker in enumerate(['o','s','^']):
            m=d.loc[d.seed.eq(seed),metric].mean();seedmeans.append(m)
            ax.scatter([m],[yy-.19],s=6,c=fs.COLORS[method],marker=marker,zorder=5)
        rows.append(dict(method=method,metric=metric,n_batches=len(a),mean=mean,batch_q025=q[0],batch_q975=q[1],
            **{f'seed{s}_mean':v for s,v in enumerate(seedmeans)}))
    ax.set(xlim=limits,ylim=(-.55,4.65),xlabel=xlab)
    ax.set_xticks(ticks);ax.set_yticks(range(5),[fs.IDS[m] for m in METHODS[::-1]])
    ax.tick_params(axis='y',length=0,pad=3);ax.spines['left'].set_visible(False);ax.xaxis.labelpad=2


def sorting(fig,ref):
    fs.panel(fig,64,88,'f','Sorting quality and residual risk (%)')
    ax=fs.axis_mm(fig,74,104,103,21)
    ax.set_gid('sorting_heatmap')
    ref=ref.copy()
    assert (ref.n_accepted>0).all() and (ref.n_packs>0).all()
    # Keep the original rejection metrics; expose the retained pool and batch-level outcomes.
    ref['retained_bad_fraction']=ref.n_retained_true_bad/ref.n_accepted
    ref['failed_pack_batch_fraction']=ref.n_failed_packs.gt(0).astype(float)
    metrics=[('rejection_rate','Reject\nrate',(65,85),'#A7BEC9',2),
        ('rejection_precision','Reject\nprecision',(90,100),'#98BDBC',2),
        ('bad_cell_recall','Bad-cell\ndetection',(99,100),'#BAB0CC',2),
        ('good_cell_false_rejection_rate','Good-cell\nfalse reject',(0,25),'#D0B6A1',2),
        ('retained_bad_fraction','Retained\nbad cells',(0,.5),'#C7A2AC',4),
        ('failed_pack_batch_fraction','Failed-pack\nbatches',(0,30),'#C7A2AC',2)]
    rows=[]
    for j,(metric,title,(lo,hi),color,digits) in enumerate(metrics):
        cmap=LinearSegmentedColormap.from_list('sort_'+metric,['#FFFFFF',color])
        fs.colorbar_mm(fig,cmap,Normalize(lo,hi,clip=True),74+j*103/6+1.2,127,103/6-2.4,2,[lo,hi])
        for i,method in enumerate(METHODS):
            z=ref.loc[ref.method.eq(method),metric]
            val=z.mean()*100
            # Preserve the small residual miss rate instead of rounding it to 100%.
            display_digits=4 if metric=='bad_cell_recall' and round(val,2)==100 else digits
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fc=cmap(Normalize(lo,hi,clip=True)(val)),ec='white',lw=.7))
            ax.text(j,i,f'{val:.{display_digits}f}',ha='center',va='center',fontsize=8.5)
            rows.append(dict(method=method,metric=metric,n_batches=len(z),mean_percent=val,
                color_min=lo,color_max=hi,display_decimals=display_digits))
    ax.axvline(3.5,c='white',lw=2)
    ax.set(xlim=(-.5,5.5),ylim=(4.5,-.5))
    ax.set_xticks(range(6),[z[1] for z in metrics]);ax.xaxis.tick_top()
    ax.set_yticks(range(5),[fs.IDS[m] for m in METHODS]);ax.tick_params(length=0,pad=2)
    for spine in ax.spines.values():spine.set_visible(False)
    fs.write_table(pd.DataFrame(rows),'main_f_sorting_quality.csv')
    fs.write_table(ref[['condition','seed','batch_id','method','n_input','n_accepted',
        'n_retained_true_bad','n_packs','n_good_packs','n_failed_packs',
        'retained_bad_fraction','failed_pack_batch_fraction']], 'main_f_residual_risk_records.csv')


def all_settings(fig,raw,paired,include=('g','h','i'),g_labelpad=1):
    selected=paired.loc[paired.comparison.eq('quadrant_c_vs_quadrant_a')]
    rows=[]
    for n,t in CONFIGS:
        z=selected.loc[selected.n_train_total.eq(n)&selected['tail'].eq(t)].set_index('metric')
        rpt=raw.loc[raw.n_train_total.eq(n)&raw['tail'].eq(t)&raw.method.eq('quadrant_c')]
        rows.append(dict(n_train_total=n,tail=t,pool_usage_percent=rpt.pool_usage_ratio.iloc[0]*100,n_batches=len(rpt),
            yield_difference_pp=z.loc['yield_rate_difference_pp','mean'],
            pass_difference_pp=z.loc['pass_rate_difference_pp','mean'],mean_rpt_records=rpt.n_rpt.mean()))
    source=pd.DataFrame(rows)
    panels=[('g','RPT: yield gain (pp)','yield_difference_pp',2,31),
        ('h','RPT: pass change (pp)','pass_difference_pp',56,83)]
    for letter,title,metric,panelx,x in panels:
        if letter not in include:continue
        fs.panel(fig,panelx,141,letter,title)
        panel_width=26 if letter=='g' else 30
        ax=fs.axis_mm(fig,x,151,panel_width,28)
        ax.set_gid('paired_'+letter)
        if metric=='yield_difference_pp':
            cmap=LinearSegmentedColormap.from_list('positive',['#FFFFFF','#BC8D9F']);norm=Normalize(0,5.5)
        else:
            cmap=LinearSegmentedColormap.from_list('signed',['#8EB3C0','#FFFFFF','#BC8D9F']);norm=TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=.4)
        cb=fs.colorbar_mm(fig,cmap,norm,x,190,panel_width,2,
            [0,5.5] if metric=='yield_difference_pp' else [-.4,0,.4])
        cb.set_label('Relative improvement (pp)' if letter=='g' else 'Change (pp)',fontsize=8.5,labelpad=1)
        for ti,t in enumerate([46,50,54]):
            for ni,n in enumerate([400,1000,1600]):
                val=float(source.loc[source.n_train_total.eq(n)&source['tail'].eq(t),metric].iloc[0])
                ax.add_patch(plt.Rectangle((ni-.5,ti-.5),1,1,fc=cmap(norm(val)),ec='white',lw=.7))
                label=f'{val:+.2f}' if metric=='yield_difference_pp' else f'{val:+.3f}'
                ax.text(ni,ti,label,ha='center',va='center',fontsize=8.5,zorder=3)
        ax.set(xlim=(-.5,2.5),ylim=(2.5,-.5),xlabel=fs.POOL_LABEL)
        ax.set_xticks(range(3),['14.3','42.9','71.4'])
        ax.set_yticks(range(3),[fs.TAIL_RANGES[t] for t in [46,50,54]])
        if letter=='g':ax.set_ylabel('Tail segment setting',labelpad=1)
        ax.tick_params(length=0,pad=2);ax.xaxis.labelpad=1;ax.yaxis.labelpad=g_labelpad if letter=='g' else 1
        for spine in ax.spines.values():spine.set_visible(False)
    if 'i' not in include:
        fs.write_table(source,'main_ghi_all_settings.csv')
        return
    fs.panel(fig,118,141,'i','RPT workload')
    ax=fs.axis_mm(fig,142,151,35,28)
    ax.set_gid('rpt_2d')
    cmap=LinearSegmentedColormap.from_list('rpt_gradient',['#D4E6EC','#85B3BF','#817295'])
    norm=Normalize(8,16)
    for row in source.itertuples():
        xx=[400,1000,1600].index(row.n_train_total);yy=[46,50,54].index(row.tail)
        val=row.mean_rpt_records
        # Additive minimum area keeps labels readable; color encodes the exact count.
        area=340+25*(val-8)
        ax.scatter([xx],[yy],s=area,c=[val],cmap=cmap,norm=norm,
            edgecolors='#617787',linewidths=.5,zorder=3)
        ax.text(xx,yy,f'{val:.1f}',ha='center',va='center',fontsize=8.5,zorder=5)
    ax.set(xlim=(-.5,2.5),ylim=(2.5,-.5),xlabel=fs.POOL_LABEL)
    ax.set_xticks(range(3),['14.3','42.9','71.4'])
    ax.set_yticks(range(3),[fs.TAIL_RANGES[t] for t in [46,50,54]])
    ax.tick_params(length=0,pad=2);ax.xaxis.labelpad=1
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fs.colorbar_mm(fig,cmap,norm,142,190,35,2,[8,12,16])
    cb.set_label('Mean RPT\n(records per batch)',fontsize=8.5,labelpad=1)
    fs.write_table(source,'main_ghi_all_settings.csv')

