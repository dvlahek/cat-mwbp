#!/usr/bin/env python3
"""Figure 3: absolute AR/CAT backward actions and matched held-out accuracy.

Read original width_pairs.csv; never reconstruct actions from rounded ratios.
Chebyshev is omitted because it was not part of this width-scaling panel.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('results/revision_width/width_pairs.csv'))
    ap.add_argument('--output',type=Path,default=Path('results/revision_width/cat_parameter_scaling_absolute'))
    args=ap.parse_args()
    if not args.input.is_file():
        ap.error(f'Missing input CSV: {args.input}')
    p=pd.read_csv(args.input)
    needed={'dataset','trainable_parameters','seed','ar_backward_actions','cat_backward_actions',
            'ar_test_accuracy','cat_test_accuracy'}
    if not needed.issubset(p.columns):
        ap.error(f'Missing original experiment columns: {sorted(needed-set(p.columns))}')
    if p.duplicated(['dataset','trainable_parameters','seed']).any():
        ap.error('Duplicate dataset/parameter-count/seed runs')
    for field in ('ar_backward_actions','cat_backward_actions'):
        if (p[field]<=0).any() or not np.isfinite(p[field]).all():
            ap.error(f'Invalid {field} data')
    if len(p)!=45:
        print(f'NOTE: {len(p)} matched pairs, rather than the 45 used in the manuscript')
    if (p['ar_backward_actions']<=p['cat_backward_actions']).any():
        print('NOTE: at least one AR/CAT pair does not show an action advantage')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.2,
                         'axes.labelsize':10,'legend.fontsize':8.6,
                         'xtick.labelsize':8.8,'ytick.labelsize':8.8,
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.8))
    series=(('breast_cancer','Breast cancer','o'),
            ('digits','Digits','s'),
            ('synthetic_large','Synthetic large','^'))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for (ds,label,marker),color in zip(series,colors):
        group=p[p.dataset==ds].groupby('trainable_parameters',as_index=False).agg(
            ar=('ar_backward_actions','mean'),cat=('cat_backward_actions','mean'),
            ar_acc=('ar_test_accuracy','mean'),cat_acc=('cat_test_accuracy','mean'))
        if group.empty:continue
        group=group.sort_values('trainable_parameters')
        axes[0].plot(group.trainable_parameters,group.ar,marker=marker,
                     ls='--',lw=1.5,markersize=4.5,color=color,label=f'{label}: AR')
        axes[0].plot(group.trainable_parameters,group.cat,marker=marker,
                     ls='-',lw=1.5,markersize=4.5,color=color,label=f'{label}: CAT')
        axes[1].plot(group.trainable_parameters,group.cat_acc,marker=marker,
                     ls='-',lw=1.6,markersize=4.5,color=color,label=label)
    axes[0].set_ylabel(r'Mean local $J^T v$ actions per backward solve')
    axes[1].set_ylabel('Held-out accuracy (CAT)')
    for ax,letter in zip(axes,'AB'):
        ax.set_xscale('log')
        ax.set_xlabel('Trainable parameters')
        ax.grid(alpha=.2,linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(-.10,1.04,letter,transform=ax.transAxes,fontweight='bold')
    axes[0].legend(loc='upper center',bbox_to_anchor=(0.5,-0.23),frameon=False,ncol=2)
    axes[1].legend(loc='best',frameon=False)
    fig.subplots_adjust(left=.10,right=.985,top=.91,bottom=.34,wspace=.28)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.07)
    fig.savefig(args.output.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    print('WIDTH_ABSOLUTE_FIGURE_PASS',args.output)

if __name__=='__main__':main()
