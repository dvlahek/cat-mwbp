#!/usr/bin/env python3
"""Summarize independent spatial denoising audit seeds; no new training."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('results'))
    p.add_argument('--prefix',default='spatial_denoising_s')
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--out',type=Path,default=Path('results/spatial_denoising_summary'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    allfirst=[];allpsnr=[];configs=[]
    for seed in [int(x) for x in a.seeds.split(',')]:
        folder=a.root/f'{a.prefix}{seed}'
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['split_by_source_image']
        assert set(manifest['train_images']).isdisjoint(manifest['test_images'])
        assert set(manifest['validation_images']).isdisjoint(manifest['test_images'])
        assert manifest['solver_audit']['all_methods_certified']
        configs.append(manifest)
        f=pd.read_csv(folder/'first_certified.csv');p0=pd.read_csv(folder/'image_metrics.csv')
        assert set(f['method'])=={'AR','CAT','Chebyshev','GMRES'}
        assert (f['certified_relative_adjoint_bound']<1e-5).all()
        allfirst.append(f)
        allpsnr.append(p0)
    solver=pd.concat(allfirst,ignore_index=True)
    images=pd.concat(allpsnr,ignore_index=True)
    solver.to_csv(a.out/'first_certified_by_seed.csv',index=False)
    images.to_csv(a.out/'image_metrics_by_seed.csv',index=False)
    summary=solver.groupby('method').agg(seeds=('seed','nunique'),
         median_charged_jtv=('charged_jtv','median'),
         min_charged_jtv=('charged_jtv','min'),max_charged_jtv=('charged_jtv','max'),
         median_charged_seconds=('charged_time_s','median')).reset_index()
    summary.to_csv(a.out/'certified_solver_summary.csv',index=False)
    q=solver.pivot(index='seed',columns='method',values='charged_jtv')
    tm=solver.pivot(index='seed',columns='method',values='charged_time_s')
    speedups=pd.DataFrame(dict(seed=q.index,
      AR_over_CAT_actions=q.AR/q.CAT,
      AR_over_CAT_time=tm.AR/tm.CAT))
    speedups.to_csv(a.out/'paired_speedups.csv',index=False)
    f=images[images.split=='test'].groupby(['seed','method'],as_index=False).psnr_db.mean()
    f.to_csv(a.out/'mean_test_psnr.csv',index=False)
    g=f.pivot(index='seed',columns='method',values='psnr_db')
    gauss_best=g[['gaussian_0.7','gaussian_1.2','gaussian_1.8']].max(axis=1)
    model_delta=g['spatial_equilibrium']-gauss_best
    print('ACTION_SPEEDUP_AR_OVER_CAT',speedups.AR_over_CAT_actions.tolist())
    print('MEDIAN_TIME_SPEEDUP_AR_OVER_CAT',float(speedups.AR_over_CAT_time.median()))
    print('MODEL_MINUS_POSTHOC_BEST_GAUSS_DB',model_delta.tolist())
    if len(configs)==3:
        print('SEEDS_COMPLETED',sorted(solver.seed.unique().tolist()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.4))
    methods=['AR','CAT','Chebyshev','GMRES']
    for i,method in enumerate(methods):
        z=solver[solver.method==method].sort_values('seed')
        axes[0].scatter([i]*len(z),z.charged_jtv,s=30,alpha=.75,zorder=3)
        axes[0].plot([i]*len(z),z.charged_jtv,'-',alpha=.35)
    axes[0].set_xticks(range(len(methods)),methods,rotation=12)
    axes[0].set_ylabel(r'Charged $J^\mathsf{T}v$ actions')
    axes[0].grid(axis='y',alpha=.20)
    axes[0].set_ylim(0,180)
    vals=model_delta.values
    axes[1].bar(range(len(vals)),vals)
    axes[1].set_xticks(range(len(vals)),[f'Seed {s}' for s in model_delta.index])
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set_ylabel('PSNR minus best tested Gaussian (dB)')
    axes[1].grid(axis='y',alpha=.20)
    for i,ax in enumerate(axes):
        ax.text(-.11,1.03,'AB'[i],fontweight='bold',transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(a.out/'spatial_denoising_comparison.png',dpi=250,bbox_inches='tight')
    fig.savefig(a.out/'spatial_denoising_comparison.pdf',bbox_inches='tight')
    plt.close(fig)
    (a.out/'summary_manifest.json').write_text(json.dumps(dict(
        seeds=sorted(solver.seed.unique().tolist()),
        ratio_median=float(speedups.AR_over_CAT_time.median()),
        ratio_actions=float(speedups.AR_over_CAT_actions.median()),
        model_minus_best_test_gaussian_by_seed={str(k):float(v) for k,v in model_delta.items()},
        caution='Best Gaussian chosen post hoc on test is a diagnostic, not a fitted evaluation baseline. Solver certificate applies to adjoint, not full output PSNR.',
        test_names=configs[0]['test_images'],
        check_time_policy='One terminal check only; no autonomous stopping claim.'),indent=2))
    print('DONE',a.out)

if __name__=='__main__':main()