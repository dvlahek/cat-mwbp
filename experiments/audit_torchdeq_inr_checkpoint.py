#!/usr/bin/env python3
"""Re-audit a frozen controlled TorchDEQ DEQ-INR checkpoint without retraining.

Reconstructs the *same* scikit-image camera and seeded pixel split as training;
reloads the tensor-only checkpoint with weights_only=True; checks provenance,
architecture, strict symmetry, contraction and positive cosine derivative;
creates a side-by-side image panel and performs a dense fixed-budget adjoint
comparison. Each budget's CPU time includes ONE timed terminal residual check
and the check's J^T v action. This is NOT online adaptive stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch

from run_torchdeq_inr_cat import official_model, image_coords, equilibrium, psnr
from run_ignn_ogbn_arxiv import stationary_budget, restarted_gmres_budget, reference_solve


def upstream_git_sha(zoo: Path) -> str:
    try:
        return subprocess.check_output(['git', '-C', str(zoo.resolve()), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def render_image_panel(path: Path, target: np.ndarray, reconstruction: np.ndarray,
                       shape: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    true_im=target.reshape(shape,shape)
    pred_im=reconstruction.reshape(shape,shape)
    diff=np.abs(pred_im-true_im)
    fig, axes = plt.subplots(1,3,figsize=(8.4,2.75), constrained_layout=True)
    for ax,im in zip(axes[:2],(true_im,pred_im)):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
    z=axes[2].imshow(diff,cmap='magma',vmin=0,vmax=max(float(np.quantile(diff,.995)),1e-5),interpolation='nearest')
    axes[2].set_xticks([]);axes[2].set_yticks([])
    for label,ax in zip('ABC',axes):
        ax.text(.02,.98,label,fontsize=11,fontweight='bold',va='top',ha='left',
                transform=ax.transAxes,color='white',bbox=dict(facecolor='black',alpha=.45,edgecolor='none',pad=1.5))
    cb=fig.colorbar(z,ax=axes[2],fraction=.047,pad=.025)
    cb.set_label('Absolute pixel error',fontsize=8)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=None,help='Defaults to sibling manifest.json')
    p.add_argument('--torchdeq-zoo',type=Path,default=Path('external/torchdeq/deq-zoo/deq-inr'))
    p.add_argument('--out',type=Path,default=Path('results/deq_inr_dense_audit'))
    p.add_argument('--start',type=int,default=35)
    p.add_argument('--stop',type=int,default=65)
    p.add_argument('--step',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--audit-pixels',type=int,default=128)
    p.add_argument('--certified-tol',type=float,default=1e-5)
    p.add_argument('--reference-tol',type=float,default=1e-10)
    p.add_argument('--max-reference',type=int,default=1800)
    p.add_argument('--max-forward',type=int,default=450)
    p.add_argument('--gmres-restart',type=int,default=15)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--allow-upstream-mismatch',action='store_true')
    a=p.parse_args()
    if not a.checkpoint.is_file():p.error('Missing checkpoint '+str(a.checkpoint))
    if a.start<1 or a.stop<a.start or a.step<1 or a.repeats<1 or a.audit_pixels<1 or a.threads<1:
        p.error('Invalid budget/range settings')
    if a.manifest is None:a.manifest=a.checkpoint.parent/'manifest.json'
    if not a.manifest.is_file():p.error('Missing original manifest '+str(a.manifest))
    old=json.loads(a.manifest.read_text(encoding='utf-8'))
    if old.get('dataset')!='skimage_camera':p.error('Only the original camera split is supported here')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    for k in ('state_dict','seed','width','rho'):
        if k not in checkpoint:p.error('Checkpoint missing '+k)
    seed=int(checkpoint['seed']); width=int(checkpoint['width']); rho=float(checkpoint['rho'])
    if not (0<rho<1):p.error('Not a contraction')
    config=old['configuration']; side=int(old['image_side'])
    if seed!=int(config['seed']) or width!=int(config['width']) or abs(rho-float(config['rho']))>1e-8:
        p.error('Checkpoint and original experiment manifest disagree')
    current_sha=upstream_git_sha(a.torchdeq_zoo)
    previous_sha=old.get('upstream_torchdeq_git_commit','unavailable')
    if current_sha!='unavailable' and previous_sha!='unavailable' and current_sha!=previous_sha and not a.allow_upstream_mismatch:
        p.error('TorchDEQ checkout differs from original model. Use the original SHA '+previous_sha)
    torch.set_num_threads(a.threads)
    torch.manual_seed(seed);np.random.seed(seed)
    model=official_model(a.torchdeq_zoo,width).double().eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    xy,target,train,holdout,name=image_coords(side,seed,False)
    xy=xy.double();target=target.double()
    W=model.f.layers[0].z_map.weight.detach()
    symmetry=float(torch.max(torch.abs(W-W.T)))
    rowcap=float(W.abs().sum(1).max())
    if symmetry>1e-10 or rowcap>rho+2e-6 or float(W.min())<0:
        raise RuntimeError(f'Spectral interval unsupported symmetry={symmetry} rowcap={rowcap} rho={rho}')
    with torch.no_grad():
        z,nfw,u=equilibrium(model,xy,a.max_forward,1e-10)
        physical=model.f(z,u)
        residual_forward=float(torch.linalg.vector_norm(physical-z)/(torch.linalg.vector_norm(z)+1e-30))
        pred=model.output_map(z)
        train_score=psnr(pred[train],target[train])
        test_score=psnr(pred[holdout],target[holdout])
        prior=old.get('image_psnr',{}).get('heldout_psnr')
        if prior is not None and abs(test_score-float(prior))>0.025:
            raise RuntimeError(f'Loaded checkpoint does not reproduce original PSNR: {test_score} versus {prior}')
        # Save the image without changing grayscale intensity mapping.
        target_im=((target+1)*.5).cpu().numpy()
        reconstructed_im=((pred+1)*.5).cpu().numpy()
        a.out.mkdir(parents=True,exist_ok=True)
        render_image_panel(a.out/'reconstruction',target_im,reconstructed_im,side)
        np.savez_compressed(a.out/'image_arrays.npz',target=target_im.reshape(side,side),
                            reconstruction=reconstructed_im.reshape(side,side),
                            train_pixel_indices=train.cpu().numpy(),heldout_pixel_indices=holdout.cpu().numpy())
        if len(holdout)<a.audit_pixels:p.error('Not enough held-out pixels')
        pick=holdout[:a.audit_pixels]
        x_small=xy[pick]; y_small=target[pick]
        z_small,_,u_small=equilibrium(model,x_small,a.max_forward,a.reference_tol)
        arg=(z_small+torch.sin(u_small[0]))@W.T+u_small[1]
        d=torch.cos(arg)
        min_d=float(d.min())
        if min_d<=0:raise RuntimeError('DEQ-INR cosine derivative not positive')
        if not torch.allclose(model.f(z_small,u_small),torch.sin(arg),atol=1e-11,rtol=1e-11):
            raise RuntimeError('The imported model architecture does not match the audited Jacobian')
        b=(2/len(z_small))*(model.output_map(z_small)-y_small)@model.output_map.weight
        def jtv(v):return (d*v)@W
        ref,ref_iters,ref_bound=reference_solve(b,jtv,rho,a.reference_tol,a.max_reference)
        gradref=d*ref
        grad_norm=float(torch.linalg.vector_norm(gradref))
        bnorm=float(torch.linalg.vector_norm(b))
        rows=[]
        first={}
        for budget in range(a.start,a.stop+1,a.step):
            for method in ('AR','CAT','Chebyshev','GMRES'):
                elapsed=[];checks=[];sol=None;actions=None;bound=None;offline=None
                for repeat in range(a.repeats):
                    t0=time.perf_counter_ns()
                    if method=='GMRES':trial,count=restarted_gmres_budget(b,jtv,budget,a.gmres_restart)
                    else:trial,count=stationary_budget(b,jtv,rho,method,budget)
                    elapsed_solver=(time.perf_counter_ns()-t0)*1e-9
                    t1=time.perf_counter_ns()
                    resid=b-(trial-jtv(trial))
                    rn=float(torch.linalg.vector_norm(resid))
                    absolute=rn/(1-rho)
                    xnorm=float(torch.linalg.vector_norm(trial))
                    check_bound=absolute/(xnorm-absolute) if xnorm>absolute else math.inf
                    elapsed_check=(time.perf_counter_ns()-t1)*1e-9
                    elapsed.append(elapsed_solver); checks.append(elapsed_check)
                    if actions is not None and count!=actions:raise RuntimeError('Non-deterministic action count')
                    sol=trial; actions=count; bound=check_bound
                offline=float(torch.linalg.vector_norm(d*(sol-ref))/(grad_norm+1e-30))
                pass_cert=(bound<a.certified_tol)
                data=dict(method=method,budget=budget,solver_jtv_actions=actions,
                          terminal_check_jtv_actions=1,total_charged_jtv_actions=actions+1,
                          median_solver_seconds=median(elapsed),median_terminal_check_seconds=median(checks),
                          median_charged_seconds=median([s+c for s,c in zip(elapsed,checks)]),
                          certified_relative_adjoint_bound=bound,certified_pass=int(pass_cert),
                          offline_relative_gradient_error=offline,reference_relative_bound=ref_bound)
                rows.append(data)
                if pass_cert and method not in first:first[method]=data
            if budget==a.start or (budget-a.start)%5==0 or budget==a.stop:
                print('DENSE',budget,'first='+str({k:v['budget'] for k,v in first.items()}),flush=True)
                pd.DataFrame(rows).to_csv(a.out/'dense_solver_budgets.csv',index=False)
        pd.DataFrame(rows).to_csv(a.out/'dense_solver_budgets.csv',index=False)
        pd.DataFrame([first[m] for m in ('AR','CAT','Chebyshev','GMRES') if m in first]).to_csv(
            a.out/'dense_first_pass.csv',index=False)
        report=dict(dataset=name,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                    original_checkpoint=str(a.checkpoint),seed=seed,width=width,rho=rho,image_side=side,
                    upstream_torchdeq_sha=current_sha,original_upstream_sha=previous_sha,
                    train_psnr=train_score,heldout_psnr=test_score,
                    baseline_heldout_psnr=old['image_psnr']['heldout_psnr'],
                    train_pixels=len(train),heldout_pixels=len(holdout),audit_pixels=len(pick),
                    layer_min_cosine_derivative=min_d,row_norm_bound=rowcap,symmetry_error=symmetry,
                    reference_relative_bound=ref_bound,reference_iterations=ref_iters,
                    forward_iterations=nfw,forward_residual_relative=residual_forward,
                    first_pass={k:{'budget':v['budget'],'charged_actions':v['total_charged_jtv_actions'],
                                  'median_charged_seconds':v['median_charged_seconds'],
                                  'bound':v['certified_relative_adjoint_bound']} for k,v in first.items()},
                    policy='Frozen model, same seed pixel split, offline reference, fixed budgets, one terminal check per solve',
                    no_autonomous_stopping_claim=True,
                    not_unmodified_torchdeq=True,
                    settings=dict(start=a.start,stop=a.stop,step=a.step,repeats=a.repeats,
                                  certified_tol=a.certified_tol,max_forward=a.max_forward,threads=a.threads))
        (a.out/'dense_audit_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('DONE',a.out.resolve(),flush=True)

if __name__=='__main__':main()