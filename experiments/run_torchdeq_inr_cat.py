#!/usr/bin/env python3
"""CAT on an actual TorchDEQ DEQ-INR MFN/SIREN-like image representation layer.

Uses `DEQINR`, `MFN`, and `SIRENInjection` from the *upstream* TorchDEQ zoo.
The original DEQ-INR model's forward architecture is preserved, but the one
recurrent z_map is FROZEN to a symmetric, nonnegative, contractive ring.
The coordinate injection is constrained so cosine derivatives remain positive.
This is therefore a controlled adaptation, NOT unmodified published DEQ-INR.

The trained model is identical for all adjoint methods; all methods receive
one frozen direct-loss gradient and matrix-free J^Tv. Training uses CAT
implicit differentiation. No training steps are differentiated through.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


def official_model(zoo_root: Path, width: int):
    root=zoo_root.resolve()
    if not (root/'modules'/'models.py').is_file():
        raise FileNotFoundError('Missing official TorchDEQ deq-inr modules at '+str(root)+
            '. Clone locuslab/torchdeq and specify --torchdeq-zoo external/torchdeq/deq-zoo/deq-inr')
    sys.path.insert(0,str(root))
    from modules.models import DEQINR
    args=SimpleNamespace(core='sliced',norm_type='none',f_solver='fixed_point_iter',
                         b_solver='fixed_point_iter',f_max_iter=50,b_max_iter=50,
                         f_tol=1e-6,b_tol=1e-6,ift=True,n_states=1,grad=[1])
    model=DEQINR(args,2,width,1,input_scale=1.0,filter_type='siren_like',
                 norm_type='none',n_layers=1,one_pass=False)
    return model


def set_contractive_ring(model,rho):
    layer=model.f.layers[0].z_map
    n=layer.weight.shape[0]
    W=torch.zeros_like(layer.weight)
    for i in range(n):
        W[i,i]=0.6*rho
        W[i,(i-1)%n]=0.2*rho
        W[i,(i+1)%n]=0.2*rho
    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.zero_()
        inj=model.injection.x_map
        torch.nn.init.uniform_(inj.weight,-.015,.015)
        inj.bias.zero_()
    layer.weight.requires_grad_(True)
    layer.bias.requires_grad_(False)
    assert torch.allclose(W,W.T,atol=1e-7)
    assert abs(float(W.abs().sum(1).max())-rho)<1e-5
    return W


@torch.no_grad()
def project_recurrent(model,rho):
    """Project recurrent feature coupling to symmetric nonnegative row-sum cap."""
    W=model.f.layers[0].z_map.weight
    symmetric=torch.clamp((W+W.T)*0.5,min=0.)
    bound=float(symmetric.sum(1).max())
    if bound>rho:symmetric.mul_(rho/bound)
    W.copy_(symmetric)
    model.f.layers[0].z_map.bias.zero_()


@torch.no_grad()
def project_injection(model,cap):
    # For coords in [-1,1]^2, bound every injection output |u_i| by cap.
    inj=model.injection.x_map
    w=inj.weight; b=inj.bias
    total=w.abs().sum(1)+b.abs()
    scale=(cap/total.clamp_min(1e-12)).clamp(max=1.)
    w.mul_(scale[:,None]); b.mul_(scale)


@torch.no_grad()
def equilibrium(model,coords,max_iters,tol):
    u=model.injection(coords)
    z=torch.zeros(coords.shape[0],model.interm_channels,device=coords.device,dtype=coords.dtype)
    for k in range(1,max_iters+1):
        new=model.f(z,u)
        r=float(torch.linalg.vector_norm(new-z)/(torch.linalg.vector_norm(new)+1e-30))
        z=new
        if r<=tol: return z,k,u
    raise RuntimeError(f'DEQ forward failed to converge ({max_iters} iterations)')


def cat_coeff(rho):
    mu=1-rho; L=1+rho
    alpha=4/(math.sqrt(L)+math.sqrt(mu))**2
    beta=((math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu)))**2
    return alpha,beta


@torch.no_grad()
def backward_cat(b,d,W,rho,tol,max_iters):
    a,bt=cat_coeff(rho)
    x=torch.zeros_like(b)
    last=torch.zeros_like(b)
    scale=float(torch.linalg.vector_norm(b))+1e-30
    for i in range(1,max_iters+1):
        r=b-(x-(d*x)@W)
        if float(torch.linalg.vector_norm(r))/scale <=tol:
            return x,i
        nxt=x+a*r+bt*(x-last)
        last,x=x,nxt
    raise RuntimeError('CAT training backward did not reach residual tol')


def training_step(model,xy,target,args):
    W=model.f.layers[0].z_map.weight
    with torch.no_grad():
        z,_,u=equilibrium(model,xy,args.max_forward,args.forward_tol)
        # Exact SIREN derivative on the frozen fixed point, n_layers=1.
        a=(z+torch.sin(u[0]))@W.T+u[1]
        d=torch.cos(a)
        if not torch.allclose(model.f(z,u),torch.sin(a),rtol=1e-5,atol=1e-5):
            raise RuntimeError('Imported DEQ-INR module changed architecture; manual Jacobian invalid')
        if float(d.min())<=0:
            raise RuntimeError('Cosine derivative non-positive: real-interval calibration not certified')
    # One differentiable equilibrium correction. Backward hook replaces the
    # incoming gradient by the implicit adjoint, then ordinary autograd provides
    # the exact injection parameter gradients for upstream DEQ-INR modules.
    zleaf=z.detach().requires_grad_(True)
    u=model.injection(xy)
    f=model.f(zleaf,u)
    def implicit_hook(gradient):
        adjoint,_=backward_cat(gradient.detach(),d.detach(),W.detach(),args.rho,
                               args.backward_tol,args.max_backward)
        return adjoint
    f.register_hook(implicit_hook)
    pred=model.output_map(f)
    loss=F.mse_loss(pred,target)
    return loss


def image_coords(side,seed,synthetic=False):
    from skimage.transform import resize
    if synthetic:
        xx,yy=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side))
        img=np.clip(0.5+0.22*np.sin(3*xx)*np.cos(2*yy)+.10*xx,.0,1.)
        name='synthetic_selftest'
    else:
        from skimage.data import camera
        img=resize(camera().astype(np.float32)/255.,(side,side),
                   anti_aliasing=True,preserve_range=True)
        name='skimage_camera'
    xx,yy=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side), indexing='xy')
    coords=np.stack((xx,yy),axis=-1).reshape(-1,2).astype(np.float32)
    target=(2*img.reshape(-1,1)-1).astype(np.float32)
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(coords))
    ntrain=int(.8*len(coords))
    return (torch.tensor(coords),torch.tensor(target),
            torch.tensor(perm[:ntrain],dtype=torch.long),
            torch.tensor(perm[ntrain:],dtype=torch.long),name)


def psnr(pred,target):
    return 10*math.log10(4/(float(F.mse_loss(pred,target))+1e-15))


def audit(args,model,coords,target,holdout,out):
    from run_ignn_ogbn_arxiv import (stationary_budget,restarted_gmres_budget,reference_solve)
    coords=coords[holdout][:args.audit_pixels].double()
    target=target[holdout][:args.audit_pixels].double()
    model=model.double()
    W=model.f.layers[0].z_map.weight.detach()
    if not torch.allclose(W,W.T,atol=1e-10):raise RuntimeError("Recurrent operator lost symmetry")
    if float(W.abs().sum(1).max())>args.rho+1e-6:raise RuntimeError("Row-sum cap failed")
    with torch.no_grad():
        z,n_forward,u=equilibrium(model,coords,args.max_forward,args.reference_tol)
        a=(z+torch.sin(u[0]))@W.T+u[1]
        d=torch.cos(a)
        if not torch.allclose(model.f(z,u),torch.sin(a),rtol=1e-5,atol=1e-5):
            raise RuntimeError('Imported DEQ-INR module changed architecture; manual Jacobian invalid')
        if float(d.min())<=0:raise RuntimeError('Signed DEQ-INR derivative: cannot apply real interval CAT')
        # Compare actual DEQ-INR layer output with the explicitly differentiated formula.
        assert torch.allclose(model.f(z,u),torch.sin(a),atol=1e-11,rtol=1e-11)
        pred=model.output_map(z)
        b=(2/len(z))*(pred-target)@model.output_map.weight
        def jtv(v): return (d*v)@W
        ref,ref_iters,ref_bound=reference_solve(b,jtv,args.rho,args.reference_tol,args.max_reference)
        eref=d*ref
        norm_eref=float(torch.linalg.vector_norm(eref))
        outrows=[]
        budgets=[int(s) for s in args.budgets.split(',')]
        for budget in budgets:
            for method in ('AR','CAT','Chebyshev','GMRES'):
                t0=time.perf_counter()
                if method=='GMRES':
                    sol,actions=restarted_gmres_budget(b,jtv,budget,args.gmres_restart)
                else:
                    sol,actions=stationary_budget(b,jtv,args.rho,method,budget)
                solver_seconds=time.perf_counter()-t0
                t0=time.perf_counter()
                residual=b-(sol-jtv(sol))
                abs_bound=float(torch.linalg.vector_norm(residual))/(1-args.rho)
                xnorm=float(torch.linalg.vector_norm(sol))
                bound=abs_bound/(xnorm-abs_bound) if xnorm>abs_bound else math.inf
                check_seconds=time.perf_counter()-t0
                error=float(torch.linalg.vector_norm(d*(sol-ref)))/(norm_eref+1e-30)
                outrows.append(dict(method=method,budget=budget,
                   solver_jtv_actions=actions,total_charged_jtv_actions=actions+1,
                   solver_seconds=solver_seconds,check_seconds=check_seconds,
                   charged_seconds=solver_seconds+check_seconds,
                   relative_gradient_error_vs_reference=error,
                   certified_relative_adjoint_bound=bound,
                   certified_pass=int(bound<args.certified_tol),
                   min_cosine_derivative=float(d.min()),
                   reference_relative_bound=ref_bound))
                print('INR',method,'budget',budget,'error',f'{error:.3e}',
                      'cert',f'{bound:.3e}',flush=True)
        import pandas as pd
        pd.DataFrame(outrows).to_csv(out/'solver_budgets.csv',index=False)
    return dict(min_cosine_derivative=float(d.min()),reference_iters=ref_iters,
                reference_relative_bound=ref_bound,forward_iters=n_forward,
                holdout_audit_pixels=len(coords))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torchdeq-zoo',type=Path,default=Path('external/torchdeq/deq-zoo/deq-inr'))
    p.add_argument('--out',type=Path,default=Path('results/deq_inr_cat'))
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--side',type=int,default=64)
    p.add_argument('--width',type=int,default=48)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--rho',type=float,default=.95)
    p.add_argument('--injection-cap',type=float,default=.30)
    p.add_argument('--lr',type=float,default=.004)
    p.add_argument('--forward-tol',type=float,default=1e-7)
    p.add_argument('--reference-tol',type=float,default=1e-10)
    p.add_argument('--max-forward',type=int,default=400)
    p.add_argument('--backward-tol',type=float,default=1e-6)
    p.add_argument('--max-backward',type=int,default=600)
    p.add_argument('--max-reference',type=int,default=1800)
    p.add_argument('--audit-pixels',type=int,default=128)
    p.add_argument('--gmres-restart',type=int,default=15)
    p.add_argument('--budgets',default='10,20,30,40,60,80')
    p.add_argument('--certified-tol',type=float,default=1e-5)
    args=p.parse_args()
    if args.smoke: args.steps=5;args.side=20;args.width=16;args.audit_pixels=24;args.budgets='10,20,30'
    if not (0<args.rho<1) or args.side<8 or args.width<4 or args.audit_pixels<1:p.error('Invalid bounds')
    if args.rho*(1+args.injection_cap)+args.injection_cap>=math.pi/2:
        p.error('Insufficient positivity guarantee: rho*(1+injection_cap)+injection_cap must be < pi/2')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=official_model(args.torchdeq_zoo,args.width)
    W=set_contractive_ring(model,args.rho)
    project_injection(model,args.injection_cap)
    coords,target,itrain,ihold,name=image_coords(args.side,args.seed,args.smoke)
    params=list(model.injection.parameters())+list(model.output_map.parameters())+\
           [model.f.layers[0].z_map.weight]
    optim=torch.optim.Adam(params,lr=args.lr)
    rng=np.random.default_rng(args.seed)
    history=[]
    for step in range(args.steps):
        picks=itrain[torch.as_tensor(rng.choice(len(itrain),size=min(args.batch_size,len(itrain)),replace=False))]
        optim.zero_grad(set_to_none=True)
        loss=training_step(model,coords[picks],target[picks],args)
        loss.backward()
        optim.step()
        project_injection(model,args.injection_cap)
        project_recurrent(model,args.rho)
        if step==0 or (step+1)%10==0 or step==args.steps-1:
            with torch.no_grad():
                z,_,_=equilibrium(model,coords,args.max_forward,args.forward_tol)
                pred=model.output_map(z)
                psnr_train=psnr(pred[itrain],target[itrain])
                psnr_holdout=psnr(pred[ihold],target[ihold])
            history.append(dict(step=step+1,loss=float(loss.detach()),train_psnr=psnr_train,heldout_psnr=psnr_holdout))
            print('TRAIN',step+1,'PSNR',f'{psnr_train:.2f}',f'{psnr_holdout:.2f}',flush=True)
    import pandas as pd
    pd.DataFrame(history).to_csv(args.out/'train_history.csv',index=False)
    with torch.no_grad():
        zz,_,_=equilibrium(model,coords,args.max_forward,args.forward_tol)
        pred=model.output_map(zz)
        scores=dict(train_psnr=psnr(pred[itrain],target[itrain]),
                    heldout_psnr=psnr(pred[ihold],target[ihold]))
    audit_data=audit(args,model,coords,target,ihold,args.out)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'seed':args.seed,'width':args.width,'rho':args.rho},args.out/'trained_model.pt')
    try:
        upstream_sha=subprocess.check_output(['git','-C',str(args.torchdeq_zoo.resolve()),
            'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        try:
            upstream_sha=subprocess.check_output(['git','-C',str(args.torchdeq_zoo.resolve().parents[2]),
                'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            upstream_sha='unavailable'
    provenance=dict(dataset=name,architecture='Official TorchDEQ DEQINR/MFN n_layers=1, SIREN-like filter',
       upstream_torchdeq_git_commit=upstream_sha,
       modification='Learned recurrent weights projected to symmetric nonnegative contraction; bounded coordinate injection',
       no_unmodified_deq_claim=True,no_general_nonnormal_claim=True,
       image_side=args.side,train_pixels=len(itrain),heldout_pixels=len(ihold),
       learned_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
       image_psnr=scores,audit=audit_data,torch_version=torch.__version__,
       torchdeq_zoo=str(args.torchdeq_zoo),configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.out/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print('DONE',args.out.resolve(),flush=True)

if __name__=='__main__':main()