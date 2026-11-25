#!/usr/bin/env python3
"""Per-iteration OFFLINE certification of frozen CAT/AR/Chebyshev adjoints.

Reconstructs the same frozen model and right-hand side as the original graph
postcheck or spatial solver_audit. Unlike the manuscript's coarse budget grid,
checks every iterate. The original fixed-budget protocol remains unchanged.
Every reported first passing iterate k gets an equivalent +1 terminal-JTv
charge, not a claim of continuous autonomous online stopping.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

METHODS=('AR','CAT','Chebyshev')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def iterated_certificates(b,jtv,rho,max_budget,certified_tol,reference=None,weight=None):
    if not 0<rho<1:raise ValueError('rho must be in (0,1)')
    if not 1<=max_budget<=100000:raise ValueError('Invalid max_budget')
    mu,L=1-rho,1+rho
    center,half=(L+mu)/2,(L-mu)/2
    q=(math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu))
    ca,cb=4/(math.sqrt(L)+math.sqrt(mu))**2,q*q
    ar=2/(mu+L)
    bn=torch.linalg.vector_norm(b)
    refnorm=torch.linalg.vector_norm(reference) if reference is not None else None
    wrefnorm=torch.linalg.vector_norm(weight*reference) if weight is not None and reference is not None else None
    rows=[]
    with torch.no_grad():
        for method in METHODS:
            x=torch.zeros_like(b)
            old=torch.zeros_like(b)
            residual=b.clone() # at x0, J^T(0)=0
            cheb_a=1/center
            for k in range(1,max_budget+1):
                if method=='AR':
                    nxt=x+ar*residual
                elif method=='CAT':
                    nxt=x+ca*residual+cb*(x-old)
                else:
                    if k==1:
                        nxt=x+cheb_a*residual
                    else:
                        factor=half*half/(2 if k==2 else 4)
                        na=1/(center-factor*cheb_a)
                        inertia=factor*cheb_a*na
                        nxt=x+na*residual+inertia*(x-old)
                        cheb_a=na
                old,x=x,nxt
                residual=b-(x-jtv(x))
                rn=torch.linalg.vector_norm(residual)
                E=rn/(1-rho)
                nx=torch.linalg.vector_norm(x)
                bound=float(E/(nx-E)) if bool(nx>E) else math.inf
                row=dict(method=method,budget=k,charged_jtv=k+1,
                         certified_relative_adjoint_bound=bound,
                         certified_pass=int(bound<certified_tol),
                         relative_residual=float(rn/(bn+1e-30)))
                if reference is not None:
                    row['offline_relative_adjoint_error']=float(torch.linalg.vector_norm(x-reference)/(refnorm+1e-30))
                    if weight is not None:
                        row['offline_relative_gradient_error']=float(torch.linalg.vector_norm(weight*(x-reference))/(wrefnorm+1e-30))
                rows.append(row)
    return pd.DataFrame(rows)

def graph_setup(args,checkpoint,device):
    from run_ignn_ogbn_arxiv import (
        GraphImplicitClassifier,accuracy,equilibrium,prepare_data,
        reference_solve,spmm)
    import torch.nn.functional as F
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ('state_dict','width','seed','rho','num_features','num_classes'):
        if key not in ck:raise ValueError(f'Graph checkpoint missing {key}')
    args.seed=int(ck['seed']);args.width=int(ck['width']);args.rho=float(ck['rho'])
    args.smoke=False # Never silently replace the real graph with synthetic data
    x,y,S,split,nclasses,graph_meta,S64=prepare_data(args,device)
    if int(x.shape[1])!=int(ck['num_features']) or nclasses!=int(ck['num_classes']):
        raise ValueError('Graph checkpoint and OGB data disagree')
    model=GraphImplicitClassifier(int(x.shape[1]),args.width,nclasses,args.seed).to(device)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    with torch.no_grad():
        h,nforward,_=equilibrium(model.drive(x),S,args.rho,args.eval_forward_tol,args.max_forward)
        logits=model.output(h)
        scores={key:accuracy(logits,y,index) for key,index in split.items()}
        p=F.softmax(logits[split['train']],dim=1)
        p[torch.arange(len(split['train']),device=device),y[split['train']]]-=1
        p/=len(split['train'])
        b=torch.zeros_like(h)
        b[split['train']]=p@model.output.weight
        h,b=h.double(),b.double()
        S64=S64.to(dtype=torch.float64,device=device)
        d=1-h*h
        jtv=lambda v:args.rho*spmm(S64,d*v)
        ref,nref,refbound=reference_solve(b,jtv,args.rho,args.reference_tol,args.max_reference)
    meta=dict(kind='graph',dataset='ogbn-arxiv',seed=args.seed,width=args.width,rho=args.rho,
              graph_meta=graph_meta,scores=scores,forward_iterations=nforward,
              reference_iterations=nref,reference_bound=refbound)
    return b,jtv,d,ref,meta

def spatial_setup(args,checkpoint,device):
    from run_spatial_implicit_denoising import (
        SpatialEquilibrium,TEST_NAMES,load_image,prepare,stencil)
    from run_ignn_ogbn_arxiv import reference_solve
    import torch.nn.functional as F
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ('state_dict','width','seed','rho','center_weight'):
        if key not in ck:raise ValueError(f'Spatial checkpoint missing {key}')
    args.seed=int(ck['seed']);args.width=int(ck['width']);args.rho=float(ck['rho'])
    args.center_weight=float(ck['center_weight'])
    if device.type!='cpu':raise ValueError('Spatial reference protocol requires --device cpu')
    torch.set_num_threads(args.threads)
    name=TEST_NAMES[0]
    rng=np.random.default_rng(500000+args.seed)
    clean=load_image(name,args.audit_side)
    feat,base,target,_=prepare(clean,args.sigma,rng,torch.float64)
    model=SpatialEquilibrium(args.width,args.rho,args.center_weight).double().to(device).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    with torch.no_grad():
        h,nforward,_=model.fix(feat,args.max_forward,args.reference_tol*.2)
        d=1-h*h
    with torch.enable_grad():
        hleaf=h.detach().requires_grad_(True)
        loss=F.mse_loss(model.prediction(hleaf,base),target)
        b=torch.autograd.grad(loss,hleaf)[0].detach()
    with torch.no_grad():
        jtv=lambda v:model.rho*stencil(d*v,model.center)
        ref,nref,refbound=reference_solve(b,jtv,model.rho,args.reference_tol,args.max_reference)
    meta=dict(kind='spatial',dataset='skimage.data held-out denoising',seed=args.seed,
              image=name,width=args.width,rho=args.rho,center_weight=args.center_weight,
              audit_side=args.audit_side,sigma=args.sigma,forward_iterations=nforward,
              reference_iterations=nref,reference_bound=refbound)
    return b,jtv,d,ref,meta

def validate_old_grid(fine,path):
    old=pd.read_csv(path)
    required={'method','budget','certified_pass'}
    if not required.issubset(old):raise ValueError(f'Old CSV missing {required-set(old)}')
    old=old[old.method.isin(METHODS)]
    merged=old.merge(fine[['method','budget','certified_pass']],on=['method','budget'],
                     how='left',suffixes=('_old','_fine'),validate='one_to_one')
    if merged.certified_pass_fine.isna().any():
        raise AssertionError('Fine run did not reach every old budget')
    mismatch=merged[merged.certified_pass_old.astype(int)!=merged.certified_pass_fine.astype(int)]
    if len(mismatch):
        raise AssertionError(f'Original grid flags disagreed with diagnostic: {mismatch[["method","budget"]].to_dict("records")}')
    return dict(original_grid_rows_checked=len(merged),all_pass_flags_match=True)

def selftest():
    # Compare to the independent stationary_budget implementation when running
    # inside the repository (or check internal certificate consistency).
    try:
        from run_ignn_ogbn_arxiv import stationary_budget
    except ImportError:
        stationary_budget=None
    torch.manual_seed(71)
    rho=.85
    vals=torch.tensor([.75,.1,-.2,.35,.0],dtype=torch.float64)
    b=torch.tensor([1.,-.7,.3,-1.2,.8],dtype=torch.float64)
    jtv=lambda v:vals*v
    fine=iterated_certificates(b,jtv,rho,40,1e-5)
    assert len(fine)==3*40 and (fine.charged_jtv==fine.budget+1).all()
    if stationary_budget is not None:
        for method in METHODS:
            for budget in (1,2,3,10,40):
                x,_=stationary_budget(b,jtv,rho,method,budget)
                r=b-(x-jtv(x))
                E=torch.linalg.vector_norm(r)/(1-rho)
                nx=torch.linalg.vector_norm(x)
                bound=float(E/(nx-E)) if bool(nx>E) else math.inf
                row=fine[(fine.method==method)&(fine.budget==budget)].iloc[0]
                ratio=float(torch.linalg.vector_norm(r)/torch.linalg.vector_norm(b))
                if abs(float(row.relative_residual)-ratio)>2e-12:
                    raise AssertionError((method,budget,'residual'))
                if math.isfinite(bound) and not math.isclose(float(row.certified_relative_adjoint_bound),bound,rel_tol=1e-10,abs_tol=1e-12):
                    raise AssertionError((method,budget,'certificate'))
    print('FINE_GRID_TRAJECTORY_SELFTEST_PASS',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=('graph','spatial'))
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--out',type=Path,default=Path('results/fine_grid_certification'))
    ap.add_argument('--old-grid',type=Path,help='Optional original solver_budgets CSV')
    ap.add_argument('--max-budget',type=int,default=200)
    ap.add_argument('--certified-tol',type=float,default=1e-5)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--data-root',default='data/ogb')
    ap.add_argument('--sparse-format',choices=('csr','coo'),default='csr')
    ap.add_argument('--trust-ogb-cache',action='store_true')
    ap.add_argument('--eval-forward-tol',type=float,default=2e-7)
    ap.add_argument('--max-forward',type=int,default=1000)
    ap.add_argument('--reference-tol',type=float,default=2e-8)
    ap.add_argument('--max-reference',type=int,default=2500)
    ap.add_argument('--audit-side',type=int,default=48)
    ap.add_argument('--sigma',type=float,default=.12)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    selftest()
    if args.selftest:return
    if not args.kind or not args.checkpoint:
        ap.error('--kind and --checkpoint are required unless --selftest')
    if not args.checkpoint.is_file():ap.error(f'Checkpoint missing: {args.checkpoint}')
    if args.kind=='graph':
        args.max_forward=700 if args.max_forward==1000 else args.max_forward
    else:
        args.reference_tol=2e-9 if args.reference_tol==2e-8 else args.reference_tol
        args.max_reference=2000 if args.max_reference==2500 else args.max_reference
    device=torch.device(args.device)
    b,jtv,weight,ref,meta=(graph_setup(args,args.checkpoint,device)
                          if args.kind=='graph' else spatial_setup(args,args.checkpoint,device))
    fine=iterated_certificates(b,jtv,args.rho,args.max_budget,args.certified_tol,ref,weight)
    fine.insert(0,'seed',args.seed)
    first=(fine[fine.certified_pass==1].sort_values('budget')
           .groupby('method',as_index=False).first())
    if set(first.method)!=set(METHODS):
        raise AssertionError('Some methods not certified by max-budget; increase --max-budget')
    summary=first[['seed','method','budget','charged_jtv','certified_relative_adjoint_bound',
                   'offline_relative_adjoint_error','offline_relative_gradient_error']].copy()
    summary.rename(columns={'budget':'first_certified_iteration',
                            'charged_jtv':'first_certified_charged_actions'},inplace=True)
    check=validate_old_grid(fine,args.old_grid) if args.old_grid else None
    args.out.mkdir(parents=True,exist_ok=True)
    fine.to_csv(args.out/'fine_grid_all_steps.csv',index=False)
    summary.to_csv(args.out/'fine_grid_first_pass.csv',index=False)
    manifest=dict(checkpoint_sha256=sha256(args.checkpoint),
                  method='per-iteration offline diagnostic on frozen checkpoint',
                  measure='first relative-adjoint certificate with one conceptual terminal JTv',
                  not_claimed='continuously monitored online stopping or hardware speedup',
                  source_metadata=meta,old_grid_validation=check,
                  configuration={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()})
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    p=summary.set_index('method')
    ar=int(p.loc['AR','first_certified_charged_actions'])
    cat=int(p.loc['CAT','first_certified_charged_actions'])
    cheb=int(p.loc['Chebyshev','first_certified_charged_actions'])
    print('FINE_GRID_CERTIFICATION_PASS',args.kind,'seed',args.seed,'AR',ar,
          'CAT',cat,'Chebyshev',cheb,'AR/CAT',round(ar/cat,6),
          'Chebyshev/CAT',round(cheb/cat,6),flush=True)
    print('OUTPUT',args.out.resolve(),flush=True)

if __name__=='__main__':main()
