#!/usr/bin/env python3
"""Frozen-adjoint audit of an UNPROJECTED upstream TorchDEQ DEQ-INR image layer.

Original upstream DEQINR/MFN/SIRENInjection forward modules are imported as-is.
The deliberately CPU-sized training setup is NOT reproduction of published
DEQ-INR benchmarks. Neither W nor the input injection is symmetrized, clipped,
or constrained by the CAT spectral assumptions. Training uses independent,
per-coordinate direct implicit-function adjoint solves; it does NOT train with
CAT. The frozen same-weights audit measures AR, CAT, scheduled Chebyshev and
GMRES against an exact dense float64 adjoint.

The real-interval CAT/Chebyshev parameterization is a PRE-DECLARED stress test,
not a justified minimax tuning for complex or nonnormal spectra. All calibration
requires external global information. A rigorous finite-dimensional residual
certificate instead uses min singular value of the exact audited I-J^T blocks
and is deliberately flagged as ORACLE, not locally obtainable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def upstream_model(zoo: Path, width: int, input_scale: float):
    root=zoo.resolve()
    if not (root/'modules'/'models.py').is_file():
        raise FileNotFoundError(f'Upstream DEQ-INR sources missing under {root}')
    sys.path.insert(0,str(root))
    from modules.models import DEQINR
    ns=SimpleNamespace(core='sliced', norm_type='none', f_solver='fixed_point_iter',
                       b_solver='fixed_point_iter', f_max_iter=250, b_max_iter=250,
                       f_tol=1e-7, b_tol=1e-7, ift=True, n_states=1, grad=[1])
    return DEQINR(ns, in_channels=2, interm_channels=width,
                  output_channels=1, input_scale=input_scale,
                  filter_type='siren_like', norm_type='none', n_layers=1,
                  one_pass=False)


def official_forward_equilibrium(model, coords, tol, max_iter):
    """Unmodified upstream model.f, iterated WITHOUT gradient history."""
    with torch.no_grad():
        u=model.injection(coords)
        z=torch.zeros((len(coords),model.interm_channels),device=coords.device,
                      dtype=coords.dtype)
        for k in range(1,max_iter+1):
            nxt=model.f(z,u)
            change=float(torch.linalg.vector_norm(nxt-z)/(torch.linalg.vector_norm(nxt)+1e-30))
            if not math.isfinite(change):
                raise RuntimeError('Upstream DEQ-INR forward iteration nonfinite')
            z=nxt
            if change<tol:
                # Explicit final residual, not just consecutive iterate change.
                residual=float(torch.linalg.vector_norm(model.f(z,u)-z)/
                               (torch.linalg.vector_norm(z)+1e-30))
                return z,k,residual
    raise RuntimeError(f'Upstream DEQ-INR forward did not converge in {max_iter} iterations')


def analytic_linearization(model, coords, z):
    """Exact upstream single-SIREN-MFN Jacobian, no operator projection."""
    u=model.injection(coords)
    layer=model.f.layers[0].z_map
    W=layer.weight
    pre=(z+torch.sin(u[0]))@W.T+layer.bias+u[1]
    f= model.f(z,u)
    predicted=torch.sin(pre)
    if not torch.allclose(f,predicted,atol=1e-7,rtol=1e-6):
        raise RuntimeError('Upstream MFN no longer matches audited formula')
    return torch.cos(pre),W


def adjoint_matrices(d,W):
    """Column operator I-J^T, J=diag(cos(pre))*W, per coordinate."""
    width=W.shape[0]
    I=torch.eye(width,dtype=W.dtype,device=W.device)
    return I[None,:,:]-W.T[None,:,:]*d[:,None,:]


def direct_adjoint(b,d,W):
    return torch.linalg.solve(adjoint_matrices(d,W),b.unsqueeze(-1)).squeeze(-1)


def image_data(side,seed):
    from skimage.data import camera
    from skimage.transform import resize
    im=resize(camera().astype(np.float32)/255.,(side,side),
              anti_aliasing=True,preserve_range=True)
    xx,yy=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side),indexing='xy')
    coords=torch.from_numpy(np.stack([xx,yy],-1).reshape(-1,2).astype(np.float32))
    target=torch.from_numpy((2*im.reshape(-1,1)-1).astype(np.float32))
    rg=np.random.default_rng(seed)
    ix=rg.permutation(len(coords))
    ntr=int(.80*len(coords))
    return coords,target,torch.tensor(ix[:ntr],dtype=torch.long),torch.tensor(ix[ntr:],dtype=torch.long)


def psnr(out, tgt):
    return 10*math.log10(4.0/(float(F.mse_loss(out,tgt))+1e-15))


def independent_training_step(model,coords,target,args):
    """Exact-dense IFT for training, to avoid selecting a CAT-favourable state."""
    z,_,_=official_forward_equilibrium(model,coords,args.forward_tol,args.max_forward)
    zleaf=z.detach()
    u=model.injection(coords)
    f=model.f(zleaf,u)
    pred=model.output_map(f)
    loss=F.mse_loss(pred,target)
    b=torch.autograd.grad(loss,f,retain_graph=True)[0].detach()
    with torch.no_grad():
        d,W=analytic_linearization(model,coords,zleaf)
        lam=direct_adjoint(b,d.detach(),W.detach())
        if not torch.isfinite(lam).all():
            raise RuntimeError('Dense IFT adjoint nonfinite')
    f.register_hook(lambda grad:lam)
    loss.backward()
    return float(loss.detach())


def verify_jtv(model,coords,z,d,W):
    zz=z.detach().requires_grad_(True)
    u=tuple(v.detach() for v in model.injection(coords))
    f=model.f(zz,u)
    v=torch.randn_like(f)
    expected=(d*v)@W
    got=torch.autograd.grad(f,zz,grad_outputs=v)[0]
    error=float((torch.linalg.vector_norm(got-expected)/(torch.linalg.vector_norm(got)+1e-30)).detach())
    if error>2e-9:raise RuntimeError(f'Actual TorchDEQ JTv mismatch: {error}')
    return error


def spectrum_audit(A,d,W):
    # J^T = I - A, all finite-dimensional blocks fully enumerated.
    J=torch.eye(W.shape[0],dtype=W.dtype)[None,:,:]-A
    singular_J=torch.linalg.svdvals(J)
    singular_A=torch.linalg.svdvals(A)
    ev=torch.linalg.eigvals(J)
    comm=J@J.transpose(-1,-2)-J.transpose(-1,-2)@J
    normJ=torch.linalg.matrix_norm(J,ord='fro',dim=(-2,-1))
    nonnorm=(torch.linalg.matrix_norm(comm,ord='fro',dim=(-2,-1))/
             normJ.square().clamp_min(1e-30))
    return dict(max_operator_two_norm=float(singular_J[:,0].max()),
      min_A_singular_value=float(singular_A[:,-1].min()),
      max_abs_eigenvalue=float(ev.abs().max()),
      max_abs_imag_eigenvalue=float(ev.imag.abs().max()),
      mean_nonnormal_commutator=float(nonnorm.mean()),
      max_nonnormal_commutator=float(nonnorm.max()),
      complex_eig_block_fraction=float((ev.imag.abs().max(1).values>1e-8).double().mean()))


def coeff_for_real_interval(nominal_rho):
    mu,L=1-nominal_rho,1+nominal_rho
    sm,sL=math.sqrt(mu),math.sqrt(L)
    return 4.0/(sm+sL)**2, ((sL-sm)/(sL+sm))**2


def oracle_complex_calibration(A, seed):
    """Global whole-sample eigenvalue fit for DIAGNOSTIC ONLY, not local."""
    from scipy.optimize import differential_evolution
    E=torch.linalg.eigvals(torch.eye(A.shape[1],dtype=A.dtype)[None,:,:]-A).cpu().numpy().ravel()
    def objective(v):
        alpha,beta=float(v[0]),float(v[1])
        x=(1+beta)-alpha*(1-E)
        roots=np.sqrt(x*x-4*beta)
        return float(np.maximum(np.abs((x+roots)*.5),np.abs((x-roots)*.5)).max())
    fit=differential_evolution(objective,[(0.02,2.0),(0.0,0.98)],
           seed=seed,popsize=12,maxiter=80,tol=1e-8,polish=True)
    return dict(alpha=float(fit.x[0]),beta=float(fit.x[1]),
                predicted_max_root_modulus=float(fit.fun),n_eigenvalues=len(E),
                calibration_is_oracle=True,calibration_includes_audited_samples=True)


def fixed_local(b,jtv,method,budget,rho,oracle=None):
    x=torch.zeros_like(b)
    old=torch.zeros_like(b)
    alpha,beta=coeff_for_real_interval(rho)
    cheb_alpha=1.0
    for k in range(budget):
        r=b-(x-jtv(x))
        if method=='AR':
            new=x+r # alpha=1 for symmetric nominal interval [1-rho,1+rho]
        elif method=='CAT':
            new=x+alpha*r+beta*(x-old)
        elif method=='CAT-complex-oracle':
            if oracle is None:raise ValueError('Missing oracle calibration')
            new=x+oracle['alpha']*r+oracle['beta']*(x-old)
        elif method=='Chebyshev':
            if k==0:new=x+cheb_alpha*r
            else:
                factor=(rho*rho/2) if k==1 else (rho*rho/4)
                nxtalpha=1/(1-factor*cheb_alpha)
                inertia=factor*cheb_alpha*nxtalpha
                new=x+nxtalpha*r+inertia*(x-old)
                cheb_alpha=nxtalpha
        else:raise ValueError(method)
        old,x=x,new
        if not torch.isfinite(x).all():break
    return x,budget


def native_gmres(b,jtv,budget,restart):
    # Existing implementation counts every JTv, including restart probes.
    from run_ignn_ogbn_arxiv import restarted_gmres_budget
    return restarted_gmres_budget(b,jtv,budget,restart)


def evaluate_solvers(model,coords,target,holdout,args,out):
    pts=holdout[:args.audit_pixels]
    xy=coords[pts].double()
    y=target[pts].double()
    model=model.double()
    model.eval()
    with torch.no_grad():
        z,nforward,fwd_res=official_forward_equilibrium(model,xy,args.audit_forward_tol,args.max_forward)
        d,W=analytic_linearization(model,xy,z)
        d,W=d.detach(),W.detach()  # audit never builds autograd graphs for JTv
        A=adjoint_matrices(d,W)
        audit=spectrum_audit(A,d,W)
        prediction=model.output_map(z)
        b=(2.0/len(xy))*(prediction-y)@model.output_map.weight
        exact=direct_adjoint(b,d,W)
    jtv_error=verify_jtv(model,xy,z,d,W)
    oracle=oracle_complex_calibration(A,args.seed) if args.calibrate_complex else None
    # Record both uncontrolled real-interval mismatch and matrix diagnostics.
    def jtv(v):return (v*d)@W
    min_singular=audit['min_A_singular_value']
    if not min_singular>0:raise RuntimeError('Singular linear system on audit subset')
    base=torch.linalg.vector_norm(exact)
    ref_g=d*exact
    rows=[]
    budgets=sorted(set(int(x) for x in args.budgets.split(',')))
    for budget in budgets:
        methods=['AR','CAT','Chebyshev','GMRES']
        if oracle is not None:methods.append('CAT-complex-oracle')
        for method in methods:
            solve_times=[]
            check_times=[]
            vals=None
            for rep in range(args.repeats):
                t=time.perf_counter()
                if method=='GMRES': val,used=native_gmres(b,jtv,budget,args.gmres_restart)
                else:val,used=fixed_local(b,jtv,method,budget,args.nominal_rho,oracle)
                solve_t=time.perf_counter()-t
                t=time.perf_counter()
                residual=b-(val-jtv(val))
                residual_norm=float(torch.linalg.vector_norm(residual))
                # Global oracle sigma_min is NOT free and NOT local. 
                E=residual_norm/min_singular
                estimated=float(torch.linalg.vector_norm(val))
                bound=E/(estimated-E) if estimated>E else math.inf
                check_t=time.perf_counter()-t
                solve_times.append(solve_t)
                check_times.append(check_t)
                vals=(val,used,bound,residual_norm)
            val,used,bound,residual_norm=vals
            gradient=float(torch.linalg.vector_norm(d*(val-exact))/
                           (torch.linalg.vector_norm(ref_g)+1e-30))
            err=float(torch.linalg.vector_norm(val-exact)/(base+1e-30))
            row=dict(seed=args.seed,budget=budget,method=method,
              solver_jtv_actions=used,total_charged_jtv_actions=used+1,
              time_median_s=median(solve_times),check_time_median_s=median(check_times),
              charged_time_median_s=median([a+c for a,c in zip(solve_times,check_times)]),
              adjoint_relative_error=err,gradient_relative_error=gradient,
              oracle_svd_relative_error_bound=bound,
              oracle_certified_pass=int(bound<args.certified_tol),
              relative_residual=residual_norm/(float(torch.linalg.vector_norm(b))+1e-30),
              finite=int(torch.isfinite(val).all()))
            rows.append(row)
            print('AUDIT',args.seed,method,budget,'used',used+1,
                  'grad',f'{gradient:.2e}','oracle_cert',f'{bound:.2e}',flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'solver_budgets.csv',index=False)
    df=pd.DataFrame(rows)
    winners=[]
    for method,g in df.groupby('method'):
        hit=g[g.oracle_certified_pass==1].sort_values('total_charged_jtv_actions')
        if len(hit):
            winner=hit.iloc[0].to_dict()
            winners.append(dict(method=method,first_tested_budget=int(winner['budget']),
                first_charged_jtv=int(winner['total_charged_jtv_actions']),
                first_charged_time_s=winner['charged_time_median_s'],
                oracle_bound=winner['oracle_svd_relative_error_bound']))
        else:winners.append(dict(method=method,first_tested_budget=None,
                first_charged_jtv=None,first_charged_time_s=None,oracle_bound=None))
    pd.DataFrame(winners).to_csv(out/'first_certified.csv',index=False)
    return dict(oracle_complex_calibration=oracle,
                audit_pixel_count=len(xy),forward_iterations=nforward,
                forward_rel_residual=fwd_res,actual_jtv_relative_error=jtv_error,
                reference_exact_dense_solve=True,reference_precision='float64',
                **audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torchdeq-zoo',type=Path,default=Path('external/torchdeq/deq-zoo/deq-inr'))
    p.add_argument('--out',type=Path,default=Path('results/official_deqinr_nonnormal'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--side',type=int,default=48)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--batch-size',type=int,default=96)
    p.add_argument('--input-scale',type=float,default=16.0)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--forward-tol',type=float,default=1e-6)
    p.add_argument('--audit-forward-tol',type=float,default=1e-10)
    p.add_argument('--max-forward',type=int,default=400)
    p.add_argument('--audit-pixels',type=int,default=16)
    p.add_argument('--nominal-rho',type=float,default=.95,
                   help='Predeclared real-interval CAT/Cheb stress coefficients; not an oracle spectral fit')
    p.add_argument('--budgets',default='5,10,15,20,30,40,60,80')
    p.add_argument('--gmres-restart',type=int,default=12)
    p.add_argument('--certified-tol',type=float,default=1e-5)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--calibrate-complex',action='store_true',
                   help='Expensive oracle eigenvalue fit for CAT, separate from predeclared CAT')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        a.side=16;a.width=12;a.steps=4;a.audit_pixels=4;a.batch_size=24
        a.budgets='5,10,15';a.repeats=1
    if (a.width<4 or a.side<8 or not (0<a.nominal_rho<1) or
        a.steps<1 or a.audit_pixels<1 or a.repeats<1):p.error('Invalid configuration')
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model=upstream_model(a.torchdeq_zoo,a.width,a.input_scale)
    coords,target,tr,hold=image_data(a.side,a.seed)
    optim=torch.optim.Adam(model.parameters(),lr=a.lr)
    rng=np.random.default_rng(a.seed)
    history=[]
    for i in range(a.steps):
        ix=tr[torch.as_tensor(rng.choice(len(tr),size=min(a.batch_size,len(tr)),replace=False))]
        model.train()
        optim.zero_grad(set_to_none=True)
        loss=independent_training_step(model,coords[ix],target[ix],a)
        optim.step()
        if i==0 or (i+1)%20==0 or i==a.steps-1:
            with torch.no_grad():
                z,steps,res=official_forward_equilibrium(model,coords,a.forward_tol,a.max_forward)
                pred=model.output_map(z)
                t=dict(step=i+1,loss=loss,train_psnr=psnr(pred[tr],target[tr]),
                       heldout_psnr=psnr(pred[hold],target[hold]),forward_steps=steps,
                       forward_residual=res,weight_two_norm=float(torch.linalg.matrix_norm(
                           model.f.layers[0].z_map.weight,ord=2)))
                history.append(t)
                print('TRAIN',a.seed,i+1, 'PSNR',f'{t["heldout_psnr"]:.3f}',
                      'W2',f'{t["weight_two_norm"]:.3f}',flush=True)
    import pandas as pd
    pd.DataFrame(history).to_csv(a.out/'train_history.csv',index=False)
    audit=evaluate_solvers(model,coords,target,hold,a,a.out)
    ckpt={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
          'seed':a.seed,'width':a.width,'input_scale':a.input_scale}
    torch.save(ckpt,a.out/'trained_model.pt')
    sha=hashlib.sha256((a.out/'trained_model.pt').read_bytes()).hexdigest()
    try:
        repo=a.torchdeq_zoo.resolve().parents[1]
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],
                     text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit='unavailable'
    manifest=dict(dataset='skimage_camera_same_image_pixel_split',
      model='Original upstream TorchDEQ DEQINR/MFN/SIRENInjection n_layers=1; unprojected nonsymmetric weights',
      training='External independent dense exact-IFT adjoint (not CAT or native TorchDEQ training recipe)',
      train_and_holdout_are_same_source_image=True,
      compact_configuration_not_official_published_training=True,
      no_symmetry_or_contraction_projection=True,
      numerical_risk='Real-interval CAT and Chebyshev not guaranteed on complex/non-normal Jacobians',
      solver_certification='Offline global SVD of all audited blocks, not a local online stopping rule',
      upstream_git_commit=commit,torch_version=torch.__version__,python=sys.version,
      checkpoint_sha256=sha,train_pixel_count=len(tr),holdout_pixel_count=len(hold),
      parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
      final_train=history[-1],adjoint_audit=audit,configuration={k:str(v) if isinstance(v,Path) else v
         for k,v in vars(a).items()})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    print('DONE',a.out.resolve(),flush=True)

if __name__=='__main__':main()