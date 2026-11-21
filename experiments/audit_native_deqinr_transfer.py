#!/usr/bin/env python3
"""Out-of-calibration-sample CAT spectral transfer on frozen native TorchDEQ checkpoints.

The original unprojected DEQ-INR/MFN/SIREN-like layer was trained INDEPENDENTLY
of CAT in experiments/run_official_deqinr_nonnormal.py. This script never trains
or alters checkpoints. It fits CAT (alpha,beta) only on 64 TRAINING pixel
Jacobians and then freezes it before opening a disjoint 64-pixel HELD-OUT
Jacobian set. The first 16 holdout pixels were previously inspected by the
exploratory in-sample oracle; they are explicitly excluded here.

The primary test transfers to unseen pixel Jacobians of the SAME trained
model. A secondary leave-one-model-out test transfers coefficients from the
OTHER TWO independently trained model checkpoints, without accessing the
held-out model Jacobian for calibration. Neither is a locally computable
calibration or evidence for unseen image datasets:
calibration still needs GLOBAL eigenvalues of training Jacobians. Test SVD and
test eigenvalues are computed strictly AFTER coefficients have been frozen,
solely for OFFLINE certification and diagnostic reporting.

This single-layer forward formula is taken directly from upstream TorchDEQ
DEQINR(n_layers=1, filter_type='siren_like') and is verified numerically
against the checkpoint's previously published audit diagnostics. Optionally
import and execute original TorchDEQ modules for a second forward/JTv check.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution
from skimage.data import camera
from skimage.transform import resize


def parse_arguments():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--root',type=Path,help='parent of official_deqinr_native_s0/s1/s2 (usually results)')
    g.add_argument('--archive',type=Path,help='ZIP containing the original nativeall experiment folders')
    p.add_argument('--prefix',default='official_deqinr_native_s')
    p.add_argument('--out',type=Path,default=Path('results/native_deqinr_spectral_transfer'))
    p.add_argument('--torchdeq-zoo',type=Path,default=None,help='optional path to official deq-zoo/deq-inr; checks upstream model.f against independent formula')
    p.add_argument('--calibration-count',type=int,default=64)
    p.add_argument('--test-count',type=int,default=64)
    p.add_argument('--exclude-prior-audit',type=int,default=16)
    p.add_argument('--random-seed-base',type=int,default=20261121)
    p.add_argument('--max-budget',type=int,default=120)
    p.add_argument('--certified-tol',type=float,default=1e-5)
    p.add_argument('--max-forward',type=int,default=1000)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.calibration_count<4 or a.test_count<4 or a.max_budget<5 or a.threads<1:p.error('Invalid size/budget')
    if a.smoke:
        a.calibration_count=8;a.test_count=8;a.max_budget=30
    return a


def get_source_bytes(a, seed, leaf):
    path=f'{a.prefix}{seed}/{leaf}'
    if a.archive is not None:
        with zipfile.ZipFile(a.archive) as z:
            return z.read(path)
    return (a.root/path).read_bytes()


def image_split(seed,side):
    img=resize(camera().astype(np.float32)/255.,(side,side),anti_aliasing=True,preserve_range=True)
    xx,yy=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side),indexing='xy')
    # EXACT original float32 pixel coordinates, promoted for frozen float64 audit.
    xy=np.stack((xx,yy),axis=-1).reshape(-1,2).astype(np.float32)
    targets=(2*img.reshape(-1,1)-1).astype(np.float32)
    indices=np.random.default_rng(seed).permutation(len(xy))
    ntrain=int(.8*len(indices))
    return torch.from_numpy(xy).double(),torch.from_numpy(targets).double(),indices[:ntrain],indices[ntrain:]


def load_state(a,seed):
    b=get_source_bytes(a,seed,'trained_model.pt')
    manifest=json.loads(get_source_bytes(a,seed,'manifest.json'))
    actual_sha=hashlib.sha256(b).hexdigest()
    if actual_sha != manifest['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint hash changed for seed {seed}')
    ckpt=torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
    if int(ckpt['seed'])!=seed:raise RuntimeError('Seed mismatch in checkpoint')
    state={k:v.detach().double() for k,v in ckpt['state_dict'].items()}
    return state,manifest,actual_sha


@torch.no_grad()
def matrices(state,xy,y,max_forward,forward_tol=1e-12):
    # Equivalent of DEQINR/MFN single-layer SIRENFilter without projection.
    W=state['f.layers.0.z_map.weight']
    bias=state['f.layers.0.z_map.bias']
    encoded=xy@state['injection.x_map.weight'].T + state['injection.x_map.bias']
    u0,u1=encoded.chunk(2,-1)
    z=torch.zeros((len(xy),W.shape[0]),dtype=torch.float64)
    for i in range(max_forward):
        nxt=torch.sin((z+torch.sin(u0))@W.T+bias+u1)
        error=float(torch.linalg.vector_norm(nxt-z)/(torch.linalg.vector_norm(nxt)+1e-30))
        z=nxt
        if error<forward_tol:break
    else:raise RuntimeError(f'Forward equilibrium did not converge in {max_forward} iterations')
    pre=(z+torch.sin(u0))@W.T+bias+u1
    d=torch.cos(pre)
    A=torch.eye(W.shape[0],dtype=torch.float64)[None]-W.T[None]*d[:,None,:]
    predictions=z@state['output_map.weight'].T+state['output_map.bias']
    b=2*(predictions-y)@state['output_map.weight']/len(xy)
    true_x=torch.linalg.solve(A,b.unsqueeze(-1)).squeeze(-1)
    return dict(W=W,z=z,d=d,A=A,b=b,exact=true_x,
                equilibrium_iterations=i+1,equilibrium_change=error)


def verify_previous_audit(state,manifest,xy,y,max_forward):
    obj=matrices(state,xy,y,max_forward)
    I=torch.eye(obj['W'].shape[0],dtype=torch.float64)
    J=I[None]-obj['A']
    norm=float(torch.linalg.matrix_norm(J,ord=2,dim=(-2,-1)).max())
    previous=manifest['adjoint_audit']['max_operator_two_norm']
    if abs(norm-previous)>2e-6:
        raise RuntimeError(f'Prior native TorchDEQ audit not reproduced: {norm} vs {previous}')
    return norm


def verify_upstream_model(a,state,xy,computed):
    # Optional independent check on installed original TorchDEQ modules.
    if a.torchdeq_zoo is None:return 'not requested'
    root=Path(__file__).resolve().parent
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    # Imports same upstream_model construction used for original unprojected training.
    from run_official_deqinr_nonnormal import upstream_model
    width=state['f.layers.0.z_map.weight'].shape[0]
    model=upstream_model(a.torchdeq_zoo,width,16.0).double().eval()
    model.load_state_dict(state,strict=True)
    with torch.no_grad():
        u=model.injection(xy)
        actual=model.f(computed['z'],u)
        formula=torch.sin((computed['z']+torch.sin(u[0]))@computed['W'].T+
                          state['f.layers.0.z_map.bias']+u[1])
        err=float(torch.linalg.vector_norm(actual-formula)/(torch.linalg.vector_norm(formula)+1e-30))
        if err>2e-12:raise RuntimeError(f'Upstream forward mismatch: {err}')
    zz=computed['z'].detach().requires_grad_(True)
    f=model.f(zz,model.injection(xy))
    grad=torch.ones_like(f)
    got=torch.autograd.grad(f,zz,grad_outputs=grad)[0]
    expected=(computed['d']*grad)@computed['W']
    rel=float(torch.linalg.vector_norm(got-expected)/(torch.linalg.vector_norm(got)+1e-30))
    if rel>2e-11:raise RuntimeError(f'Upstream JTv mismatch: {rel}')
    return dict(formula_rel_error=err,jtv_rel_error=rel)


def worst_characteristic_root(eigJ,alpha,beta):
    t=1+beta-alpha*(1-eigJ)
    sq=np.sqrt(t*t-4*beta+0j)
    return float(max(np.abs((t+sq)*.5).max(),np.abs((t-sq)*.5).max()))


def calibration_from_training(A,seed):
    eigJ=np.linalg.eigvals(np.eye(A.shape[1])-A.numpy()).ravel()
    fit=differential_evolution(lambda v:worst_characteristic_root(eigJ,v[0],v[1]),
                               bounds=[(.02,2.0),(0.0,.98)],seed=seed,
                               popsize=12,maxiter=80,tol=1e-8,polish=True,workers=1)
    alpha,beta=map(float,fit.x)
    return dict(alpha=alpha,beta=beta,calibration_modal_factor=float(fit.fun),
                training_eigenvalues=int(eigJ.size))


def nominal_cat_coefficients():
    mu,L=0.05,1.95
    alpha=4/(math.sqrt(mu)+math.sqrt(L))**2
    beta=((math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu)))**2
    return alpha,beta


def evaluate_method(obj,mu,method,alpha,beta,max_budget,tol):
    b,d,W,exact=obj['b'],obj['d'],obj['W'],obj['exact']
    if method=='AR':alpha,beta=1.0,0.0
    if method=='CAT-nominal':alpha,beta=nominal_cat_coefficients()
    x=torch.zeros_like(b)
    old=torch.zeros_like(x)
    first=None
    rows=[]
    for budget in range(1,max_budget+1):
        r=b-(x-(x*d)@W)
        nxt=x+alpha*r+beta*(x-old)
        old,x=x,nxt
        # This is an *offline* diagnostic at every budget. Online norm checks
        # at each iteration would incur additional global communication/JTv.
        residual=b-(x-(x*d)@W)
        upper=float(torch.linalg.vector_norm(residual))/mu
        xnorm=float(torch.linalg.vector_norm(x))
        cert=upper/(xnorm-upper) if xnorm>upper else math.inf
        relative_actual=float(torch.linalg.vector_norm(x-exact)/(torch.linalg.vector_norm(exact)+1e-30))
        if not math.isfinite(cert):cert=math.inf
        if first is None and cert<tol:first=budget
        rows.append(dict(method=method,budget=budget,terminal_charged_actions=budget+1,
                         offline_svd_relative_bound=cert,actual_relative_adjoint_error=relative_actual))
    return first,rows


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    rows=[];curves=[];provenance=[]
    cached={}
    # Phase I: load frozen models, select disjoint indices and construct ONLY
    # training-derived calibration spectra. No new heldout test Jacobian yet.
    for seed in (0,1,2):
        state,manifest,sha=load_state(a,seed)
        side=int(manifest['configuration']['side'])
        xy,y,train,hold=image_split(seed,side)
        if len(hold)<a.exclude_prior_audit+a.test_count:
            raise RuntimeError('Not enough previously unexamined held-out pixels')
        if len(train)<a.calibration_count:raise RuntimeError('Not enough train pixels')
        old_norm=verify_previous_audit(state,manifest,xy[hold[:a.exclude_prior_audit]],
                                       y[hold[:a.exclude_prior_audit]],a.max_forward)
        rng=np.random.default_rng(a.random_seed_base+seed)
        cal=rng.choice(train,size=a.calibration_count,replace=False)
        test=hold[a.exclude_prior_audit:a.exclude_prior_audit+a.test_count]
        if np.intersect1d(cal,test).size:raise RuntimeError('Calibration/test pixel leakage')
        cal_obj=matrices(state,xy[cal],y[cal],a.max_forward)
        upstream_check=verify_upstream_model(a,state,xy[cal],cal_obj)
        cached[seed]=dict(state=state,manifest=manifest,sha=sha,xy=xy,y=y,
                          cal=cal,test=test,cal_obj=cal_obj,old_norm=old_norm,
                          upstream_check=upstream_check)
    # Phase II: freeze BOTH parameter-selection protocols before we even
    # construct the 64-pixel target Jacobians.
    fits={}
    for seed in (0,1,2):
        own_A=cached[seed]['cal_obj']['A']
        other_A=torch.cat([cached[k]['cal_obj']['A'] for k in (0,1,2) if k!=seed],dim=0)
        fits[seed]={
            'CAT-train-pixels':calibration_from_training(own_A,a.random_seed_base+seed),
            'CAT-other-models':calibration_from_training(other_A,a.random_seed_base+1000+seed)
        }
    # Phase III: coefficients are frozen. Only now form evaluation Jacobians,
    # calculate exact per-block test solutions and OFFLINE test SVD certificate.
    for seed in (0,1,2):
        t=cached[seed]
        obj=matrices(t['state'],t['xy'][t['test']],t['y'][t['test']],a.max_forward)
        sigma_min=float(torch.linalg.svdvals(obj['A'])[:,-1].min())
        eigJ_test=np.linalg.eigvals(np.eye(obj['A'].shape[1])-obj['A'].numpy()).ravel()
        for method in ('AR','CAT-nominal','CAT-train-pixels','CAT-other-models'):
            if method in fits[seed]:
                fit=fits[seed][method]
                alpha,beta=fit['alpha'],fit['beta']
                cal_factor=fit['calibration_modal_factor']
                test_factor=worst_characteristic_root(eigJ_test,alpha,beta)
                n_cal_eigs=fit['training_eigenvalues']
            else:
                alpha,beta=(1.0,0.0) if method=='AR' else nominal_cat_coefficients()
                cal_factor='';test_factor='';n_cal_eigs=''
            # 'CAT-train-pixels' and 'CAT-other-models' are identical
            # fixed-coefficient CAT updates; only calibration data differ.
            solver_method='CAT-transfer' if method.startswith('CAT-') and method!='CAT-nominal' else method
            first,points=evaluate_method(obj,sigma_min,solver_method,alpha,beta,
                                         a.max_budget,a.certified_tol)
            record=dict(seed=seed,method=method,first_tested_passing_budget=first,
                        charged_actions=(first+1 if first is not None else None),
                        alpha=alpha,beta=beta,calibration_modal_factor=cal_factor,
                        test_modal_factor_after_freeze=test_factor,
                        n_training_eigenvalues=n_cal_eigs,min_test_sigma_A=sigma_min)
            rows.append(record)
            curves.extend(dict(seed=seed,protocol=method,**point) for point in points)
            print('TRANSFER',seed,method,'charged',record['charged_actions'],
                  'alpha',f'{alpha:.5f}','beta',f'{beta:.5f}',flush=True)
        provenance.append(dict(seed=seed,checkpoint_sha256=t['sha'],
            upstream_git_commit=t['manifest']['upstream_git_commit'],
            previous_audit_max_norm_reproduced=t['old_norm'],
            upstream_direct_check=t['upstream_check'],calibration_count=len(t['cal']),
            test_count=len(t['test']),excluded_prior_holdout_count=a.exclude_prior_audit,
            calibration_indices=[int(z) for z in t['cal']],
            test_indices=[int(z) for z in t['test']],
            trained_on_same_image=True,test_min_singular_postfit=sigma_min,
            frozen_fits=fits[seed]))
    with (a.out/'transfer_first_pass.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    with (a.out/'transfer_curves.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(curves[0].keys()));wr.writeheader();wr.writerows(curves)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    report=dict(classification='NATIVE_UNPROJECTED_DEQINR_PIXEL_AND_LEAVE_ONE_MODEL_OUT_SPECTRAL_TRANSFER',
        training_models_original_unprojected_TorchDEQ=True,new_training=False,
        fit_uses_any_test_eigenvalues=False,calibration_is_local=False,
        calibration_uses_global_eigenvalues_of_training_pixel_blocks=True,
        leave_one_model_out_fit_uses_no_Jacobians_of_heldout_model=True,
        stopping_bound_uses_offline_global_test_SVD=True,
        retrospective_per_iteration_checks_not_charged_as_online_stopping=True,
        prior_first_16_heldout_pixels_excluded_from_new_evaluation=True,
        caveat='Models and camera image were explored earlier; transfer experiment does not constitute new independent image-domain validation',
        parameters=config,models=provenance)
    (a.out/'transfer_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('DONE',a.out.resolve(),flush=True)


if __name__=='__main__':run(parse_arguments())