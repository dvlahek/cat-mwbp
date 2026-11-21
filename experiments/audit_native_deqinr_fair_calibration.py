#!/usr/bin/env python3
"""Fair post-hoc AR versus CAT calibration study on frozen native TorchDEQ checkpoints.

Two exploratory tests, both independent of fitting-target Jacobian spectra:
  A. Spectral minimax on 64 training-pixel blocks, evaluation at heldout 80:208.
     Compare freely tuned AR and CAT, leave-one-model-out AR and CAT, and CAT
     constrained to nontrivial beta >= 0.05, 0.10 and 0.20.
  B. Finite-horizon (10, 16, 22-step normalized residual) optimization on the
     SAME training-pixel blocks, evaluation at heldout 208:336. Same objective
     and optimization domain for AR and CAT, with the latter having beta >= 0.

All models were independently trained; no model is retrained. The data domain
(single camera image, original pixel split) has been used in previous studies.
These are diagnostic, post-hoc checks, not an independent publication-grade
benchmark or a test of end-to-end locally implementable stopping/calibration.

Requires audit_native_deqinr_transfer.py in this directory and archived trained
checkpoints. Checkpoint hashes are verified against original manifests.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution, minimize_scalar
import audit_native_deqinr_transfer as base


def factor(eigs, alpha, beta):
    t = 1. + beta - alpha * (1. - eigs)
    sq = np.sqrt(t*t - 4.*beta + 0j)
    return float(np.maximum(np.abs((t + sq)*.5), np.abs((t - sq)*.5)).max())


def minmax_coefficients(eig, seed, beta_min=0.):
    result = differential_evolution(
        lambda ab: factor(eig, float(ab[0]), float(ab[1])),
        bounds=[(.1, 1.9), (beta_min, .85)],seed=seed,popsize=9,
        maxiter=85,tol=1e-7,polish=True,workers=1)
    return float(result.x[0]),float(result.x[1]),float(result.fun)


def minmax_ar(eig):
    r=minimize_scalar(lambda a:factor(eig,float(a),0.),bounds=(.1,1.9),
                      method='bounded',options={'xatol':1e-12})
    return float(r.x),0.,float(r.fun)


def residual_objective(W,d,b,eig,alpha,beta):
    if not (.2<=alpha<=1.8 and 0<=beta<=.6):return 1000.
    fac=factor(eig,alpha,beta)
    if fac>=.98:return 50.+100.*(fac-.98)
    horizons=(10,16,22)
    weights=(.3,.4,.3)
    x=torch.zeros_like(b)
    old=torch.zeros_like(b)
    denom=torch.linalg.vector_norm(b,dim=-1).clamp_min(1e-15)
    score=0.
    for k in range(1,23):
        r=b-x+(x*d)@W
        nxt=x+alpha*r+beta*(x-old)
        old,x=x,nxt
        if k in horizons:
            r=b-x+(x*d)@W
            logs=torch.log((torch.linalg.vector_norm(r,dim=-1)/denom).clamp_min(1e-100))
            score+=weights[horizons.index(k)]*float(logs.mean()+.25*torch.quantile(logs,.9))
    return float(score)


def finite_coefficients(W,d,b,eig,seed):
    fn=lambda a,beta:residual_objective(W,d,b,eig,a,beta)
    r=minimize_scalar(lambda a:fn(float(a),0.),bounds=(.2,1.8),
                      method='bounded',options={'xatol':1e-8})
    arbest=min([(float(r.fun),float(r.x))]+[(fn(float(a),0.),float(a))
                                         for a in np.linspace(.2,1.8,161)])
    c=differential_evolution(lambda ab:fn(float(ab[0]),float(ab[1])),
                             bounds=[(.2,1.8),(0.,.6)],seed=seed,
                             popsize=10,maxiter=85,tol=1e-7,polish=True,workers=1)
    return {'AR-finite':(arbest[1],0.,arbest[0]),
            'CAT-finite':(float(c.x[0]),float(c.x[1]),float(c.fun))}


def iterate(obj,alpha,beta,minimum_singular,max_iter,threshold,seed,protocol,name):
    W,d,b,exact=obj['W'],obj['d'],obj['b'],obj['exact']
    x=torch.zeros_like(b);old=torch.zeros_like(b)
    ref=float(torch.linalg.vector_norm(exact))
    first=None;curves=[]
    for k in range(1,max_iter+1):
        r=b-x+(x*d)@W
        xn=x+alpha*r+beta*(x-old)
        old,x=x,xn
        residual=b-x+(x*d)@W  # one extra JTv, offline every iteration
        E=float(torch.linalg.vector_norm(residual))/minimum_singular
        nx=float(torch.linalg.vector_norm(x))
        cert=E/(nx-E) if nx>E else math.inf
        actual=float(torch.linalg.vector_norm(x-exact))/(ref+1e-30)
        if cert<actual-1e-11*max(1.,actual):
            raise RuntimeError(f'SVD error bound invalid: {seed} {name} {k}')
        curves.append({'seed':seed,'protocol':protocol,'method':name,'budget':k,
                       'terminal_charged_actions':k+1,
                       'offline_svd_relative_bound':cert,
                       'actual_rel_adjoint_error':actual})
        if first is None and cert<threshold:first=k+1
    return first,curves


def make_args():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--root',type=Path)
    g.add_argument('--archive',type=Path)
    p.add_argument('--torchdeq-zoo',type=Path,default=None,
                   help='Optional original upstream modules for independent forward/JTv check')
    p.add_argument('--out',type=Path,default=Path('results/native_deqinr_fair_calibration'))
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-budget',type=int,default=120)
    p.add_argument('--tolerance',type=float,default=1e-5)
    p.add_argument('--calibration-seed-base',type=int,default=20261121)
    p.add_argument('--calibration-count',type=int,default=64)
    p.add_argument('--prefix',default='official_deqinr_native_s')
    p.add_argument('--max-forward',type=int,default=1000)
    a=p.parse_args()
    if a.calibration_count!=64 or a.max_budget<25:
        p.error('Predeclared study requires exactly 64 training calibration pixels and max-budget >=25')
    return a


def main():
    a=make_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    cache={};provenance=[]
    for seed in range(3):
        state,old_manifest,sha=base.load_state(a,seed)
        xy,target,train,hold=base.image_split(seed,int(old_manifest['configuration']['side']))
        if len(hold)<336:raise ValueError('Insufficient untouched heldout pixels')
        generator=np.random.default_rng(a.calibration_seed_base+seed)
        calix=generator.choice(train,size=a.calibration_count,replace=False)
        obj=base.matrices(state,xy[calix],target[calix],a.max_forward)
        original=base.verify_previous_audit(state,old_manifest,
                    xy[hold[:16]],target[hold[:16]],a.max_forward)
        upstream=base.verify_upstream_model(a,state,xy[calix],obj)
        eig=np.linalg.eigvals(np.eye(obj['A'].shape[1])-obj['A'].numpy()).ravel()
        cache[seed]={'model':state,'xy':xy,'target':target,'train':train,'hold':hold,
                     'calix':calix,'object':obj,'eig':eig,'sha':sha,
                     'orig_Jnorm':original,'upstream':upstream}
    # All fitted parameters are frozen before constructing any evaluation matrices.
    fits={}
    for seed in range(3):
        q=cache[seed];own=q['eig']
        other=np.concatenate([cache[j]['eig'] for j in range(3) if j!=seed])
        sfits={'AR-alpha1':(1.,0.,factor(own,1.,0.)),
               'AR-spectral':minmax_ar(own),
               'CAT-spectral':minmax_coefficients(own,seed+2026,0.),
               'AR-other-models':minmax_ar(other),
               'CAT-other-models':minmax_coefficients(other,seed+3026,0.)}
        for betamin in (.025,.05,.10,.15,.20,.30):
            sfits[f'CAT-beta-min-{betamin:g}']=minmax_coefficients(own,seed+2026,betamin)
        finit=finite_coefficients(q['object']['W'],q['object']['d'],q['object']['b'],
                                  own,20261121+seed)
        fits[seed]={'spectral':sfits,'finite':finit}
    print('CALIBRATION_FROZEN_BEFORE_EVALUATION',flush=True)
    rows=[];curves=[]
    for seed in range(3):
        q=cache[seed]
        for prot,lo,hi in [('spectral',80,208),('finite',208,336)]:
            testix=q['hold'][lo:hi]
            assert len(testix)==128
            assert len(np.intersect1d(q['calix'],testix))==0
            obj=base.matrices(q['model'],q['xy'][testix],q['target'][testix],a.max_forward)
            mu=float(torch.linalg.svdvals(obj['A'])[:,-1].min())
            eigtest=np.linalg.eigvals(np.eye(obj['A'].shape[1])-obj['A'].numpy()).ravel()
            protfits=fits[seed][prot].copy()
            if prot=='finite':protfits['AR-alpha1']=(1.,0.,residual_objective(
                q['object']['W'],q['object']['d'],q['object']['b'],q['eig'],1.,0.))
            for name,(alpha,beta,train_objective) in protfits.items():
                first,pts=iterate(obj,alpha,beta,mu,a.max_budget,a.tolerance,seed,prot,name)
                row={'seed':seed,'protocol':prot,'method':name,'alpha':alpha,'beta':beta,
                     'calibration_objective':train_objective,
                     'training_modal_factor':factor(q['eig'],alpha,beta),
                     'postfit_test_modal_factor':factor(eigtest,alpha,beta),
                     'first_charged_actions':first,'test_min_singular_A':mu,
                     'test_pixels':len(testix)}
                rows.append(row);curves.extend(pts)
                print('RESULT',seed,prot,name,'charged',first,
                      'beta',f'{beta:.5f}',flush=True)
        provenance.append({'seed':seed,'checkpoint_sha256':q['sha'],
                           'upstream_model_check':q['upstream'],
                           'previous_audit_Jnorm_reproduced':q['orig_Jnorm'],
                           'calibration_indices':[int(x) for x in q['calix']],
                           'spectral_test_indices':[int(x) for x in q['hold'][80:208]],
                           'finite_test_indices':[int(x) for x in q['hold'][208:336]]})
    pd.DataFrame(rows).to_csv(a.out/'fair_calibration_summary.csv',index=False)
    pd.DataFrame(curves).to_csv(a.out/'fair_calibration_curves.csv',index=False)
    manifest={'classification':'EXPLORATORY_POSTHOC_FAIR_COMPLEX_DEQ_CALIBRATION',
              'context':'Unmodified upstream DEQ-INR model module, compact custom training; one camera image',
              'prior_pilot_used_holdout_0_to_16':True,
              'prior_transfer_used_holdout_16_to_80':True,
              'prior_current_spectral_diagnostic_used_holdout_80_to_208':True,
              'spectral_test_holdout_positions':'80:208',
              'finite_test_holdout_positions':'208:336',
              'no_test_jacobians_used_for_coefficients':True,
              'training_global_eigenvalues_used':True,
              'global_svd_used_for_test_certification':True,
              'every_iteration_certification_is_retrospective_and_not_charged':True,
              'actual_online_termination_is_not_implemented':True,
              'fair_comparators':'AR and CAT calibrated on identical spectral or finite-horizon training objective',
              'finite_horizons':[10,16,22],
              'finite_weights':[.3,.4,.3],
              'finite_training_loss':'weighted log normalized training-pixel residual with 0.25 x 90th percentile log residual; modal stability penalty for factor >= 0.98',
              'number_models':3,'target_relative_adjoint_bound':a.tolerance,
              'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'provenance':provenance}
    (a.out/'fair_calibration_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('NATIVE_DEQ_FAIR_CALIBRATION_COMPLETE',len(rows),len(curves),flush=True)

if __name__=='__main__':main()