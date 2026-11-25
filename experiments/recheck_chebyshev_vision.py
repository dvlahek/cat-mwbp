#!/usr/bin/env python3
"""Re-evaluate corrected oracle Chebyshev on CAT-trained MNIST/CIFAR-10.

Run from the root of dvlahek/cat-mwbp, or give --repo-root. The original
models are not archived, so the script deterministically retrains only
CAT-bound checkpoints (3 seeds x 2 datasets), then compares exact-error
CAT-oracle, AR-oracle and normalized first-kind Chebyshev on identical frozen
checkpoint adjoints (8 held-out samples per seed). It does not claim results
until the run completes on actual datasets.

Dataset provenance mirrors experiments/run_implicit_dataset_suite.py:
MNIST from OpenML mnist_784 v1 and CIFAR10 from the original Toronto archive.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-6


def chebyshev_corrected(rhs, jtv, error, mu, L, max_steps=4000):
    """Normalized first-kind Chebyshev 3-term iteration for A=I-J^T.

    For k=2 use delta**2/2, then delta**2/4 thereafter. Here k denotes
    the 1-based action index, and all solvers use the same zero initial guess.
    """
    center, delta = (L+mu)/2, (L-mu)/2
    x, prev = np.zeros_like(rhs), np.zeros_like(rhs)
    alpha = 1/center
    for k in range(1, max_steps+1):
        r = rhs - (x-jtv(x))
        if k == 1:
            nxt = x + alpha*r
        else:
            factor = (delta*delta/2) if k == 2 else (delta*delta/4)
            next_alpha = 1/(center - factor*alpha)
            momentum = factor * alpha * next_alpha
            nxt = x + next_alpha*r + momentum*(x-prev)
            alpha = next_alpha
        prev, x = x, nxt
        if error(x) <= TOL:
            return k, x, error(x)
    raise RuntimeError('Chebyshev did not reach the specified tolerance')


def stationary(rhs, jtv, error, alpha, beta, max_steps=4000):
    x, prev = np.zeros_like(rhs), np.zeros_like(rhs)
    for k in range(1,max_steps+1):
        r = rhs-(x-jtv(x))
        nxt = x+alpha*r+beta*(x-prev)
        prev,x=x,nxt
        if error(x) <= TOL:
            return k,x,error(x)
    raise RuntimeError('stationary solver did not reach the specified tolerance')


def self_test():
    # Independent polynomial-identity test: P_k(a) = T_k(z(a))/T_k(z(0)).
    from numpy.polynomial.chebyshev import chebval
    mu,L=0.08,1.80
    evals=np.array([mu,0.33,0.8,1.2,L])
    rhs=np.ones(len(evals)); center,half=(L+mu)/2,(L-mu)/2
    x,old=np.zeros_like(rhs),np.zeros_like(rhs); alpha=1/center
    for k in range(1,31):
        r=rhs-evals*x
        if k==1:
            nx=x+alpha*r
        else:
            factor=(half**2/2) if k==2 else (half**2/4)
            na=1/(center-factor*alpha)
            nx=x+na*r+factor*alpha*na*(x-old)
            alpha=na
        old,x=x,nx
        e=1-evals*x # since rhs=1, exact solution=1/evals
        z=(center-evals)/half; z0=center/half
        coeff=np.zeros(k+1);coeff[-1]=1
        expected=chebval(z,coeff)/chebval(z0,coeff)
        if np.max(np.abs(e-expected))>4e-10:
            raise AssertionError(f'Chebyshev polynomial mismatch at k={k}: {e-expected}')
    return True


def audit_one(model,x,y,recur,recur_dense,coeff,seed,dataset):
    x=x[:8]; y=y[:8]
    h,_=model.equilibrium(x)
    z=h@model.Wout+model.bo
    z=z-z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    p[np.arange(len(y)),y]-=1
    rhs_all=p@model.Wout.T
    W=recur_dense(model.hidden,model.rho)
    rows=[]
    for index,(rhs,hi) in enumerate(zip(rhs_all,h)):
        d=1-hi*hi
        A=np.eye(model.hidden)-W@np.diag(d)
        exact=np.linalg.solve(A,rhs)
        exact_gradient=d*exact
        denominator=np.linalg.norm(exact_gradient)+1e-30
        sd=np.sqrt(np.maximum(d,1e-15))
        ev=np.linalg.eigvalsh(sd[:,None]*W*sd[None,:])
        mu,L=1-float(ev.max()),1-float(ev.min())
        ar,ca,cb=coeff(mu,L)
        jtv=lambda v:recur(d*v,model.rho)
        error=lambda v:float(np.linalg.norm(d*v-exact_gradient)/denominator)
        solvers={
            'AR-oracle':stationary(rhs,jtv,error,ar,0),
            'CAT-oracle':stationary(rhs,jtv,error,ca,cb),
            'Chebyshev-oracle-corrected':chebyshev_corrected(rhs,jtv,error,mu,L),
        }
        for name,(actions,lam,e) in solvers.items():
            rows.append({'dataset':dataset,'seed':seed,'sample':index,'method':name,
                         'actions':actions,'relative_gradient_error':e,
                         'rho_jacobian':float(max(abs(ev.min()),abs(ev.max()))),
                         'test_accuracy':None})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('results/chebyshev_vision_recheck'))
    parser.add_argument('--datasets',default='mnist,cifar10')
    parser.add_argument('--seeds',default='0,1,2')
    parser.add_argument('--smoke',action='store_true',help='Polynomial identity only; no data download')
    args=parser.parse_args()
    self_test()
    print('CORRECTED_CHEBYSHEV_POLYNOMIAL_SELFTEST_PASS',flush=True)
    if args.smoke: return
    root=args.repo_root.resolve()
    script=root/'experiments'/'run_implicit_dataset_suite.py'
    if not script.is_file(): parser.error(f'Missing {script}: run from the CAT repository root')
    sys.path.insert(0,str(script.parent))
    from run_implicit_dataset_suite import train,recur,recur_dense,coeff
    datasets=[s.strip() for s in args.datasets.split(',') if s.strip()]
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    if any(x not in ('mnist','cifar10') for x in datasets):parser.error('Only mnist,cifar10 are supported')
    args.out.mkdir(parents=True,exist_ok=True)
    allrows=[]; trainrows=[]
    for dataset in datasets:
        for seed in seeds:
            print('TRAIN CAT',dataset,'seed',seed,flush=True)
            try:
                tr,model,xt,yt=train(dataset,seed,.95,'CAT-bound','standard',35,20,48,64,64,8000,7)
            except Exception as ex:
                raise RuntimeError(f'Cannot load/train {dataset} seed={seed}. Original public dataset is needed. {ex}') from ex
            trainrows.append(tr)
            # Save the exact trained checkpoint and held-out audit inputs so the
            # comparison can be rerun without downloading or retraining again.
            np.savez_compressed(args.out/f'{dataset}_seed{seed}_cat_checkpoint.npz',
                                Win=model.Win,bh=model.bh,Wout=model.Wout,bo=model.bo,
                                audit_x=xt[:8],audit_y=yt[:8],rho=model.rho,
                                seed=seed,test_accuracy=tr['test_accuracy'])
            rows=audit_one(model,xt,yt,recur,recur_dense,coeff,seed,dataset)
            allrows.extend(rows)
            pd.DataFrame(allrows).to_csv(args.out/'vision_corrected_chebyshev_actions.csv',index=False)
            pd.DataFrame(trainrows).to_csv(args.out/'vision_cat_retraining.csv',index=False)
            a=pd.DataFrame(rows).pivot(index='sample',columns='method',values='actions')
            print('EVAL',dataset,'seed',seed,'accuracy',tr['test_accuracy'],
                  'median CAT',np.median(a['CAT-oracle']),
                  'median Chebyshev',np.median(a['Chebyshev-oracle-corrected']),
                  'median per-sample Cheb/CAT',np.median(a['Chebyshev-oracle-corrected']/a['CAT-oracle']),flush=True)
    data=pd.DataFrame(allrows)
    pivot=data.pivot(index=['dataset','seed','sample'],columns='method',values='actions').reset_index()
    pivot['chebyshev_over_cat']=pivot['Chebyshev-oracle-corrected']/pivot['CAT-oracle']
    pivot['ar_over_cat']=pivot['AR-oracle']/pivot['CAT-oracle']
    pivot.to_csv(args.out/'vision_corrected_chebyshev_pairs.csv',index=False)
    seed_summary=pivot.groupby(['dataset','seed'],as_index=False).agg(
        median_chebyshev_over_cat=('chebyshev_over_cat','median'),
        median_ar_over_cat=('ar_over_cat','median'),
        median_cat_actions=('CAT-oracle','median'),
        median_chebyshev_actions=('Chebyshev-oracle-corrected','median'),
    )
    seed_summary.to_csv(args.out/'vision_corrected_chebyshev_seed_summary.csv',index=False)
    overall=seed_summary.groupby('dataset',as_index=False).agg(
        dataset_median_chebyshev_over_cat=('median_chebyshev_over_cat','median'),
        dataset_median_cat_actions=('median_cat_actions','median'),
        dataset_median_chebyshev_actions=('median_chebyshev_actions','median'),
        n_seeds=('seed','nunique'))
    overall.to_csv(args.out/'vision_corrected_chebyshev_dataset_summary.csv',index=False)
    print('\nDATASET_SUMMARY\n'+overall.to_string(index=False),flush=True)
    manifest=dict(source_script='experiments/run_implicit_dataset_suite.py',
                  source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  datasets=datasets,seeds=seeds,tolerance=TOL,
                  training='CAT-bound only, profile standard: 8000 samples, width 64, max 20 epochs, 7 validation-patience',
                  test='8 held-out samples per seed, same frozen CAT-trained checkpoint and error criterion for all solvers',
                  chebyshev='normalized first-kind: delta^2/2 on second update, delta^2/4 subsequently',
                  caveat='The original checkpoints are not archived; retraining should be cross-checked against old accuracy and CAT action summaries.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
