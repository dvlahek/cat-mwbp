#!/usr/bin/env python3
"""Re-audit manuscript Chebyshev controls with standard Chebyshev recurrence.

The v1.0.0 run_implicit_spectral_suite.py used the wrong coefficient for the
first inertial step and a mismatched momentum factor thereafter. This new
script DOES NOT overwrite frozen v1.0.0 result files. It reconstructs each
reference case and compares CAT to the corrected Chebyshev semi-iteration.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from run_implicit_spectral_suite import graph, tanh_state, interval, coeff


def chebyshev(rhs,jtv,error,mu,L,max_actions):
    c=(mu+L)/2
    delta=(L-mu)/2
    alpha=1/c
    x=np.zeros_like(rhs)
    prev=np.zeros_like(rhs)
    for k in range(max_actions+1):
        e,ge=error(x)
        if e<=1e-6 and ge<=1e-6:return k,x,e,ge
        if k==max_actions:break
        r=rhs-(x-jtv(x))
        if k==0:
            xn=x+alpha*r
        else:
            factor=(delta**2/2) if k==1 else (delta**2/4)
            new_alpha=1/(c-factor*alpha)
            momentum=factor*alpha*new_alpha
            xn=x+new_alpha*r+momentum*(x-prev)
            alpha=new_alpha
        prev,x=x,xn
    return max_actions,x,*error(x)


def cat(rhs,jtv,error,mu,L,max_actions):
    _,alpha,beta=coeff(mu,L)
    x=np.zeros_like(rhs)
    old=np.zeros_like(rhs)
    for k in range(max_actions+1):
        e,ge=error(x)
        if e<=1e-6 and ge<=1e-6:return k,x,e,ge
        if k==max_actions:break
        r=rhs-(x-jtv(x))
        xn=x+alpha*r+beta*(x-old)
        old,x=x,xn
    return max_actions,x,*error(x)


def make_case(topology,problem,rho,seed,side):
    asym=topology=='asym_grid2d'
    W=graph(side,rho,asym)
    n=W.shape[0]
    rng=np.random.default_rng(seed+10000*asym+100000*(problem=='tanh'))
    if problem=='linear':
        d=np.ones(n)
        rand=rng.normal(size=n);rand/=np.linalg.norm(rand)+1e-30
        slow=np.ones(n)/np.sqrt(n)
        rhs=.85*rand+.15*slow;rhs/=np.linalg.norm(rhs)+1e-30
        exact=np.asarray(spsolve(eye(n,format='csr')-W,rhs))
        jtv=lambda v:np.asarray(W@v)
    else:
        h=tanh_state(W,.05*rng.normal(size=n))
        d=1-h*h
        readout=rng.normal(size=n);readout/=np.linalg.norm(readout)
        rhs=(float(readout@h)-1.0)*readout
        exact=np.asarray(spsolve(eye(n,format='csr')-W@diags(d),rhs))
        jtv=lambda v:np.asarray(W@(d*v))
    grad=d*exact
    def error(x):
        e=float(np.linalg.norm(x-exact)/(np.linalg.norm(exact)+1e-30))
        ge=float(np.linalg.norm(d*x-grad)/(np.linalg.norm(grad)+1e-30))
        return e,ge
    return rhs,jtv,error,interval(W,d)


def selftest():
    # Polynomials reproduced by recurrence are Chebyshev T_k(Ahat)/T_k(c/d).
    import numpy.polynomial.chebyshev as ch
    mu,L=.05,1.95
    c=(L+mu)/2;delta=(L-mu)/2
    vals=np.array([mu,(mu+L)/3,L])
    A=np.diag(vals)
    b=np.ones(3)
    exact=np.linalg.solve(A,b)
    err=lambda x:(np.linalg.norm(x-exact),np.linalg.norm(x-exact))
    for k in [1,2,3,5,10]:
        # Ensure iteration gives exactly the Chebyshev polynomial after k steps.
        x=np.zeros(3);old=np.zeros(3);alpha=1/c
        for t in range(k):
            r=b-A@x
            if t==0: xn=x+alpha*r
            else:
                f=delta**2/2 if t==1 else delta**2/4
                new=1/(c-f*alpha)
                mom=f*alpha*new
                xn=x+new*r+mom*(x-old);alpha=new
            old,x=x,xn
        relmode=(exact-x)/exact
        arg=(c-vals)/delta
        expected=ch.chebval(arg,[0]*k+[1])/ch.chebval(c/delta,[0]*k+[1])
        if not np.allclose(relmode,expected,atol=1e-9,rtol=1e-9):
            raise RuntimeError(f'Chebyshev polynomial mismatch at k={k}: {relmode} vs {expected}')
    print('CHEBYSHEV_POLYNOMIAL_SELFTEST_PASS')


def run(a):
    selftest()
    side=8 if a.quick else 12
    rhos=(.95,) if a.quick else (.5,.8,.9,.95,.98,.99)
    seeds=(0,1) if a.quick else tuple(range(5))
    rows=[]
    for topology in ('grid2d','asym_grid2d'):
      for problem in ('linear','tanh'):
       for rho in rhos:
        for seed in seeds:
         rhs,jtv,error,spec=make_case(topology,problem,rho,seed,side)
         for calibration in ('oracle','bound'):
          mu,L=(spec['mu_exact'],spec['L_exact']) if calibration=='oracle' else (spec['mu_bound'],spec['L_bound'])
          for method,solver in [('CAT',cat),('Chebyshev',chebyshev)]:
           actions,x,e,ge=solver(rhs,jtv,error,mu,L,a.max_actions)
           if e>1e-6 or ge>1e-6:
               raise RuntimeError(f'Not converged {topology} {problem} {rho} {seed} {method} {calibration}')
           rows.append(dict(topology=topology,problem=problem,rho=rho,seed=seed,
                            calibration=calibration,method=method,actions=actions,
                            adjoint_error=e,gradient_error=ge))
         print(topology,problem,rho,seed,'OK',flush=True)
         pd.DataFrame(rows).to_csv(a.out/'chebyshev_corrected_runs.csv',index=False)
    df=pd.DataFrame(rows)
    p=df.pivot_table(index=['topology','problem','rho','seed','calibration'],columns='method',values='actions').reset_index()
    p['cheb_actions_over_cat_actions']=p.Chebyshev/p.CAT
    p.to_csv(a.out/'chebyshev_corrected_pairs.csv',index=False)
    summary=p.groupby(['topology','problem','rho','calibration'],as_index=False).agg(
        n=('seed','count'),median_cat_actions=('CAT','median'),
        median_cheb_actions=('Chebyshev','median'),
        median_cheb_over_cat=('cheb_actions_over_cat_actions','median'))
    summary.to_csv(a.out/'chebyshev_corrected_summary.csv',index=False)
    print('DONE',a.out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--out',type=Path,default=Path('results/chebyshev_corrected'))
    p.add_argument('--max-actions',type=int,default=16000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    run(a)