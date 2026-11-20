#!/usr/bin/env python3
"""CAT/AR finite-tolerance spectral audit.

Standalone reproduction of the controlled 2-D graph/tanh setup and coefficient
formulas from dvlahek/cat-mwbp experiments/run_implicit_spectral_suite.py.
The experiment here is new and must not be conflated with eight-dataset audit.
Reports exact adjoint (linear) or implicit-gradient (tanh) error, measured
against an offline direct solve; this is NOT a deployable local stop test.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, diags, eye
from scipy.sparse.linalg import spsolve


def graph(side, rho, asymmetric=True):
    n = side * side
    W = lil_matrix((n, n), dtype=float)
    def idx(r, c): return (r % side) * side + (c % side)
    for r in range(side):
        for c in range(side):
            i = idx(r, c)
            if asymmetric:
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    W[i, idx(rr, cc)] = 0.15 * rho
                W[i, idx(r-1, c+1)] = 0.20 * rho
                W[i, idx(r+1, c-1)] = 0.20 * rho
            else:
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    W[i, idx(rr, cc)] = 0.25 * rho
    return W.tocsr()


def coeff(mu, L):
    sm, sL = np.sqrt(mu), np.sqrt(L)
    return 2/(L+mu), 4/(sL+sm)**2, ((sL-sm)/(sL+sm))**2


def tanh_state(W, u):
    h = np.zeros_like(u)
    for _ in range(30000):
        new = np.tanh(W @ h + u)
        if np.linalg.norm(new-h) <= 1e-12*(1+np.linalg.norm(new)):
            return new
        h = new
    raise RuntimeError('tanh equilibrium did not converge')


def setup(problem, side, rho, seed):
    W = graph(side, rho, True)
    n = W.shape[0]
    # Match the original manuscript spectral experiment's seed offset and RHS.
    rng = np.random.default_rng(seed + 10000 + 100000*(problem == 'tanh'))
    if problem == 'linear':
        d = np.ones(n)
        random_part = rng.normal(size=n)
        random_part /= np.linalg.norm(random_part) + 1e-30
        slow = np.ones(n)/np.sqrt(n)
        rhs = 0.85*random_part + 0.15*slow
        rhs /= np.linalg.norm(rhs) + 1e-30
    else:
        u = 0.05 * rng.normal(size=n)
        h = tanh_state(W, u)
        d = 1 - h * h
        readout = rng.normal(size=n)
        readout /= np.linalg.norm(readout) + 1e-30
        rhs = (float(readout @ h)-1.0)*readout
    A = eye(n, format='csr') - W @ diags(d)
    exact = np.asarray(spsolve(A, rhs))
    target = (d * exact) if problem == 'tanh' else exact
    sd = np.sqrt(np.maximum(d, 1e-15))
    eigs = np.linalg.eigvalsh((diags(sd) @ W @ diags(sd)).toarray())
    mu, L = 1 - float(eigs.max()), 1 - float(eigs.min())
    return W, d, rhs, target, (mu, L), float(max(abs(eigs.min()), abs(eigs.max())))


def solve_trace(W, d, rhs, target, problem, alpha, beta, max_actions, tolerances):
    x = np.zeros_like(rhs)
    previous = np.zeros_like(rhs)
    tnorm = np.linalg.norm(target) + 1e-30
    hits = {}
    for k in range(0, max_actions+1):
        observed = (d*x) if problem == 'tanh' else x
        err = float(np.linalg.norm(observed-target) / tnorm)
        for tol in tolerances:
            if tol not in hits and err <= tol:
                hits[tol] = (k, err)
        if len(hits) == len(tolerances) or k == max_actions:
            break
        residual = rhs - (x - W @ (d*x))
        next_x = x + alpha*residual + beta*(x-previous)
        previous, x = x, next_x
        if not np.all(np.isfinite(x)):
            break
    for tol in tolerances:
        hits.setdefault(tol, (np.nan, np.nan))
    return hits


def run(args):
    tolerances = sorted({float(s) for s in args.tolerances.split(',')}, reverse=True)
    rhos = [float(s) for s in args.rhos.split(',')]
    rows=[]
    for problem in ('linear','tanh'):
        for rho in rhos:
            for seed in range(args.seeds):
                W, d, rhs, target, oracle_interval, rho_actual = setup(problem, args.side, rho, seed)
                for calibration in ('bound','oracle'):
                    mu,L = (1-rho,1+rho) if calibration=='bound' else oracle_interval
                    aa,ca,cb=coeff(mu,L)
                    qar=(L-mu)/(L+mu)
                    qcat=(np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu))
                    theoretical_ratio=float(np.log(qcat)/np.log(qar))
                    for method,alpha,beta in (('AR',aa,0.),('CAT',ca,cb)):
                        hits=solve_trace(W,d,rhs,target,problem,alpha,beta,args.max_actions,tolerances)
                        for tol,(steps,err) in hits.items():
                            rows.append(dict(problem=problem,rho_cap=rho,actual_rho=rho_actual,seed=seed,
                                side=args.side,state_dimension=args.side**2,calibration=calibration,
                                tolerance=tol,method=method,actions=steps,relative_target_error=err,
                                mu=mu,L=L,q_ar=qar,q_cat=qcat,theoretical_ratio=theoretical_ratio))
                print(f'{problem} rho={rho:.3f} seed={seed} complete',flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    runs=pd.DataFrame(rows)
    runs.to_csv(out/'tolerance_runs.csv',index=False)
    joined=runs.pivot_table(index=['problem','rho_cap','actual_rho','seed','side','state_dimension',
                    'calibration','tolerance','mu','L','q_ar','q_cat','theoretical_ratio'],
              columns='method',values='actions').reset_index()
    joined['ar_over_cat_ratio']=joined['AR']/joined['CAT']
    joined['measured_over_theoretical']=joined['ar_over_cat_ratio']/joined['theoretical_ratio']
    joined.rename(columns={'AR':'ar_actions','CAT':'cat_actions'},inplace=True)
    joined.to_csv(out/'tolerance_pairs.csv',index=False)
    summary=joined.groupby(['problem','calibration','rho_cap','tolerance']).agg(
        n=('ar_over_cat_ratio','count'),median_ar_actions=('ar_actions','median'),
        median_cat_actions=('cat_actions','median'),
        median_ratio=('ar_over_cat_ratio','median'),
        median_measured_over_theoretical=('measured_over_theoretical','median'),
        theoretical_ratio=('theoretical_ratio','median')).reset_index()
    summary.to_csv(out/'tolerance_summary.csv',index=False)
    return runs,joined,summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rhos',default='0.8,0.95,0.99')
    p.add_argument('--tolerances',default='1e-3,1e-4,1e-5,1e-6,1e-7')
    p.add_argument('--side',type=int,default=12)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--max-actions',type=int,default=12000)
    p.add_argument('--out',default='results/cat_tolerance')
    a=p.parse_args()
    run(a)
