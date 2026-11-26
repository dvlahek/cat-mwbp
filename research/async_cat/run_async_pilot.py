#!/usr/bin/env python3
"""Paired bounded-staleness pilot; offline diagnostics, not network timing."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS = ('AR', 'CAT', 'Chebyshev')


def problem(side, rho, seed, kind):
    n = side**2
    grid = np.arange(n).reshape(side, side)
    nbr = np.stack([np.roll(grid, shift, axis).ravel()
                    for axis in (0, 1) for shift in (-1, 1)] +
                   [np.roll(grid, (s, -s), axis=(0, 1)).ravel()
                    for s in (-1, 1)], axis=1)
    weights = rho * np.array([.15]*4 + [.20]*2)
    W = np.zeros((n, n))
    W[np.arange(n)[:, None], nbr] = weights
    rng = np.random.default_rng(seed)
    if kind == 'tanh':
        u = .05*rng.normal(size=n)
        h = np.zeros(n)
        for _ in range(30000):
            new = np.tanh(W@h+u)
            if np.linalg.norm(new-h) <= 1e-12*(1+np.linalg.norm(new)):
                h = new
                break
            h = new
        else:
            raise RuntimeError('forward equilibrium not reached')
        d = 1-h*h
        readout = rng.normal(size=n)
        readout /= np.linalg.norm(readout)
        b = (readout@h-1)*readout
    else:
        d = np.ones(n)
        b = rng.normal(size=n)
        b = .85*b/np.linalg.norm(b)+.15/np.sqrt(n)
    B = W*d[None, :]
    exact = np.linalg.solve(np.eye(n)-B, b)
    return nbr, weights*d[nbr], b, d, exact, B


def solve(case, rho, method, p, lag, seed, max_ticks, tol=1e-6):
    nbr, weights, b, d, exact, B = case
    n = len(b)
    rng = np.random.default_rng(seed)
    x = np.zeros(n); prev = x.copy()
    hist = np.zeros((lag+1, n))
    counter = np.zeros(n, dtype=int)
    idle = np.zeros(n, dtype=int)
    alpha = np.ones(n)
    q = (np.sqrt(1+rho)-np.sqrt(1-rho))/(np.sqrt(1+rho)+np.sqrt(1-rho))
    cat_a = 4/(np.sqrt(1+rho)+np.sqrt(1-rho))**2
    status = 'budget'; work = 0
    for tick in range(1, max_ticks+1):
        active = np.flatnonzero((rng.random(n)<p) | (idle>=7))
        delays = rng.integers(0, lag+1, size=nbr.shape)
        if len(active):
            r = b[active]-x[active]+np.sum(weights[active]*hist[delays[active], nbr[active]], axis=1)
            if method == 'AR':
                a, beta = 1., 0.
            elif method == 'CAT':
                a, beta = cat_a, q*q
            else:
                k = counter[active]
                f = np.where(k==1, rho*rho/2, rho*rho/4)
                old = alpha[active]
                a = np.where(k==0, 1., 1/(1-f*old))
                beta = np.where(k==0, 0., f*old*a)
                alpha[active] = a
            new = x[active]+a*r+beta*(x[active]-prev[active])
            prev[active] = x[active]; x[active] = new
            counter[active] += 1
            work += len(active)
        idle += 1; idle[active] = 0
        hist[1:] = hist[:-1].copy(); hist[0] = x
        err = np.linalg.norm(x-exact)/np.linalg.norm(exact)
        gerr = np.linalg.norm(d*(x-exact))/np.linalg.norm(d*exact)
        if not np.isfinite(err) or err>1e8:
            status = 'diverged'; break
        if max(err,gerr)<=tol:
            status = 'converged'; break
    return dict(method=method, activation=p, max_delay=lag, schedule_seed=seed,
                ticks=tick, row_actions=work, sweep_equiv=work/n,
                edge_reads=work*nbr.shape[1], adjoint_error=float(err),
                gradient_error=float(gerr), status=status)


def selftest():
    # Zero-delay convergence, charged work and direct-reference consistency.
    case = problem(4, .8, 0, 'linear')
    b, exact = case[2:5:2]
    for method in METHODS:
        r = solve(case,.8,method,1.,0,9,2000)
        assert r['status']=='converged' and r['row_actions']==16*r['ticks']
    for p in (1., .5):
        r = solve(case,.8,'AR',p,2,9,2000)
        assert r['status']=='converged'
    assert np.linalg.norm((np.eye(len(b))-case[-1])@exact-b)<1e-10


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('results/async_pilot'))
    ap.add_argument('--profile',choices=('quick','standard'),default='quick')
    args=ap.parse_args(); selftest(); args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for kind in (('linear',) if args.profile=='quick' else ('linear','tanh')):
        for rho in (.8,.95,.99):
            for seed in range(2 if args.profile=='quick' else 5):
                case=problem(12,rho,seed,kind)
                for p in (1.,.5):
                    for lag in (0,1,2,4):
                        for method in METHODS:
                            row=solve(case,rho,method,p,lag,1000+seed,3000)
                            rows.append(dict(problem=kind,rho=rho,seed=seed,**row))
    with (args.out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method in METHODS:
        for p in (1.,.5):
            for lag in (0,1,2,4):
                g=[r for r in rows if (r['method'],r['activation'],r['max_delay'])==(method,p,lag)]
                ok=[r for r in g if r['status']=='converged']
                summary.append(dict(method=method,activation=p,max_delay=lag,n=len(g),converged=len(ok),diverged=sum(r['status']=='diverged' for r in g),median_success_sweeps=float(np.median([r['sweep_equiv'] for r in ok])) if ok else None))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('ASYNC_PILOT_PASS',len(rows),'runs; PASS means execution, not solver convergence')
    for r in summary: print(r)

if __name__=='__main__': main()
