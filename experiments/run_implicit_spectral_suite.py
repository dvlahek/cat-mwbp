#!/usr/bin/env python3
"""Implicit/recurrent spectral benchmark for CAT.

Compares first-order activation relaxation (AR), second-order CAT, and
Chebyshev semi-iteration using the same local J^T v budget. The asymmetric
2-D graph adds one diagonal orientation to break the bipartite spectral
symmetry of the ordinary square grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, diags, eye
from scipy.sparse.linalg import spsolve

METHODS = ("AR-oracle", "CAT-oracle", "AR-bound", "CAT-bound", "Cheb-oracle", "Cheb-bound")


def settings(profile):
    if profile == "quick":
        return 8, (0.8, 0.95, 0.99), (0, 1), 4000
    if profile == "standard":
        return 12, (0.5, 0.8, 0.9, 0.95, 0.98, 0.99), tuple(range(5)), 8000
    return 16, (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995), tuple(range(10)), 18000


def graph(side, rho, asymmetric):
    n = side * side
    W = lil_matrix((n, n), dtype=float)
    def idx(r, c): return (r % side) * side + (c % side)
    for r in range(side):
        for c in range(side):
            i = idx(r, c)
            if asymmetric:
                for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    W[i, idx(rr, cc)] = 0.15 * rho
                W[i, idx(r-1, c+1)] = 0.20 * rho
                W[i, idx(r+1, c-1)] = 0.20 * rho
            else:
                for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    W[i, idx(rr, cc)] = 0.25 * rho
    return W.tocsr()


def coeff(mu, L):
    sm, sL = np.sqrt(mu), np.sqrt(L)
    return 2/(L+mu), 4/(sL+sm)**2, ((sL-sm)/(sL+sm))**2


def interval(W, d):
    sd = np.sqrt(np.maximum(d, 1e-15))
    S = diags(sd) @ W @ diags(sd)
    ev = np.linalg.eigvalsh(S.toarray())
    lo, hi = float(ev.min()), float(ev.max())
    row_bound = min(float(np.abs(S).sum(axis=1).A.ravel().max()), 1-1e-10)
    return {
        "lambda_min": lo, "lambda_max": hi,
        "rho_actual": max(abs(lo), abs(hi)),
        "symmetry_error": abs(lo + hi),
        "mu_exact": 1-hi, "L_exact": 1-lo,
        "mu_bound": 1-row_bound, "L_bound": 1+row_bound,
    }


def solve_stationary(rhs, exact, jtv, alpha, beta, max_steps, d=None):
    x = np.zeros_like(rhs); prev = np.zeros_like(rhs)
    exact_g = None if d is None else d * exact
    for k in range(1, max_steps+1):
        r = rhs - (x - jtv(x))
        new = x + alpha*r + beta*(x-prev)
        prev, x = x, new
        err = np.linalg.norm(x-exact)/(np.linalg.norm(exact)+1e-30)
        gerr = np.nan if d is None else np.linalg.norm(d*x-exact_g)/(np.linalg.norm(exact_g)+1e-30)
        if err <= 1e-6 and (d is None or gerr <= 1e-6): break
    return k, err, gerr


def solve_cheb(rhs, exact, jtv, mu, L, max_steps, d=None):
    center, half = (L+mu)/2, (L-mu)/2
    x = np.zeros_like(rhs); prev = np.zeros_like(rhs); alpha = 1/center
    exact_g = None if d is None else d * exact
    for k in range(1, max_steps+1):
        r = rhs - (x - jtv(x))
        if k == 1:
            new = x + alpha*r
        else:
            beta = (half*alpha/2)**2
            new_alpha = 1/(center - beta/alpha)
            new = x + new_alpha*r + beta*(x-prev)
            alpha = new_alpha
        prev, x = x, new
        err = np.linalg.norm(x-exact)/(np.linalg.norm(exact)+1e-30)
        gerr = np.nan if d is None else np.linalg.norm(d*x-exact_g)/(np.linalg.norm(exact_g)+1e-30)
        if err <= 1e-6 and (d is None or gerr <= 1e-6): break
    return k, err, gerr


def tanh_state(W, u):
    h = np.zeros_like(u)
    for _ in range(30000):
        new = np.tanh(W @ h + u)
        if np.linalg.norm(new-h) <= 1e-12*(1+np.linalg.norm(new)): return new
        h = new
    raise RuntimeError("tanh equilibrium did not converge")


def run_case(topology, problem, rho, seed, side, max_steps):
    asymmetric = topology == "asym_grid2d"
    W = graph(side, rho, asymmetric)
    n = W.shape[0]
    rng = np.random.default_rng(seed + 10000*asymmetric + 100000*(problem=="tanh"))
    if problem == "linear":
        d = np.ones(n)
        random_part = rng.normal(size=n); random_part /= np.linalg.norm(random_part) + 1e-30
        slow = np.ones(n) / np.sqrt(n)
        rhs = .85 * random_part + .15 * slow; rhs /= np.linalg.norm(rhs) + 1e-30
        exact = np.asarray(spsolve(eye(n, format="csr")-W, rhs))
        jtv = lambda v: np.asarray(W @ v)
        grad_weight = None
    else:
        u = 0.05*rng.normal(size=n)
        h = tanh_state(W, u); d = 1-h*h
        readout = rng.normal(size=n); readout /= np.linalg.norm(readout)
        rhs = (float(readout@h)-1.0)*readout
        exact = np.asarray(spsolve(eye(n, format="csr")-W@diags(d), rhs))
        jtv = lambda v: np.asarray(W @ (d*v))
        grad_weight = d
    audit = interval(W, d)
    rows = []
    for method in METHODS:
        oracle = method.endswith("oracle")
        mu = audit["mu_exact"] if oracle else audit["mu_bound"]
        L = audit["L_exact"] if oracle else audit["L_bound"]
        ar_a, cat_a, cat_b = coeff(mu, L)
        if method.startswith("AR"):
            steps, err, gerr = solve_stationary(rhs, exact, jtv, ar_a, 0, max_steps, grad_weight)
        elif method.startswith("CAT"):
            steps, err, gerr = solve_stationary(rhs, exact, jtv, cat_a, cat_b, max_steps, grad_weight)
        else:
            steps, err, gerr = solve_cheb(rhs, exact, jtv, mu, L, max_steps, grad_weight)
        rows.append({"topology":topology,"problem":problem,"rho":rho,"seed":seed,"method":method,
                     "jtv_actions":steps,"adjoint_error":err,"gradient_error":gerr,
                     "ar_alpha":ar_a,"mu":mu,"L":L,**audit})
    return rows


def summarize(runs):
    out=[]
    for key,g in runs.groupby(["topology","problem","rho","seed"]):
        m=g.set_index("method"); top,problem,rho,seed=key
        measure="jtv_actions"
        out.append({"topology":top,"problem":problem,"rho":rho,"seed":seed,
                    "cat_bound_vs_ar_oracle":m.loc["AR-oracle",measure]/m.loc["CAT-bound",measure],
                    "cat_oracle_vs_ar_oracle":m.loc["AR-oracle",measure]/m.loc["CAT-oracle",measure],
                    "cat_oracle_vs_cheb_oracle":m.loc["Cheb-oracle",measure]/m.loc["CAT-oracle",measure],
                    "rho_actual":float(g.rho_actual.mean()),
                    "ar_alpha_oracle":float(m.loc["AR-oracle","ar_alpha"]),
                    "ar_alpha_bound":float(m.loc["AR-bound","ar_alpha"])})
    return pd.DataFrame(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--profile",choices=("quick","standard","full"),default="quick")
    p.add_argument("--output",type=Path,default=Path("results/implicit_spectral"))
    p.add_argument("--fresh",action="store_true")
    args=p.parse_args(); side,rhos,seeds,max_steps=settings(args.profile)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for top in ("grid2d","asym_grid2d"):
        for problem in ("linear","tanh"):
            for rho in rhos:
                for seed in seeds:
                    rows.extend(run_case(top,problem,rho,seed,side,max_steps))
                    print(top,problem,rho,seed)
    runs=pd.DataFrame(rows); pairs=summarize(runs)
    runs.to_csv(args.output/"runs.csv",index=False)
    pairs.to_csv(args.output/"paired_speedups.csv",index=False)
    summary=pairs.groupby(["topology","problem","rho"],as_index=False).median(numeric_only=True)
    summary.to_csv(args.output/"summary.csv",index=False)
    audit=(runs.groupby(["topology","problem","rho"],as_index=False)[
        ["rho_actual","lambda_min","lambda_max","symmetry_error","ar_alpha"]].median())
    audit.to_csv(args.output/"asymmetry_audit.csv",index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
