#!/usr/bin/env python3
"""Oracle complex-spectrum diagnostics for the directed CAT recurrence.

The script reproduces the manuscript controls at rho=0.95 for:
  1. a translation-invariant directed normal operator, and
  2. five heterogeneous directed non-normal operators.

For each operator, CAT is evaluated with the original real-interval
coefficients, coefficients optimized on an oracle ellipse enclosing the exact
complex spectrum of A = I - J^T, and coefficients optimized directly on the
exact discrete spectrum. The recurrence itself is unchanged.

All action counts stop at relative adjoint error 1e-6. The spectral enclosure
is oracle information and is used only as a diagnostic.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

TOL = 1e-6


def directed_matrix(n: int, rho: float, operator_seed: int | None) -> np.ndarray:
    W = np.zeros((n, n), dtype=float)
    base = np.array([0.42, 0.18, 0.28, 0.12], dtype=float)
    rng = None if operator_seed is None else np.random.default_rng(operator_seed + 9107)
    for i in range(n):
        weights = base.copy()
        if rng is not None:
            weights *= np.exp(0.60 * rng.normal(size=4))
        weights = rho * weights / weights.sum()
        for offset, weight in zip((1, -1, 2, -2), weights):
            W[i, (i + offset) % n] = weight
    return W


def nonnormality(W: np.ndarray) -> float:
    comm = W.T @ W - W @ W.T
    return float(np.linalg.norm(comm, "fro") / (np.linalg.norm(W, "fro")**2 + 1e-30))


def fit_ellipse(eig: np.ndarray, seed: int) -> dict:
    x, y = eig.real, eig.imag
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 1e-6)
    def unpack(z):
        center = float(z[0]); ratio = float(np.exp(z[1]))
        b = float(np.sqrt(np.max(((x-center)/ratio)**2 + y*y)))
        return center, ratio*b, b
    def objective(z):
        _, a, b = unpack(z)
        return float(np.log(a*b + 1e-30))
    result = differential_evolution(objective, [(xmin-.5*span, xmax+.5*span), (-4.,4.)], seed=seed, tol=1e-11, polish=True, workers=1)
    local = minimize(objective, result.x, method="Nelder-Mead")
    z = local.x if local.fun <= result.fun else result.x
    center, a, b = unpack(z)
    return {"center":center, "a":a*(1.+1e-9), "b":b*(1.+1e-9)}


def ellipse_boundary(e: dict, points: int = 4096) -> np.ndarray:
    t = np.linspace(0., 2.*np.pi, points, endpoint=False)
    return e["center"] + e["a"]*np.cos(t) + 1j*e["b"]*np.sin(t)


def root_radius(lam: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    trace = 1. + beta - alpha*lam
    disc = np.sqrt((trace*trace - 4.*beta).astype(complex))
    return np.maximum(np.abs(.5*(trace+disc)), np.abs(.5*(trace-disc)))


def worst_radius(region: np.ndarray, alpha: float, beta: float) -> float:
    if alpha <= 0. or not (0. <= beta < 1.): return 1e6
    return float(np.max(root_radius(region, alpha, beta)))


def optimize_cat(region: np.ndarray, seed: int) -> tuple[float,float,float]:
    scale = max(float(np.max(np.abs(region))), 1e-6)
    def objective(z): return worst_radius(region, float(z[0]), float(z[1]))
    result = differential_evolution(objective, [(1e-6,max(2.,12./scale)), (0.,.999)], seed=seed, tol=1e-10, popsize=20, polish=True, workers=1)
    local = minimize(objective, result.x, method="Nelder-Mead")
    z = local.x if local.fun <= result.fun else result.x
    return float(z[0]), float(z[1]), float(objective(z))


def interval_cat(rho: float) -> tuple[float,float]:
    mu, L = 1.-rho, 1.+rho
    sm, sL = math.sqrt(mu), math.sqrt(L)
    return 4./(sL+sm)**2, ((sL-sm)/(sL+sm))**2


def solve(A, rhs, exact, alpha, beta, max_actions=20000):
    x = np.zeros_like(rhs); previous = np.zeros_like(rhs)
    norm = np.linalg.norm(exact) + 1e-30
    for actions in range(max_actions+1):
        error = float(np.linalg.norm(x-exact)/norm)
        if error <= TOL: return actions, error, True
        if actions == max_actions or not np.isfinite(error) or error > 1e12: break
        residual = rhs - A@x
        new = x + alpha*residual + beta*(x-previous)
        previous, x = x, new
    error = float(np.linalg.norm(x-exact)/norm) if np.all(np.isfinite(x)) else float("inf")
    return max_actions, error, False


def evaluate_operator(W, label, operator_seed, rho, rhs_seeds, output_rows, parameter_rows):
    A = np.eye(W.shape[0]) - W.T
    eig = np.linalg.eigvals(A)
    ellipse = fit_ellipse(eig, 1234 + (0 if operator_seed is None else operator_seed))
    boundary = ellipse_boundary(ellipse)
    ar, br = interval_cat(rho)
    ae, be, _ = optimize_cat(boundary, 2100 + (0 if operator_seed is None else operator_seed))
    ad, bd, _ = optimize_cat(eig, 2300 + (0 if operator_seed is None else operator_seed))
    methods = {"real_interval":(ar,br), "ellipse_oracle":(ae,be), "spectrum_oracle":(ad,bd)}
    for method,(alpha,beta) in methods.items():
        parameter_rows.append({"operator":label,"operator_seed":operator_seed,"method":method,"alpha":alpha,"beta":beta,"root_factor":worst_radius(eig,alpha,beta),"nonnormality":nonnormality(W),"max_abs_imag":float(np.max(np.abs(eig.imag)))})
    for rhs_seed in rhs_seeds:
        rng = np.random.default_rng(rhs_seed)
        rhs = rng.normal(size=W.shape[0]); rhs /= np.linalg.norm(rhs)+1e-30
        exact = np.linalg.solve(A, rhs)
        for method,(alpha,beta) in methods.items():
            actions,error,converged = solve(A,rhs,exact,alpha,beta)
            output_rows.append({"operator":label,"operator_seed":operator_seed,"rhs_seed":rhs_seed,"method":method,"actions":actions,"error":error,"converged":converged})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/complex_spectrum_oracle"))
    p.add_argument("--rho", type=float, default=.95)
    p.add_argument("--n", type=int, default=144)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    runs=[]; pars=[]
    evaluate_operator(directed_matrix(args.n,args.rho,None), "directed_normal", None, args.rho, [73001+i for i in range(20)], runs, pars)
    for opseed in range(5):
        evaluate_operator(directed_matrix(args.n,args.rho,opseed), "heterogeneous_non_normal", opseed, args.rho, [83000+100*opseed+i for i in range(4)], runs, pars)
    runs=pd.DataFrame(runs); pars=pd.DataFrame(pars)
    runs.to_csv(args.output/"runs.csv",index=False); pars.to_csv(args.output/"operator_parameters.csv",index=False)
    summary=runs.groupby(["operator","method"],as_index=False).agg(cases=("converged","size"),converged=("converged","sum"),median_actions=("actions","median"),min_actions=("actions","min"),max_actions=("actions","max"),max_error=("error","max"))
    roots=pars.groupby(["operator","method"],as_index=False).agg(root_factor_min=("root_factor","min"),root_factor_max=("root_factor","max"),nonnormality_min=("nonnormality","min"),nonnormality_max=("nonnormality","max"))
    summary=summary.merge(roots,on=["operator","method"])
    summary.to_csv(args.output/"summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__": main()
