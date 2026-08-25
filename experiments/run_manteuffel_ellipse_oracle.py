#!/usr/bin/env python3
"""Oracle elliptic spectral control for the directed CAT recurrence.

The experiment uses the translation-invariant directed local operator at
rho=0.95. It compares the original real-interval CAT coefficients with
coefficients obtained from an oracle ellipse enclosing the exact complex
spectrum of A = I - J^T. A direct-spectrum oracle is included as a check on
ellipse conservatism.

All reported action counts stop at relative adjoint error 1e-6. The exact
spectrum is used only for this diagnostic control.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

TOL = 1e-6


def directed_matrix(n: int, rho: float) -> np.ndarray:
    W = np.zeros((n, n), dtype=float)
    weights = {1: .42*rho, -1: .18*rho, 2: .28*rho, -2: .12*rho}
    for i in range(n):
        for offset, weight in weights.items():
            W[i, (i + offset) % n] = weight
    return W


def fit_ellipse(eig: np.ndarray) -> dict:
    x = eig.real
    y = eig.imag
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 1e-6)

    def unpack(z):
        center = float(z[0])
        ratio = float(np.exp(z[1]))
        b = float(np.sqrt(np.max(((x-center)/ratio)**2 + y*y)))
        a = ratio*b
        return center, a, b

    def objective(z):
        _, a, b = unpack(z)
        return float(np.log(a*b + 1e-30))

    result = differential_evolution(
        objective,
        [(xmin-.5*span, xmax+.5*span), (-4., 4.)],
        seed=1234,
        tol=1e-11,
        polish=True,
        workers=1,
    )
    local = minimize(objective, result.x, method="Nelder-Mead")
    z = local.x if local.fun <= result.fun else result.x
    center, a, b = unpack(z)
    a *= 1. + 1e-9
    b *= 1. + 1e-9
    return {"center": center, "a": a, "b": b}


def ellipse_boundary(ellipse: dict, points: int) -> np.ndarray:
    theta = np.linspace(0., 2.*np.pi, points, endpoint=False)
    return (
        ellipse["center"]
        + ellipse["a"]*np.cos(theta)
        + 1j*ellipse["b"]*np.sin(theta)
    )


def root_radius(lam: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    trace = 1. + beta - alpha*lam
    disc = np.sqrt((trace*trace - 4.*beta).astype(complex))
    q1 = .5*(trace + disc)
    q2 = .5*(trace - disc)
    return np.maximum(np.abs(q1), np.abs(q2))


def worst_radius(region: np.ndarray, alpha: float, beta: float) -> float:
    if alpha <= 0. or not (0. <= beta < 1.):
        return 1e6
    value = root_radius(region, alpha, beta)
    return float(np.max(value)) if np.all(np.isfinite(value)) else 1e6


def optimize_cat(region: np.ndarray, seed: int) -> tuple[float, float, float]:
    scale = max(float(np.max(np.abs(region))), 1e-6)

    def objective(z):
        return worst_radius(region, float(z[0]), float(z[1]))

    result = differential_evolution(
        objective,
        [(1e-6, max(2., 12./scale)), (0., .999)],
        seed=seed,
        tol=1e-10,
        popsize=20,
        polish=True,
        workers=1,
    )
    local = minimize(objective, result.x, method="Nelder-Mead")
    z = local.x if local.fun <= result.fun else result.x
    return float(z[0]), float(z[1]), float(objective(z))


def interval_cat(rho: float) -> tuple[float, float]:
    mu, L = 1.-rho, 1.+rho
    sm, sL = math.sqrt(mu), math.sqrt(L)
    alpha = 4./(sL+sm)**2
    beta = ((sL-sm)/(sL+sm))**2
    return alpha, beta


def solve(A, rhs, exact, alpha, beta, max_actions=20000, history=False):
    x = np.zeros_like(rhs)
    previous = np.zeros_like(rhs)
    norm = np.linalg.norm(exact) + 1e-30
    rows = []

    for actions in range(max_actions+1):
        error = float(np.linalg.norm(x-exact)/norm)
        if history:
            rows.append((actions, error))
        if error <= TOL:
            return actions, error, True, pd.DataFrame(rows, columns=["actions","error"])
        if actions == max_actions:
            break
        residual = rhs - A@x
        new = x + alpha*residual + beta*(x-previous)
        previous, x = x, new
        if not np.all(np.isfinite(x)) or error > 1e12:
            break

    error = float(np.linalg.norm(x-exact)/norm) if np.all(np.isfinite(x)) else float("inf")
    return actions, error, False, pd.DataFrame(rows, columns=["actions","error"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/manteuffel_ellipse_oracle"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=144)
    parser.add_argument("--rho", type=float, default=.95)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    W = directed_matrix(args.n, args.rho)
    A = np.eye(args.n) - W.T
    eig = np.linalg.eigvals(A)
    ellipse = fit_ellipse(eig)
    boundary = ellipse_boundary(ellipse, 8192)

    ar, br = interval_cat(args.rho)
    ae, be, re = optimize_cat(boundary, 2101)
    ad, bd, rd = optimize_cat(eig, 2102)

    params = pd.DataFrame([
        {"method":"CAT-real-interval","alpha":ar,"beta":br,
         "worst_root_on_actual_spectrum":worst_radius(eig, ar, br)},
        {"method":"CAT-ellipse-oracle","alpha":ae,"beta":be,
         "worst_root_on_actual_spectrum":worst_radius(eig, ae, be)},
        {"method":"CAT-spectrum-oracle","alpha":ad,"beta":bd,
         "worst_root_on_actual_spectrum":worst_radius(eig, ad, bd)},
    ])
    params.to_csv(args.output/"oracle_parameters.csv", index=False)

    methods = {r.method:(r.alpha,r.beta) for r in params.itertuples()}
    runs = []
    histories = []
    slow = np.ones(args.n)
    slow /= np.linalg.norm(slow)

    for seed in range(args.seeds):
        rng = np.random.default_rng(73001+seed)
        rhs = .85*rng.normal(size=args.n) + .15*slow
        rhs /= np.linalg.norm(rhs) + 1e-30
        exact = np.linalg.solve(A, rhs)

        for method, (alpha, beta) in methods.items():
            actions, error, converged, h = solve(
                A, rhs, exact, alpha, beta, history=(seed == 0)
            )
            runs.append({"seed":seed,"method":method,"jtv_actions":actions,
                         "relative_adjoint_error":error,"converged":converged})
            if seed == 0:
                h["method"] = method
                histories.append(h)

    runs = pd.DataFrame(runs)
    runs.to_csv(args.output/"runs.csv", index=False)
    summary = runs.groupby("method", as_index=False).agg(
        n_seeds=("seed","count"), converged=("converged","sum"),
        median_actions=("jtv_actions","median"), min_actions=("jtv_actions","min"),
        max_actions=("jtv_actions","max"), max_final_error=("relative_adjoint_error","max")
    ).merge(params, on="method")
    summary.to_csv(args.output/"summary.csv", index=False)

    audit = {
        "rho": args.rho,
        "n": args.n,
        "ellipse_center": ellipse["center"],
        "ellipse_real_semiaxis": ellipse["a"],
        "ellipse_imag_semiaxis": ellipse["b"],
        "spectrum_max_abs_imag": float(np.max(np.abs(eig.imag))),
        "matrix_asymmetry": float(np.linalg.norm(W-W.T,"fro")/(np.linalg.norm(W,"fro")+1e-30)),
        "matrix_nonnormality": float(np.linalg.norm(W.T@W-W@W.T,"fro")/(np.linalg.norm(W,"fro")**2+1e-30)),
    }
    (args.output/"ellipse_audit.json").write_text(json.dumps(audit, indent=2))

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(args.output/"convergence_history_seed0.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,5))
    for method, group in history.groupby("method"):
        group = group[np.isfinite(group.error) & (group.error > 0)]
        ax.plot(group.actions, group.error, label=method)
    ax.axhline(TOL, label="target 1e-6")
    ax.set_yscale("log")
    ax.set_xlabel(r"$J^\top v$ actions")
    ax.set_ylabel("Relative adjoint error")
    ax.set_title("Directed complex-spectrum CAT with oracle spectral tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/"ellipse_oracle_convergence.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
