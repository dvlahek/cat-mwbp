#!/usr/bin/env python3
"""Directed-spectrum scope control for the CAT implicit-adjoint study.

The experiment uses local degree-four periodic operators outside the symmetric
or diagonally symmetrizable setting of the main spectral analysis. CAT keeps
the same real-interval coefficients used for the symmetric problem. The run is
therefore a scope test, not an acceleration claim for complex spectra.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, gmres


def coefficients(rho: float):
    mu, L = 1.0 - rho, 1.0 + rho
    sm, sL = np.sqrt(mu), np.sqrt(L)
    return 2.0 / (L + mu), 4.0 / (sL + sm) ** 2, ((sL - sm) / (sL + sm)) ** 2


def directed_matrix(n: int, rho: float, heterogeneous: bool, seed: int) -> np.ndarray:
    W = np.zeros((n, n), dtype=float)
    base = np.array([0.42, 0.18, 0.28, 0.12])
    rng = np.random.default_rng(seed + 9107)
    for i in range(n):
        weights = base.copy()
        if heterogeneous:
            weights *= np.exp(0.60 * rng.normal(size=4))
        weights = rho * weights / weights.sum()
        for offset, weight in zip((1, -1, 2, -2), weights):
            W[i, (i + offset) % n] = weight
    return W


def nonnormality(A: np.ndarray) -> float:
    comm = A.T @ A - A @ A.T
    return float(np.linalg.norm(comm, "fro") / (np.linalg.norm(A, "fro") ** 2 + 1e-30))


def tanh_state(W: np.ndarray, u: np.ndarray) -> np.ndarray:
    h = np.zeros_like(u)
    for _ in range(30000):
        new = np.tanh(W @ h + u)
        if np.linalg.norm(new - h) <= 1e-12 * (1.0 + np.linalg.norm(new)):
            return new
        h = new
    raise RuntimeError("tanh equilibrium did not converge")


def make_case(problem: str, rho: float, seed: int, n: int, heterogeneous: bool):
    W = directed_matrix(n, rho, heterogeneous, seed)
    rng = np.random.default_rng(seed + 21001 + 100000 * (problem == "tanh"))
    if problem == "linear":
        rhs = rng.normal(size=n)
        rhs /= np.linalg.norm(rhs) + 1e-30
        operator = W
        exact = np.linalg.solve(np.eye(n) - W.T, rhs)
        jtv = lambda v: W.T @ v
    else:
        u = 0.05 * rng.normal(size=n)
        h = tanh_state(W, u)
        d = 1.0 - h * h
        readout = rng.normal(size=n)
        readout /= np.linalg.norm(readout) + 1e-30
        rhs = (float(readout @ h) - 1.0) * readout
        operator = np.diag(d) @ W
        exact = np.linalg.solve(np.eye(n) - operator.T, rhs)
        jtv = lambda v: W.T @ (d * v)
    eig = np.linalg.eigvals(operator)
    audit = {
        "spectral_radius": float(np.max(np.abs(eig))),
        "max_imag_eigenvalue": float(np.max(np.abs(eig.imag))),
        "nonnormality": nonnormality(operator),
        "matrix_asymmetry": float(np.linalg.norm(operator - operator.T, "fro") / (np.linalg.norm(operator, "fro") + 1e-30)),
    }
    return rhs, exact, jtv, audit


def stationary(rhs, jtv, alpha, beta, tol=1e-6, max_actions=15000):
    x = np.zeros_like(rhs)
    previous = np.zeros_like(rhs)
    rhs_norm = np.linalg.norm(rhs) + 1e-30
    rel = np.inf
    for actions in range(1, max_actions + 1):
        residual = rhs - (x - jtv(x))
        rel = np.linalg.norm(residual) / rhs_norm
        if rel <= tol:
            return actions, rel
        new = x + alpha * residual + beta * (x - previous)
        previous, x = x, new
        if not np.all(np.isfinite(x)):
            break
    return actions, float(rel)


def gmres_actions(rhs, jtv, tol=1e-6):
    count = {"n": 0}
    def matvec(v):
        count["n"] += 1
        return v - jtv(v)
    A = LinearOperator((len(rhs), len(rhs)), matvec=matvec, dtype=float)
    try:
        _, info = gmres(A, rhs, rtol=tol, atol=0.0, restart=min(len(rhs), 200))
    except TypeError:
        _, info = gmres(A, rhs, tol=tol, restart=min(len(rhs), 200))
    return count["n"], info == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--output", type=Path, default=Path("results/directed_spectrum"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.profile == "quick":
        n, rhos, seeds = 64, (0.80, 0.95), (0, 1)
    else:
        n, rhos, seeds = 144, (0.80, 0.95, 0.99), tuple(range(5))

    rows = []
    for heterogeneous in (False, True):
        topology = "directed_uniform" if not heterogeneous else "directed_heterogeneous_nonnormal"
        for problem in ("linear", "tanh"):
            for rho in rhos:
                ar_alpha, cat_alpha, cat_beta = coefficients(rho)
                for seed in seeds:
                    rhs, exact, jtv, audit = make_case(problem, rho, seed, n, heterogeneous)
                    for method, alpha, beta in (
                        ("AR-realbound", ar_alpha, 0.0),
                        ("CAT-realbound", cat_alpha, cat_beta),
                    ):
                        actions, residual = stationary(rhs, jtv, alpha, beta)
                        rows.append({"topology": topology, "problem": problem, "rho": rho, "seed": seed, "method": method,
                                     "jtv_actions": actions, "relative_residual": residual, "converged": residual <= 1e-6, **audit})
                    actions, converged = gmres_actions(rhs, jtv)
                    rows.append({"topology": topology, "problem": problem, "rho": rho, "seed": seed, "method": "GMRES",
                                 "jtv_actions": actions, "relative_residual": np.nan, "converged": converged, **audit})
                    print(topology, problem, rho, seed)

    runs = pd.DataFrame(rows)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary = runs.groupby(["topology", "problem", "rho", "method"], as_index=False).agg(
        median_actions=("jtv_actions", "median"), min_actions=("jtv_actions", "min"), max_actions=("jtv_actions", "max"),
        convergence_rate=("converged", "mean"), spectral_radius=("spectral_radius", "median"),
        max_imag_eigenvalue=("max_imag_eigenvalue", "median"), nonnormality=("nonnormality", "median"),
        matrix_asymmetry=("matrix_asymmetry", "median"))
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
