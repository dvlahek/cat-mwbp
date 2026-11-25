#!/usr/bin/env python3
"""Width scaling and finite-tolerance experiments for structured recurrent networks."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

# Works when placed into the existing experiments/ directory.
from run_trainable_recurrent_control import train
from run_implicit_spectral_suite import graph, tanh_state, coeff
from metric_wave.data import load_dataset_three_way


def csv_save(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def learned_parameter_count(dataset: str, seed: int, hidden: int) -> int:
    xr, _xv, _xt, yr, yv, yt = load_dataset_three_way(dataset, seed=seed)
    nin = int(xr.shape[1])
    nout = int(max(yr.max(), yv.max(), yt.max()) + 1)
    # Win, bh, Wout, bo, and the TWO trainable recurrent edge vectors.
    return nin * hidden + hidden + hidden * nout + nout + 2 * hidden


def width_experiment(args) -> None:
    datasets = args.datasets.split(",")
    widths = [int(x) for x in args.widths.split(",")]
    rows = []
    for dataset in datasets:
        for width in widths:
            for seed in range(args.seeds):
                nparams = learned_parameter_count(dataset, seed, width)
                for method in ("AR-bound", "CAT-bound"):
                    started = time.perf_counter()
                    result = train(dataset, seed, method, args.epochs, width,
                                   args.patience, rho=args.rho)
                    elapsed = time.perf_counter() - started
                    rows.append({
                        "dataset": dataset, "hidden": width,
                        "trainable_parameters": nparams,
                        "seed": seed, "method": method,
                        "rho_cap": args.rho, "epochs_cap": args.epochs,
                        "test_accuracy": result["test_accuracy"],
                        "test_loss": result["test_loss"],
                        "mean_backward_jtv_actions": result["mean_backward_jtv_actions"],
                        "actual_rho_median": result["actual_rho_median"],
                        "recurrent_change": result["recurrent_change"],
                        "elapsed_run_s_including_audit": elapsed,
                        "best_epoch": result["best_epoch"],
                    })
                    print("WIDTH", dataset, width, seed, method,
                          "parameters", nparams, "accuracy", result["test_accuracy"],
                          "actions", result["mean_backward_jtv_actions"], flush=True)
                    csv_save(args.out / "width_runs.csv", rows)
    pairs = []
    for dataset in datasets:
        for width in widths:
            for seed in range(args.seeds):
                group = [r for r in rows if r["dataset"] == dataset
                         and r["hidden"] == width and r["seed"] == seed]
                m = {r["method"]: r for r in group}
                ar, cat = m["AR-bound"], m["CAT-bound"]
                pairs.append({
                    "dataset": dataset, "hidden": width,
                    "trainable_parameters": ar["trainable_parameters"],
                    "seed": seed,
                    "ar_test_accuracy": ar["test_accuracy"],
                    "cat_test_accuracy": cat["test_accuracy"],
                    "delta_accuracy": cat["test_accuracy"] - ar["test_accuracy"],
                    "action_ratio_ar_over_cat":
                        ar["mean_backward_jtv_actions"] / cat["mean_backward_jtv_actions"],
                    "run_time_ratio_ar_over_cat_including_audit":
                        ar["elapsed_run_s_including_audit"] / max(cat["elapsed_run_s_including_audit"], 1e-12),
                    "ar_backward_actions": ar["mean_backward_jtv_actions"],
                    "cat_backward_actions": cat["mean_backward_jtv_actions"],
                })
    csv_save(args.out / "width_pairs.csv", pairs)
    plot_width(pairs, args.out)


def plot_width(pairs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in pairs})
    for dataset in datasets:
        drows = [r for r in pairs if r["dataset"] == dataset]
        xs = sorted({r["trainable_parameters"] for r in drows})
        mean = lambda key, n: float(np.mean([r[key] for r in drows if r["trainable_parameters"] == n]))
        med = lambda key, n: float(np.median([r[key] for r in drows if r["trainable_parameters"] == n]))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        axes[0].plot(xs, [mean("ar_backward_actions", n) for n in xs], "o-", label="AR")
        axes[0].plot(xs, [mean("cat_backward_actions", n) for n in xs], "s-", label="CAT")
        axes[0].set_ylabel("Mean local JTv actions per backward solve")
        axes[0].legend()
        axes[1].plot(xs, [mean("ar_test_accuracy", n) for n in xs], "o-", label="AR")
        axes[1].plot(xs, [mean("cat_test_accuracy", n) for n in xs], "s-", label="CAT")
        axes[1].set_ylabel("Held-out test accuracy")
        axes[1].legend()
        axes[2].plot(xs, [med("action_ratio_ar_over_cat", n) for n in xs], "o-")
        axes[2].axhline(1, linewidth=0.8, linestyle="--", color="gray")
        axes[2].set_ylabel("Median AR / CAT local-action ratio")
        for a in axes:
            a.set_xscale("log")
            a.set_xlabel("Number of learned parameters (log scale)")
            a.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"width_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(out / f"width_{dataset}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def make_spectral_case(side: int, rho: float, seed: int, problem: str):
    W = graph(side, rho, True)
    n = W.shape[0]
    rng = np.random.default_rng(seed + 100000 * (problem == "tanh"))
    if problem == "linear":
        rhs = rng.normal(size=n)
        rhs /= np.linalg.norm(rhs) + 1e-30
        d = np.ones(n)
    else:
        u = .05 * rng.normal(size=n)
        h = tanh_state(W, u)
        d = 1 - h * h
        rhs = rng.normal(size=n)
        rhs /= np.linalg.norm(rhs) + 1e-30
    A = eye(n, format="csr") - W @ diags(d)
    exact = np.asarray(spsolve(A, rhs))
    gexact = d * exact
    jtv = lambda v: np.asarray(W @ (d * v))
    # Conservative, configured row-sum bound (no eigensolve).
    mu, L = 1 - rho, 1 + rho
    return rhs, exact, gexact, d, jtv, mu, L


def solve_to_tolerance(rhs, exact, gexact, d, jtv, alpha, beta,
                       tolerance: float, max_actions: int):
    x = np.zeros_like(rhs)
    previous = np.zeros_like(rhs)
    # Exact reference checked offline; no uncharged JTv in the stopping test.
    en = np.linalg.norm(exact) + 1e-30
    gn = np.linalg.norm(gexact) + 1e-30
    err = np.linalg.norm(x - exact) / en
    gerr = np.linalg.norm(d * x - gexact) / gn
    if max(err, gerr) <= tolerance:
        return 0, err, gerr, True
    for k in range(1, max_actions + 1):
        residual = rhs - (x - jtv(x))
        xn = x + alpha * residual + beta * (x - previous)
        previous, x = x, xn
        err = np.linalg.norm(x - exact) / en
        gerr = np.linalg.norm(d * x - gexact) / gn
        if max(err, gerr) <= tolerance:
            return k, err, gerr, True
        if not np.isfinite(err):
            break
    return k, err, gerr, False


def tolerance_experiment(args) -> None:
    rows = []
    tolerances = [float(t) for t in args.tolerances.split(",")]
    rhos = [float(x) for x in args.rhos.split(",")]
    for problem in ("linear", "tanh"):
        for rho in rhos:
            for seed in range(args.seeds):
                rhs, exact, gexact, d, jtv, mu, L = make_spectral_case(
                    args.side, rho, seed, problem)
                ar_alpha, cat_alpha, cat_beta = coeff(mu, L)
                for tol in tolerances:
                    for method, alpha, beta in (
                        ("AR-bound", ar_alpha, 0.0),
                        ("CAT-bound", cat_alpha, cat_beta),
                    ):
                        steps, err, gerr, converged = solve_to_tolerance(
                            rhs, exact, gexact, d, jtv, alpha, beta,
                            tol, args.max_actions)
                        rows.append({
                            "problem": problem, "side": args.side,
                            "state_dimension": args.side ** 2,
                            "rho_cap": rho, "seed": seed, "tolerance": tol,
                            "method": method, "actions": steps,
                            "adjoint_relative_error": err,
                            "gradient_relative_error": gerr,
                            "converged": int(converged),
                        })
                print("TOL", problem, rho, seed, flush=True)
                csv_save(args.out / "tolerance_runs.csv", rows)
    pairs = []
    for problem in ("linear", "tanh"):
        for rho in rhos:
            for seed in range(args.seeds):
                for tol in tolerances:
                    sub = [r for r in rows if (r["problem"], r["rho_cap"], r["seed"], r["tolerance"])
                           == (problem, rho, seed, tol)]
                    m = {r["method"]: r for r in sub}
                    ar, cat = m["AR-bound"], m["CAT-bound"]
                    pairs.append({
                        "problem": problem, "rho_cap": rho, "seed": seed,
                        "tolerance": tol,
                        "ar_actions": ar["actions"], "cat_actions": cat["actions"],
                        "both_converged": int(ar["converged"] and cat["converged"]),
                        "ar_over_cat_ratio": (ar["actions"] / cat["actions"]
                                              if ar["converged"] and cat["converged"] and cat["actions"] else float("nan")),
                    })
    csv_save(args.out / "tolerance_pairs.csv", pairs)
    plot_tolerance(pairs, args.out)


def plot_tolerance(pairs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for (problem, rho) in sorted({(r["problem"], r["rho_cap"]) for r in pairs}):
        rows = [r for r in pairs if r["problem"] == problem and r["rho_cap"] == rho
                and r["both_converged"]]
        if not rows:
            continue
        xs = sorted({r["tolerance"] for r in rows})
        ys = [np.median([r["ar_over_cat_ratio"] for r in rows if r["tolerance"] == t]) for t in xs]
        ax.plot(xs, ys, marker="o", label=f"{problem} rho={rho:g}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Exact relative-error tolerance (offline audit)")
    ax.set_ylabel("Median AR / CAT local-action ratio")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "tolerance_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(out / "tolerance_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=("width", "tolerance"), required=True)
    p.add_argument("--out", type=Path, default=Path("results/revision_scaling"))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--widths", default="32,64,128,256")
    p.add_argument("--datasets", default="breast_cancer,digits,synthetic_large")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--rho", type=float, default=.95)
    p.add_argument("--tolerances", default="1e-3,1e-5,1e-6,1e-7")
    p.add_argument("--rhos", default="0.8,0.95,0.99")
    p.add_argument("--side", type=int, default=12)
    p.add_argument("--max-actions", type=int, default=12000)
    args = p.parse_args()
    if args.seeds < 1 or args.side < 3 or not (0 < args.rho < 1):
        p.error("Require seeds>=1, side>=3, and 0<rho<1")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.study == "width":
        width_experiment(args)
    else:
        tolerance_experiment(args)


if __name__ == "__main__":
    main()
