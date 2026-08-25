#!/usr/bin/env python3
"""Trainable local recurrent control for the CAT implicit-adjoint study.

The degree-four symmetric recurrent edges are trained jointly with the input
map and classifier on moons, circles, and breast_cancer. AR and CAT use the
same initialization, minibatch order, contraction cap, spectral envelope, and
stopping tolerance. Standard uses 20 seeds per dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metric_wave.data import load_dataset_three_way

DATASETS = ("moons", "circles", "breast_cancer")
METHODS = ("AR-bound", "CAT-bound")


def coefficients(rho: float):
    mu, L = 1.0 - rho, 1.0 + rho
    sm, sL = np.sqrt(mu), np.sqrt(L)
    return 2.0 / (L + mu), 4.0 / (sL + sm) ** 2, ((sL - sm) / (sL + sm)) ** 2


def recur(v, edge1, edge2):
    return (edge1 * np.roll(v, -1, axis=-1) + np.roll(edge1, 1) * np.roll(v, 1, axis=-1)
            + edge2 * np.roll(v, -2, axis=-1) + np.roll(edge2, 2) * np.roll(v, 2, axis=-1))


def row_bound(edge1, edge2):
    row = np.abs(edge1) + np.roll(np.abs(edge1), 1) + np.abs(edge2) + np.roll(np.abs(edge2), 2)
    return float(np.max(row))


def project(edge1, edge2, rho):
    bound = row_bound(edge1, edge2)
    if bound > rho:
        edge1 *= rho / bound
        edge2 *= rho / bound


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, nin, hidden, nout, rho, seed):
        rng = np.random.default_rng(seed)
        self.rho, self.hidden = rho, hidden
        self.Win = 0.10 / np.sqrt(max(1, nin)) * rng.normal(size=(nin, hidden))
        self.bh = np.zeros(hidden)
        limit = np.sqrt(6.0 / (hidden + nout))
        self.Wout = rng.uniform(-limit, limit, size=(hidden, nout))
        self.bo = np.zeros(nout)
        self.edge1 = 0.30 * rho * (1.0 + 0.02 * rng.normal(size=hidden))
        self.edge2 = 0.20 * rho * (1.0 + 0.02 * rng.normal(size=hidden))
        project(self.edge1, self.edge2, rho)
        self.edge1_initial = self.edge1.copy()
        self.edge2_initial = self.edge2.copy()

    def blocks(self):
        return [x.copy() for x in (self.Win, self.bh, self.Wout, self.bo, self.edge1, self.edge2)]

    def restore(self, blocks):
        for target, source in zip((self.Win, self.bh, self.Wout, self.bo, self.edge1, self.edge2), blocks):
            target[...] = source

    def equilibrium(self, x, tol=1e-8, max_steps=2000):
        drive = x @ self.Win + self.bh
        h = np.zeros((len(x), self.hidden))
        for step in range(1, max_steps + 1):
            new = np.tanh(drive + recur(h, self.edge1, self.edge2))
            rel = np.linalg.norm(new - h) / (np.linalg.norm(new) + 1e-30)
            h = new
            if rel <= tol:
                return h, step
        return h, max_steps

    def evaluate(self, x, y):
        h, _ = self.equilibrium(x)
        z = h @ self.Wout + self.bo
        p = softmax(z)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), float(np.mean(z.argmax(1) == y))

    def recurrent_change(self):
        before = np.r_[self.edge1_initial, self.edge2_initial]
        after = np.r_[self.edge1, self.edge2]
        return float(np.linalg.norm(after - before) / (np.linalg.norm(before) + 1e-30))


class Momentum:
    def __init__(self, model, beta=0.9):
        self.beta = beta
        self.velocity = [np.zeros_like(x) for x in model.blocks()]

    def step(self, model, gradients, lr_main=0.04, lr_recurrent=0.005, weight_decay=1e-4):
        updated = []
        for i, (param, grad) in enumerate(zip(model.blocks(), gradients)):
            total = grad + (weight_decay * param if param.ndim > 1 else 0.0)
            self.velocity[i] = self.beta * self.velocity[i] + (1.0 - self.beta) * total
            lr = lr_recurrent if i >= 4 else lr_main
            updated.append(param - lr * self.velocity[i])
        model.restore(updated)
        project(model.edge1, model.edge2, model.rho)


def solve(rhs, d, model, method, tol=1e-5, max_actions=5000):
    ar, cat, beta = coefficients(model.rho)
    alpha, momentum = (ar, 0.0) if method == "AR-bound" else (cat, beta)
    lam = np.zeros_like(rhs)
    previous = np.zeros_like(rhs)
    norm = np.linalg.norm(rhs) + 1e-30
    for actions in range(1, max_actions + 1):
        residual = rhs - (lam - recur(lam * d, model.edge1, model.edge2))
        if np.linalg.norm(residual) / norm <= tol:
            return lam, actions
        new = lam + alpha * residual + momentum * (lam - previous)
        previous, lam = lam, new
    return lam, max_actions


def loss_grad(model, x, y, method):
    h, _ = model.equilibrium(x)
    z = h @ model.Wout + model.bo
    p = softmax(z)
    n = len(y)
    loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
    dz = p.copy(); dz[np.arange(n), y] -= 1.0; dz /= n
    d = 1.0 - h * h
    lam, actions = solve(dz @ model.Wout.T, d, model, method)
    force = lam * d
    grad_edge1 = np.sum(force * np.roll(h, -1, axis=-1) + np.roll(force, -1, axis=-1) * h, axis=0)
    grad_edge2 = np.sum(force * np.roll(h, -2, axis=-1) + np.roll(force, -2, axis=-1) * h, axis=0)
    grads = (x.T @ force, force.sum(0), h.T @ dz, dz.sum(0), grad_edge1, grad_edge2)
    return loss, grads, actions


def actual_rho(model, x, samples=8):
    h, _ = model.equilibrium(x[:min(samples, len(x))])
    W = recur(np.eye(model.hidden), model.edge1, model.edge2)
    values = []
    for state in h:
        d = 1.0 - state * state
        sd = np.sqrt(np.maximum(d, 1e-15))
        values.append(float(np.max(np.abs(np.linalg.eigvalsh(sd[:, None] * W * sd[None, :])))))
    return float(np.median(values))


def train(dataset, seed, method, epochs, hidden, patience, rho=0.95):
    xr, xv, xt, yr, yv, yt = load_dataset_three_way(dataset, seed=seed)
    classes = int(max(yr.max(), yv.max(), yt.max()) + 1)
    model = Model(xr.shape[1], hidden, classes, rho, seed + 1000)
    opt = Momentum(model)
    rng = np.random.default_rng(seed + 3000)
    best, best_blocks, stale, counts = np.inf, model.blocks(), 0, []
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(xr))
        for start in range(0, len(xr), 64):
            ii = order[start:start + 64]
            _, grads, actions = loss_grad(model, xr[ii], yr[ii], method)
            opt.step(model, grads)
            counts.append(actions)
        val_loss, _ = model.evaluate(xv, yv)
        if val_loss < best - 1e-4:
            best, best_blocks, best_epoch, stale = val_loss, model.blocks(), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.restore(best_blocks)
    test_loss, test_acc = model.evaluate(xt, yt)
    return {"dataset": dataset, "seed": seed, "method": method, "best_epoch": best_epoch,
            "test_loss": test_loss, "test_accuracy": test_acc, "mean_backward_jtv_actions": float(np.mean(counts)),
            "actual_rho_median": actual_rho(model, xt), "recurrent_change": model.recurrent_change()}


def bootstrap(values, draws=10000, seed=12345):
    rng = np.random.default_rng(seed); values = np.asarray(values)
    med = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(draws)]
    return np.quantile(med, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--output", type=Path, default=Path("results/trainable_recurrent"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.profile == "quick":
        datasets, seeds, epochs, hidden, patience = ("moons", "circles"), range(3), 15, 32, 4
    else:
        datasets, seeds, epochs, hidden, patience = DATASETS, range(20), 40, 48, 8
    rows = []
    for dataset in datasets:
        for seed in seeds:
            for method in METHODS:
                row = train(dataset, seed, method, epochs, hidden, patience)
                rows.append(row); print(dataset, seed, method, row["test_accuracy"], row["mean_backward_jtv_actions"])
    runs = pd.DataFrame(rows); runs.to_csv(args.output / "runs.csv", index=False)
    pairs = []
    for (dataset, seed), group in runs.groupby(["dataset", "seed"]):
        m = group.set_index("method"); ar, cat = m.loc["AR-bound"], m.loc["CAT-bound"]
        pairs.append({"dataset": dataset, "seed": seed, "delta_accuracy": cat.test_accuracy - ar.test_accuracy,
                      "training_jtv_speedup": ar.mean_backward_jtv_actions / cat.mean_backward_jtv_actions,
                      "cat_actual_rho": cat.actual_rho_median, "cat_recurrent_change": cat.recurrent_change})
    pairs = pd.DataFrame(pairs); pairs.to_csv(args.output / "pairs.csv", index=False)
    summary = []
    for dataset, group in pairs.groupby("dataset"):
        speed = group.training_jtv_speedup.to_numpy(); lo, hi = bootstrap(speed, 2000 if args.profile == "quick" else 10000)
        summary.append({"dataset": dataset, "n_pairs": len(group), "cat_wins": int(np.sum(speed > 1)),
                        "median_jtv_speedup": float(np.median(speed)), "speedup_ci_low": lo, "speedup_ci_high": hi,
                        "wilcoxon_p_greater_1": float(wilcoxon(speed - 1, alternative="greater").pvalue),
                        "mean_delta_accuracy": float(group.delta_accuracy.mean()), "max_abs_delta_accuracy": float(group.delta_accuracy.abs().max()),
                        "median_cat_actual_rho": float(group.cat_actual_rho.median()), "median_cat_recurrent_change": float(group.cat_recurrent_change.median())})
    speed = pairs.training_jtv_speedup.to_numpy(); lo, hi = bootstrap(speed, 2000 if args.profile == "quick" else 10000, 54321)
    rho_s, rho_p = spearmanr(pairs.cat_actual_rho, pairs.training_jtv_speedup)
    summary.append({"dataset": "ALL", "n_pairs": len(pairs), "cat_wins": int(np.sum(speed > 1)),
                    "median_jtv_speedup": float(np.median(speed)), "speedup_ci_low": lo, "speedup_ci_high": hi,
                    "wilcoxon_p_greater_1": float(wilcoxon(speed - 1, alternative="greater").pvalue),
                    "mean_delta_accuracy": float(pairs.delta_accuracy.mean()), "max_abs_delta_accuracy": float(pairs.delta_accuracy.abs().max()),
                    "median_cat_actual_rho": float(pairs.cat_actual_rho.median()), "median_cat_recurrent_change": float(pairs.cat_recurrent_change.median()),
                    "spearman_rho_speedup_vs_rho": float(rho_s), "spearman_p_speedup_vs_rho": float(rho_p)})
    summary = pd.DataFrame(summary); summary.to_csv(args.output / "summary.csv", index=False); print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
