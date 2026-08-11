#!/usr/bin/env python3
"""Optional real-data vision probe, separate from the non-vision suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metric_wave.data_vision import raw_vision_dataset, vision_widths
from metric_wave.model import MLP
from metric_wave.riemannian_metric import OutputFactorTransportPullback, RiemannianPullback
from metric_wave.training import evaluate


def split_scale(x, y, seed):
    xdev, xte, ydev, yte = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    xtr, xva, ytr, yva = train_test_split(
        xdev, ydev, test_size=0.20, random_state=seed + 104729, stratify=ydev
    )
    scaler = StandardScaler().fit(xtr)
    return scaler.transform(xtr), scaler.transform(xva), scaler.transform(xte), ytr, yva, yte


def make_optimizer(method, model, seed, metric_batch):
    common = dict(
        model=model,
        lr=0.05,
        mass=1.0,
        output_metric="gauss_newton",
        momentum=0.9,
        metric_batch=metric_batch,
        seed=seed,
    )
    diameter = model.n_layers - 1
    if method == "instantaneous":
        return RiemannianPullback(**common)
    if method == "local_output_only":
        return OutputFactorTransportPullback(
            **common, metric_steps=diameter, wave_speed=0.0, relax_rate=0.6
        )
    if method == "partial_3hop":
        return OutputFactorTransportPullback(
            **common, metric_steps=3, wave_speed=1.0, relax_rate=0.6
        )
    if method == "full_reach":
        return OutputFactorTransportPullback(
            **common, metric_steps=None, wave_speed=1.0, relax_rate=0.6,
            require_full_reach=True,
        )
    if method == "relaxed_10hop":
        return OutputFactorTransportPullback(
            **common, metric_steps=10, wave_speed=1.0, relax_rate=0.6,
            require_full_reach=True,
        )
    raise ValueError(method)


def train(method, dataset, x, y, seed, epochs, batch_size, metric_batch):
    xtr, xva, xte, ytr, yva, yte = split_scale(x, y, seed)
    model = MLP(vision_widths(dataset), seed=seed)
    optimizer = make_optimizer(method, model, seed, metric_batch)
    rng = np.random.default_rng(seed)
    best_loss, best_blocks = float("inf"), model.parameter_blocks()
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        for start in range(0, xtr.shape[0], batch_size):
            order = rng.permutation(xtr.shape[0]) if start == 0 else order
            index = order[start : start + batch_size]
            _, gradients = model.loss_and_gradients(xtr[index], ytr[index])
            optimizer.set_metric_batch(xtr[index], ytr[index])
            model.set_parameter_blocks(optimizer.step(model.parameter_blocks(), gradients))
        value = model.loss(xva, yva)
        if value < best_loss - 1e-4:
            best_loss = value
            best_epoch = epoch
            best_blocks = model.parameter_blocks()
    model.set_parameter_blocks(best_blocks)
    result = evaluate(model, xte, yte)
    row = {
        "dataset": dataset,
        "real_data": True,
        "seed": seed,
        "method": method,
        "best_epoch": best_epoch,
        "test_accuracy": result["accuracy"],
        "test_loss": result["loss"],
    }
    row.update(optimizer.diagnostics())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=8000)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("instantaneous", "local_output_only", "partial_3hop", "full_reach", "relaxed_10hop"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        x, y = raw_vision_dataset(args.dataset, seed=seed, subsample=args.subsample)
        for method in args.methods:
            row = train(
                method, args.dataset, x, y, seed, args.epochs,
                args.batch_size, args.metric_batch,
            )
            rows.append(row)
            print(seed, method, row["test_accuracy"], row["test_loss"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}; real_data is required and no synthetic fallback is used")


if __name__ == "__main__":
    main()
