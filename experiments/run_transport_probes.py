#!/usr/bin/env python3
"""Non-vision probes for output-factor and gauge transport.

No MNIST or CIFAR-10 data are loaded.  The pullback probe distinguishes a
three-hop partial field from a graph-diameter field that reaches every block.
The gauge probe uses explicit Procrustes edge maps and reports a non-trivial
transport residual in a shared probe-output space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metric_wave.data import load_dataset_three_way
from metric_wave.model import MLP
from metric_wave.optimizers import MetricWave
from metric_wave.riemannian_metric import (
    OutputFactorTransportPullback,
    RiemannianPullback,
)
from metric_wave.training import evaluate
from metric_wave.transported_gauge import TransportedGaugeMetricWave


WIDTHS = {
    "moons": [2, 32, 32, 24, 16, 8, 2],
    "digits": [64, 64, 64, 48, 32, 16, 10],
}


@dataclass(frozen=True)
class ProbeConfig:
    profile: str
    datasets: Sequence[str]
    seeds: Sequence[int]
    epochs: int
    patience: int
    batch_size: int = 64
    metric_batch: int = 8
    learning_rate: float = 0.045
    momentum: float = 0.9


def profile_config(profile: str) -> ProbeConfig:
    if profile == "quick":
        return ProbeConfig(profile, ("moons",), (0,), 8, 4)
    if profile == "standard":
        return ProbeConfig(profile, ("moons", "digits"), (0, 1, 2), 25, 8)
    if profile == "full":
        return ProbeConfig(profile, ("moons", "digits"), (0, 1, 2, 3, 4), 40, 12)
    raise ValueError(profile)


def _restore(model: MLP, blocks: Sequence[np.ndarray]) -> None:
    model.set_parameter_blocks([block.copy() for block in blocks])


def _train(
    dataset: str,
    seed: int,
    config: ProbeConfig,
    kind: str,
    family: str,
) -> Dict[str, float]:
    xtr, xva, xte, ytr, yva, yte = load_dataset_three_way(dataset, seed=seed)
    model = MLP(WIDTHS[dataset], seed=seed)
    diameter = model.n_layers - 1
    common = dict(
        model=model,
        lr=config.learning_rate,
        mass=1.0,
        output_metric="gauss_newton",
        metric_batch=config.metric_batch,
        momentum=config.momentum,
        seed=seed,
    )

    if family == "pullback":
        if kind == "instantaneous":
            optimizer = RiemannianPullback(**common)
        elif kind == "local_output_only":
            optimizer = OutputFactorTransportPullback(
                **common, metric_steps=diameter, wave_speed=0.0, relax_rate=0.6
            )
        elif kind == "partial_3hop":
            optimizer = OutputFactorTransportPullback(
                **common, metric_steps=3, wave_speed=1.0, relax_rate=0.6
            )
        elif kind == "full_reach":
            optimizer = OutputFactorTransportPullback(
                **common,
                metric_steps=None,
                wave_speed=1.0,
                relax_rate=0.6,
                require_full_reach=True,
            )
        elif kind == "relaxed_10hop":
            optimizer = OutputFactorTransportPullback(
                **common,
                metric_steps=10,
                wave_speed=1.0,
                relax_rate=0.6,
                require_full_reach=True,
            )
        else:
            raise ValueError(kind)
    elif family == "gauge":
        if kind not in {"transported_propagating", "transported_local"}:
            raise ValueError(kind)
        optimizer = TransportedGaugeMetricWave(
            lr=config.learning_rate,
            beta=config.momentum,
            rank=6,
            rho=0.8,
            coupling=3.0,
            damping=1.3,
            restoring=1.0,
            wave_speed=2.0 if kind == "transported_propagating" else 0.0,
            dt=0.06,
            substeps=3,
            max_metric_eigenvalue=2.0,
            tensor=True,
            source_mode="output",
            gauge_batch=min(16, config.metric_batch * 2),
            seed=seed,
        )
    else:
        raise ValueError(family)

    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_blocks = model.parameter_blocks()
    diag_rows: List[Dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(xtr.shape[0])
        for start in range(0, xtr.shape[0], config.batch_size):
            idx = order[start : start + config.batch_size]
            _, gradients = model.loss_and_gradients(xtr[idx], ytr[idx])
            if family == "pullback":
                optimizer.set_metric_batch(xtr[idx], ytr[idx])
            else:
                optimizer.set_gauge_batch(model, xtr[idx])
            model.set_parameter_blocks(optimizer.step(model.parameter_blocks(), gradients))
            diag_rows.append(optimizer.diagnostics())
        validation_loss = model.loss(xva, yva)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_blocks = model.parameter_blocks()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    _restore(model, best_blocks)
    test = evaluate(model, xte, yte)
    diagnostic_keys = sorted({key for row in diag_rows for key in row})
    diagnostics = {
        key: float(np.mean([row[key] for row in diag_rows if key in row]))
        for key in diagnostic_keys
    }
    row: Dict[str, float] = {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "method": kind,
        "affine_depth": model.n_layers,
        "graph_diameter": diameter,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_validation_loss": best_loss,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    row.update(diagnostics)
    return row


def _paired_summary(frame: pd.DataFrame, family: str, left: str, right: str) -> Dict:
    subset = frame[frame.family == family]
    pivot_loss = subset.pivot_table(index=["dataset", "seed"], columns="method", values="test_loss")
    pivot_acc = subset.pivot_table(index=["dataset", "seed"], columns="method", values="test_accuracy")
    result = {"family": family, "left": left, "right": right}
    for name, pivot in (("loss", pivot_loss), ("accuracy", pivot_acc)):
        values = (pivot[left] - pivot[right]).dropna().to_numpy()
        result[f"delta_{name}_mean"] = float(values.mean())
        result[f"delta_{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if len(values) > 1 and np.any(np.abs(values) > 0):
            result[f"wilcoxon_{name}_p"] = float(wilcoxon(values).pvalue)
        else:
            result[f"wilcoxon_{name}_p"] = 1.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "standard", "full"), default="quick")
    parser.add_argument("--output", type=Path, default=ROOT / "transport_results" / "quick")
    args = parser.parse_args()
    config = profile_config(args.profile)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    pullback_methods = (
        "instantaneous",
        "local_output_only",
        "partial_3hop",
        "full_reach",
        "relaxed_10hop",
    )
    gauge_methods = ("transported_propagating", "transported_local")
    total = len(config.datasets) * len(config.seeds) * (len(pullback_methods) + len(gauge_methods))
    index = 0
    for dataset in config.datasets:
        for seed in config.seeds:
            for method in pullback_methods:
                index += 1
                row = _train(dataset, seed, config, method, "pullback")
                rows.append(row)
                print(f"[{index}/{total}] {dataset} s{seed} pullback/{method}: loss={row['test_loss']:.6f}")
            for method in gauge_methods:
                index += 1
                row = _train(dataset, seed, config, method, "gauge")
                rows.append(row)
                print(f"[{index}/{total}] {dataset} s{seed} gauge/{method}: loss={row['test_loss']:.6f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "transport_probe_runs.csv", index=False)
    summaries = [
        _paired_summary(frame, "pullback", "partial_3hop", "local_output_only"),
        _paired_summary(frame, "pullback", "full_reach", "local_output_only"),
        _paired_summary(frame, "pullback", "relaxed_10hop", "instantaneous"),
        _paired_summary(frame, "gauge", "transported_propagating", "transported_local"),
    ]
    pd.DataFrame(summaries).to_csv(args.output / "transport_probe_comparisons.csv", index=False)

    config_dict = asdict(config)
    config_dict["datasets"] = list(config.datasets)
    config_dict["seeds"] = list(config.seeds)
    payload = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    metadata = {
        "config": config_dict,
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "vision_datasets_executed": False,
        "interpretation": {
            "partial_3hop": "partial support; not an all-block transport result",
            "full_reach": "minimum graph-diameter support at every block",
            "coherent_gauge": "uses explicit Procrustes edge transports R H R^T",
        },
    }
    (args.output / "transport_probe_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote non-vision probes to {args.output}")


if __name__ == "__main__":
    main()
