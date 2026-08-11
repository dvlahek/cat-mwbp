#!/usr/bin/env python3
"""Validation-controlled CAT-MWBP Neural Processing Letters suite.

The protocol uses a train/validation/test split, restores the best validation
checkpoint, includes FA, DFA, layer-local auxiliary-head, and matched
Activation Relaxation controls, validates the exact six-block training depth,
compares first- and second-order local adjoint relaxation at equal local sweep
budgets, audits the analytic overdamped CAT limit, calibrates a train-only
energy-matched non-propagating metric ablation, and generates
dataset-hierarchical comparisons and practical-equivalence tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "cat_mwbp_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import friedmanchisquare, t, wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metric_wave.data import load_dataset_three_way
from metric_wave.direct_feedback import DirectFeedbackAlignment, NeighborFeedbackAlignment
from metric_wave.local_adjoint import (
    activation_relaxation_gradients,
    gradient_comparison,
    local_adjoint_gradients,
    overdamped_adjoint_gradients,
    overdamped_relaxation_rate,
)
from metric_wave.local_heads import LocalAuxiliaryHeads
from metric_wave.model import MLP
from metric_wave.optimizers import Adam, MetricWave, Momentum
from metric_wave.training import evaluate
from metric_wave.validated_training import train_with_validation


ARCHITECTURES: Dict[str, Tuple[int, ...]] = {
    "moons": (2, 32, 32, 24, 16, 8, 2),
    "circles": (2, 32, 32, 24, 16, 8, 2),
    "iris": (4, 32, 32, 24, 16, 8, 3),
    "wine": (13, 40, 40, 32, 24, 12, 3),
    "breast_cancer": (30, 48, 48, 32, 24, 12, 2),
    "anisotropic": (20, 48, 48, 32, 24, 16, 2),
    "digits": (64, 64, 64, 48, 32, 16, 10),
    "synthetic_large": (100, 128, 128, 96, 64, 32, 10),
}

METHODS = (
    "BP-Momentum",
    "BP-Adam",
    "BP-MWBP",
    "FA-Momentum",
    "DFA-Momentum",
    "LocalHead-Momentum",
    "AR64-Momentum",
    "CAT40-Momentum",
    "CAT64-Momentum",
    "CAT64-MWBP",
    "CAT64-MWBP-local",
    "CAT64-MWBP-local-EM",
)

CORE_DATASETS = ("moons", "circles", "iris", "wine", "breast_cancer", "anisotropic", "digits")
SCALING_METHODS = (
    "BP-Momentum",
    "DFA-Momentum",
    "LocalHead-Momentum",
    "AR64-Momentum",
    "CAT64-Momentum",
    "CAT64-MWBP",
    "CAT64-MWBP-local-EM",
)


@dataclass(frozen=True)
class JournalConfig:
    profile: str
    max_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    core_seeds: Tuple[int, ...]
    scaling_seeds: Tuple[int, ...]
    include_scaling: bool
    learning_rate: float = 0.045
    momentum: float = 0.9
    adjoint_dt: float = 0.04
    adjoint_damping: float = 8.0
    adjoint_frequency: float = 8.0
    ar_steps: int = 64
    ar_relaxation_rate: float = 0.16
    calibration_batches: int = 8
    calibration_iterations: int = 12
    accuracy_equivalence_margin: float = 0.005
    loss_equivalence_margin: float = 0.001
    hierarchical_bootstrap_draws: int = 10000


@dataclass(frozen=True)
class RunTask:
    dataset: str
    method: str
    seed: int
    experiment_group: str
    config: JournalConfig


def profile_config(profile: str) -> JournalConfig:
    if profile == "quick":
        return JournalConfig(
            profile, 25, 5, 1e-4, 64, (0,), (), False,
            calibration_batches=3, calibration_iterations=7,
            hierarchical_bootstrap_draws=2000,
        )
    if profile == "standard":
        return JournalConfig(profile, 120, 15, 1e-4, 64, tuple(range(5)), (0, 1), True)
    return JournalConfig(profile, 150, 20, 1e-4, 64, tuple(range(10)), (0, 1, 2), True)


def task_plan(config: JournalConfig) -> List[RunTask]:
    if config.profile == "quick":
        datasets = ("moons", "breast_cancer")
        methods = (
            "BP-Momentum", "FA-Momentum", "DFA-Momentum", "LocalHead-Momentum",
            "AR64-Momentum", "CAT40-Momentum", "CAT64-Momentum", "CAT64-MWBP",
            "CAT64-MWBP-local-EM",
        )
    else:
        datasets, methods = CORE_DATASETS, METHODS
    tasks = [RunTask(dataset, method, seed, "core", config) for dataset in datasets for method in methods for seed in config.core_seeds]
    if config.include_scaling:
        tasks.extend(
            RunTask("synthetic_large", method, seed, "scaling", config)
            for method in SCALING_METHODS
            for seed in config.scaling_seeds
        )
    return tasks


def config_payload(config: JournalConfig) -> Dict[str, object]:
    payload = asdict(config)
    payload.update(
        methods=list(METHODS),
        core_datasets=list(CORE_DATASETS),
        scaling_methods=list(SCALING_METHODS),
        architectures={key: list(value) for key, value in ARCHITECTURES.items()},
        split={"train": 0.60, "validation": 0.15, "test": 0.25, "scaler_fit": "train_only"},
        overdamped_analysis={
            "derived_rate": overdamped_relaxation_rate(
                config.adjoint_dt,
                config.adjoint_damping,
                config.adjoint_frequency,
            ),
            "ar_rate_predeclared": config.ar_relaxation_rate,
            "comparison_budget": "equal synchronous local sweeps and equal one-hop Jacobian actions",
        },
    )
    return payload


def fingerprint(config: JournalConfig) -> str:
    return hashlib.sha256(json.dumps(config_payload(config), sort_keys=True).encode()).hexdigest()[:16]


def method_parts(method: str) -> Tuple[str, int, str]:
    if method.startswith("BP-"):
        return "bp", 0, method.removeprefix("BP-").lower()
    if method.startswith("DFA-"):
        return "dfa", 0, method.removeprefix("DFA-").lower()
    if method.startswith("FA-"):
        return "fa", 0, method.removeprefix("FA-").lower()
    if method.startswith("LocalHead-"):
        return "localhead", 0, method.removeprefix("LocalHead-").lower()
    if method.startswith("AR"):
        credit, optimizer = method.split("-", 1)
        return "ar", int(credit.removeprefix("AR")), optimizer.lower()
    credit, optimizer = method.split("-", 1)
    return "cat", int(credit.removeprefix("CAT")), optimizer.lower()


def optimizer_factory(kind: str, seed: int, config: JournalConfig, coupling: float = 3.0):
    if kind == "momentum":
        return Momentum(lr=config.learning_rate, beta=config.momentum)
    if kind == "adam":
        return Adam(lr=0.003)
    common = dict(
        lr=config.learning_rate,
        beta=config.momentum,
        rank=6,
        rho=0.8,
        coupling=coupling,
        damping=1.3,
        restoring=1.0,
        dt=0.06,
        substeps=3,
        max_metric_eigenvalue=2.0,
        tensor=True,
        source_mode="output",
        seed=seed,
    )
    if kind == "mwbp":
        return MetricWave(wave_speed=2.0, **common)
    if kind in {"mwbp-local", "mwbp-local-em"}:
        return MetricWave(wave_speed=0.0, **common)
    raise ValueError(kind)


def make_gradient_function(credit: str, steps: int, seed: int, config: JournalConfig):
    if credit == "bp":
        def bp(model, x, y):
            loss, gradients = model.loss_and_gradients(x, y)
            return loss, gradients, {}
        return bp
    if credit == "dfa":
        dfa = DirectFeedbackAlignment(seed=seed)
        return dfa.gradients
    if credit == "fa":
        feedback = NeighborFeedbackAlignment(seed=seed)
        return feedback.gradients
    if credit == "localhead":
        local_heads = LocalAuxiliaryHeads(seed=seed)
        return local_heads.gradients
    if credit == "ar":
        def ar(model, x, y):
            return activation_relaxation_gradients(
                model,
                x,
                y,
                steps=steps,
                rate=config.ar_relaxation_rate,
            )
        return ar

    def cat(model, x, y):
        return local_adjoint_gradients(
            model,
            x,
            y,
            steps=steps,
            dt=config.adjoint_dt,
            damping=config.adjoint_damping,
            frequency=config.adjoint_frequency,
        )
    return cat


def metric_energy_for_sequence(gradients, parameters, config, seed, wave_speed, coupling):
    optimizer = MetricWave(
        lr=config.learning_rate,
        beta=config.momentum,
        rank=6,
        rho=0.8,
        coupling=coupling,
        damping=1.3,
        restoring=1.0,
        wave_speed=wave_speed,
        dt=0.06,
        substeps=3,
        max_metric_eigenvalue=2.0,
        tensor=True,
        source_mode="output",
        seed=seed,
    )
    energies = []
    for gradient in gradients:
        optimizer.step(parameters, gradient)
        energies.append(optimizer.diagnostics()["metric_energy"])
    tail = energies[max(0, len(energies) // 2):]
    return float(np.mean(tail))


def calibrate_energy_matched_local(model, gradient_function, x_train, y_train, seed, config):
    rng = np.random.default_rng(seed + 7919)
    gradients = []
    for _ in range(config.calibration_batches):
        size = min(config.batch_size, len(x_train))
        idx = rng.choice(len(x_train), size=size, replace=False)
        _, gradient, _ = gradient_function(model, x_train[idx], y_train[idx])
        gradients.append([np.asarray(block).copy() for block in gradient])
    parameters = [block.copy() for block in model.parameter_blocks()]
    optimizer_seed = seed + 2000
    target = metric_energy_for_sequence(gradients, parameters, config, optimizer_seed, 2.0, 3.0)
    low, high = 0.02, 3.0
    for _ in range(config.calibration_iterations):
        middle = 0.5 * (low + high)
        observed = metric_energy_for_sequence(gradients, parameters, config, optimizer_seed, 0.0, middle)
        if observed < target:
            low = middle
        else:
            high = middle
    coupling = 0.5 * (low + high)
    achieved = metric_energy_for_sequence(gradients, parameters, config, optimizer_seed, 0.0, coupling)
    return float(coupling), target, achieved


def run_task(task: RunTask):
    config = task.config
    x_train, x_validation, x_test, y_train, y_validation, y_test = load_dataset_three_way(task.dataset, task.seed)
    model = MLP(ARCHITECTURES[task.dataset], seed=task.seed + 1000)
    credit, steps, optimizer_kind = method_parts(task.method)
    gradient_function = make_gradient_function(credit, steps, task.seed + 4000, config)
    coupling, target_energy, achieved_energy = 3.0, np.nan, np.nan
    started = time.perf_counter()
    if optimizer_kind == "mwbp-local-em":
        coupling, target_energy, achieved_energy = calibrate_energy_matched_local(
            model, gradient_function, x_train, y_train, task.seed, config
        )
    optimizer = optimizer_factory(optimizer_kind, task.seed + 2000, config, coupling=coupling)
    trained = train_with_validation(
        model,
        optimizer,
        gradient_function,
        x_train,
        y_train,
        x_validation,
        y_validation,
        max_epochs=config.max_epochs if task.experiment_group == "core" else min(config.max_epochs, 100),
        patience=config.patience,
        min_delta=config.min_delta,
        batch_size=config.batch_size,
        seed=task.seed + 3000,
    )
    elapsed = time.perf_counter() - started
    history = pd.DataFrame(trained.history)
    history["dataset"] = task.dataset
    history["method"] = task.method
    history["seed"] = task.seed
    history["experiment_group"] = task.experiment_group
    history["credit_rule"] = credit
    history["local_steps"] = steps
    train = evaluate(model, x_train, y_train)
    validation = evaluate(model, x_validation, y_validation)
    test = evaluate(model, x_test, y_test)
    prediction = model.predict(x_test)
    diagnostics = optimizer.diagnostics()
    final: Dict[str, object] = {
        "dataset": task.dataset,
        "method": task.method,
        "seed": task.seed,
        "experiment_group": task.experiment_group,
        "credit_rule": credit,
        "local_steps": steps,
        "credit_dynamics_order": 1 if credit == "ar" else (2 if credit == "cat" else 0),
        "parallel_relaxation_rounds_per_batch": steps,
        "neighbor_jacobian_actions_per_batch": steps * max(0, model.n_layers - 1),
        "first_order_relaxation_rate": config.ar_relaxation_rate if credit == "ar" else np.nan,
        "best_epoch": trained.best_epoch,
        "stopped_epoch": trained.stopped_epoch,
        "early_stopped": trained.early_stopped,
        "best_validation_loss": trained.best_validation_loss,
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "elapsed_seconds": elapsed,
        "n_train": len(x_train),
        "n_validation": len(x_validation),
        "n_test": len(x_test),
        "n_features": x_train.shape[1],
        "n_classes": int(np.unique(y_train).size),
        "affine_depth": model.n_layers,
        "parameter_count": int(sum(block.size for block in model.parameter_blocks())),
        "calibrated_local_coupling": coupling if optimizer_kind == "mwbp-local-em" else np.nan,
        "calibration_target_energy": target_energy,
        "calibration_achieved_energy": achieved_energy,
    }
    final.update(diagnostics)
    return history, final


def checkpoint_path(output: Path, task: RunTask) -> Path:
    method = task.method.lower().replace("-", "_")
    return output / "checkpoints" / f"{task.experiment_group}__{task.dataset}__{method}__seed{task.seed}.json"


def save_checkpoint(path: Path, history: pd.DataFrame, final: Dict[str, object], signature: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": signature,
        "history": history.replace({np.nan: None}).to_dict(orient="records"),
        "final": {key: (None if pd.isna(value) else value) for key, value in final.items()},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path, signature: str):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != signature:
        return None
    return pd.DataFrame(payload["history"]), payload["final"]


def execute(tasks: Sequence[RunTask], output: Path, jobs: int, force: bool):
    signature = fingerprint(tasks[0].config)
    histories, finals, pending, failures = [], [], [], []
    for task in tasks:
        saved = None if force else load_checkpoint(checkpoint_path(output, task), signature)
        if saved is None:
            pending.append(task)
        else:
            histories.append(saved[0]); finals.append(saved[1])
    print(f"Training: {len(tasks)-len(pending)} resumed, {len(pending)} pending.", flush=True)

    def accept(task, result):
        history, final = result
        save_checkpoint(checkpoint_path(output, task), history, final, signature)
        histories.append(history); finals.append(final)

    if jobs == 1:
        for index, task in enumerate(pending, 1):
            print(f"[{index:04d}/{len(pending):04d}] {task.dataset} {task.method} seed={task.seed}", flush=True)
            try:
                accept(task, run_task(task))
            except Exception as exc:
                failures.append({"dataset": task.dataset, "method": task.method, "seed": task.seed, "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_task, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                print(f"[{index:04d}/{len(pending):04d}] {task.dataset} {task.method} seed={task.seed}", flush=True)
                try:
                    accept(task, future.result())
                except Exception as exc:
                    failures.append({"dataset": task.dataset, "method": task.method, "seed": task.seed, "error": repr(exc)})
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    final = pd.DataFrame(finals)
    pd.DataFrame(failures, columns=["dataset", "method", "seed", "error"]).to_csv(output / "failures.csv", index=False)
    return history, final, failures


def depth_architecture(input_dim: int, classes: int, depth: int):
    hidden = tuple(max(12, min(48, input_dim)) for _ in range(depth - 1))
    return (input_dim, *hidden, classes)


def validate_relaxation_methods(config: JournalConfig, quick: bool = False) -> pd.DataFrame:
    """Compare CAT and matched AR under equal local communication budgets."""
    datasets = ("moons", "breast_cancer") if quick else CORE_DATASETS
    seeds = config.core_seeds[:1] if quick else config.core_seeds[:5]
    depths = (3, 6, 9) if quick else (3, 5, 6, 7, 9)
    steps_grid = (0, 16, 40, 64) if quick else (0, 1, 2, 4, 8, 16, 24, 32, 40, 64, 100, 160)
    rows = []
    rules = ("CAT", "AR")
    total = len(datasets) * len(seeds) * len(depths) * len(steps_grid) * len(rules)
    count = 0
    for dataset in datasets:
        for seed in seeds:
            x_train, _, _, y_train, _, _ = load_dataset_three_way(dataset, seed)
            x, y = x_train[:min(64, len(x_train))], y_train[:min(64, len(y_train))]
            for depth in depths:
                model = MLP(depth_architecture(x.shape[1], int(np.unique(y_train).size), depth), seed=seed + 5000 + depth)
                _, exact = model.loss_and_gradients(x, y)
                for steps in steps_grid:
                    for rule in rules:
                        count += 1
                        print(
                            f"Gradient [{count:04d}/{total:04d}] {rule} {dataset} "
                            f"depth={depth} seed={seed} sweeps={steps}",
                            flush=True,
                        )
                        if rule == "CAT":
                            _, approximate, diagnostics = local_adjoint_gradients(
                                model, x, y, steps=steps, dt=config.adjoint_dt,
                                damping=config.adjoint_damping, frequency=config.adjoint_frequency,
                            )
                        else:
                            _, approximate, diagnostics = activation_relaxation_gradients(
                                model, x, y, steps=steps, rate=config.ar_relaxation_rate
                            )
                        row = gradient_comparison(exact, approximate)
                        row.update(diagnostics)
                        row.update(
                            dataset=dataset,
                            seed=seed,
                            affine_depth=depth,
                            local_steps=steps,
                            credit_rule=rule,
                            parallel_relaxation_rounds=steps,
                            neighbor_jacobian_actions=steps * max(0, depth - 1),
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def validate_overdamped_limit(config: JournalConfig, quick: bool = False) -> pd.DataFrame:
    """Numerically verify that matched AR equals the explicit overdamped CAT limit."""
    datasets = ("moons",) if quick else ("moons", "breast_cancer", "digits")
    seeds = config.core_seeds[:1] if quick else config.core_seeds[:3]
    depths = (3, 6) if quick else (3, 6, 9)
    steps_grid = (1, 16, 64)
    derived_rate = overdamped_relaxation_rate(
        config.adjoint_dt, config.adjoint_damping, config.adjoint_frequency
    )
    rows = []
    for dataset in datasets:
        for seed in seeds:
            x_train, _, _, y_train, _, _ = load_dataset_three_way(dataset, seed)
            x, y = x_train[:min(64, len(x_train))], y_train[:min(64, len(y_train))]
            for depth in depths:
                model = MLP(
                    depth_architecture(x.shape[1], int(np.unique(y_train).size), depth),
                    seed=seed + 7000 + depth,
                )
                for steps in steps_grid:
                    _, ar_gradients, _ = activation_relaxation_gradients(
                        model, x, y, steps=steps, rate=config.ar_relaxation_rate
                    )
                    _, od_gradients, _ = overdamped_adjoint_gradients(
                        model,
                        x,
                        y,
                        steps=steps,
                        dt=config.adjoint_dt,
                        damping=config.adjoint_damping,
                        frequency=config.adjoint_frequency,
                    )
                    comparison = gradient_comparison(ar_gradients, od_gradients)
                    ar_flat = np.concatenate([block.ravel() for block in ar_gradients])
                    od_flat = np.concatenate([block.ravel() for block in od_gradients])
                    rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "affine_depth": depth,
                        "local_steps": steps,
                        "configured_ar_rate": config.ar_relaxation_rate,
                        "derived_overdamped_rate": derived_rate,
                        "rate_absolute_difference": abs(config.ar_relaxation_rate - derived_rate),
                        "gradient_relative_difference": comparison["gradient_relative_error"],
                        "gradient_cosine": comparison["gradient_cosine"],
                        "gradient_max_absolute_difference": float(np.max(np.abs(ar_flat - od_flat))),
                        "neighbor_jacobian_actions": steps * max(0, depth - 1),
                    })
    return pd.DataFrame(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()); std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(t.ppf(0.975, len(values)-1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, std, mean-half, mean+half


def summarize(final: pd.DataFrame):
    rows = []
    for (group, dataset, method), frame in final.groupby(["experiment_group", "dataset", "method"], sort=True):
        row = {"experiment_group": group, "dataset": dataset, "method": method, "n": len(frame)}
        for metric in ("test_accuracy", "balanced_accuracy", "macro_f1", "test_loss", "elapsed_seconds", "best_epoch"):
            mean, std, low, high = mean_ci(frame[metric])
            row.update({f"{metric}_mean": mean, f"{metric}_std": std, f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        row["early_stop_fraction"] = float(frame.early_stopped.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p_values):
    values = np.asarray(p_values, dtype=float); order = np.argsort(values); result = np.empty_like(values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank) * values[index]); result[index] = min(1.0, running)
    return result


def bootstrap_ci(values, seed=271828, samples=10000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed); means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


PRIMARY_COMPARISONS = (
    ("FA-Momentum", "BP-Momentum"),
    ("DFA-Momentum", "BP-Momentum"),
    ("LocalHead-Momentum", "BP-Momentum"),
    ("AR64-Momentum", "BP-Momentum"),
    ("CAT40-Momentum", "BP-Momentum"),
    ("CAT64-Momentum", "BP-Momentum"),
    ("BP-MWBP", "BP-Momentum"),
    ("CAT64-Momentum", "AR64-Momentum"),
    ("CAT64-Momentum", "CAT40-Momentum"),
    ("CAT64-MWBP", "CAT64-Momentum"),
    ("CAT64-MWBP", "CAT64-MWBP-local"),
    ("CAT64-MWBP", "CAT64-MWBP-local-EM"),
)


def hierarchical_bootstrap_ci(frame, column, seed=161803, samples=10000):
    """Bootstrap datasets first and seeds second to preserve nesting."""
    datasets = tuple(sorted(frame.dataset.unique()))
    if len(datasets) < 2:
        return bootstrap_ci(frame[column].to_numpy(), seed=seed, samples=samples)
    grouped = {dataset: frame[frame.dataset == dataset][column].to_numpy() for dataset in datasets}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for draw in range(samples):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for dataset in selected:
            values = grouped[dataset]
            means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[draw] = float(np.mean(means))
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def paired_frame(final, method, baseline, group="core"):
    subset = final[final.experiment_group == group]
    columns = ["dataset", "seed", "test_loss", "test_accuracy", "elapsed_seconds"]
    left = subset[subset.method == method][columns]
    right = subset[subset.method == baseline][columns]
    merged = left.merge(right, on=["dataset", "seed"], suffixes=("_method", "_baseline"))
    merged["loss_difference"] = merged.test_loss_method - merged.test_loss_baseline
    merged["accuracy_difference"] = merged.test_accuracy_method - merged.test_accuracy_baseline
    merged["runtime_ratio"] = merged.elapsed_seconds_method / merged.elapsed_seconds_baseline
    return merged


def aggregate_paired(final, config):
    """Primary comparisons across all core datasets with nested uncertainty."""
    rows = []
    for index, (method, baseline) in enumerate(PRIMARY_COMPARISONS):
        frame = paired_frame(final, method, baseline)
        if frame.empty:
            continue
        loss = frame.loss_difference.to_numpy()
        accuracy = frame.accuracy_difference.to_numpy()
        dataset_loss = frame.groupby("dataset").loss_difference.mean().to_numpy()
        loss_low, loss_high = hierarchical_bootstrap_ci(
            frame, "loss_difference", seed=161803 + index,
            samples=config.hierarchical_bootstrap_draws,
        )
        accuracy_low, accuracy_high = hierarchical_bootstrap_ci(
            frame, "accuracy_difference", seed=271828 + index,
            samples=config.hierarchical_bootstrap_draws,
        )
        try:
            pooled_p = float(wilcoxon(loss, zero_method="zsplit").pvalue)
        except ValueError:
            pooled_p = 1.0
        try:
            dataset_p = float(wilcoxon(dataset_loss, zero_method="zsplit").pvalue)
        except ValueError:
            dataset_p = 1.0
        rows.append({
            "method": method,
            "baseline": baseline,
            "n_dataset_seed_pairs": len(frame),
            "n_datasets": frame.dataset.nunique(),
            "loss_difference_mean": float(loss.mean()),
            "loss_difference_hierarchical_ci95_low": loss_low,
            "loss_difference_hierarchical_ci95_high": loss_high,
            "accuracy_difference_mean": float(accuracy.mean()),
            "accuracy_difference_hierarchical_ci95_low": accuracy_low,
            "accuracy_difference_hierarchical_ci95_high": accuracy_high,
            "method_better_loss_count": int((loss < 0).sum()),
            "dataset_mean_loss_win_count": int((dataset_loss < 0).sum()),
            "wilcoxon_loss_p_pooled": pooled_p,
            "wilcoxon_loss_p_dataset_means": dataset_p,
            "runtime_ratio_mean": float(frame.runtime_ratio.mean()),
            "runtime_ratio_median": float(frame.runtime_ratio.median()),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_loss_p_pooled_holm"] = holm(result.wilcoxon_loss_p_pooled)
        result["wilcoxon_loss_p_dataset_means_holm"] = holm(result.wilcoxon_loss_p_dataset_means)
    return result


def tost(values, margin):
    """Two one-sided t tests for equivalence around a symmetric margin."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean, 1.0, False
    std = float(values.std(ddof=1))
    half = float(t.ppf(.975, len(values)-1) * std / np.sqrt(len(values)))
    if std <= np.finfo(float).eps:
        p_value = 0.0 if -margin < mean < margin else 1.0
    else:
        standard_error = std / np.sqrt(len(values))
        lower_statistic = (mean + margin) / standard_error
        upper_statistic = (mean - margin) / standard_error
        lower_p = float(1.0 - t.cdf(lower_statistic, len(values)-1))
        upper_p = float(t.cdf(upper_statistic, len(values)-1))
        p_value = max(lower_p, upper_p)
    return mean, mean-half, mean+half, p_value, bool(p_value < .05)


def equivalence_tests(final, config):
    rows = []
    for method in ("AR64-Momentum", "CAT40-Momentum", "CAT64-Momentum"):
        frame = paired_frame(final, method, "BP-Momentum")
        if frame.empty:
            continue
        for metric, margin in (
            ("test_accuracy", config.accuracy_equivalence_margin),
            ("test_loss", config.loss_equivalence_margin),
        ):
            difference = frame[f"{metric}_method"] - frame[f"{metric}_baseline"]
            dataset_means = difference.groupby(frame.dataset).mean().to_numpy()
            mean, low, high, p_value, equivalent = tost(dataset_means, margin)
            rows.append({
                "method": method,
                "baseline": "BP-Momentum",
                "metric": metric,
                "margin": margin,
                "n_datasets": len(dataset_means),
                "dataset_mean_difference": mean,
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "tost_p": p_value,
                "equivalent_at_0_05": equivalent,
            })
    return pd.DataFrame(rows)


def global_rank_tests(final):
    core = final[final.experiment_group == "core"]
    rows = []
    for level, frame in (
        ("dataset_seed", core.pivot(index=["dataset", "seed"], columns="method", values="test_loss")),
        ("dataset_mean", core.groupby(["dataset", "method"]).test_loss.mean().unstack()),
    ):
        frame = frame.dropna()
        if frame.shape[0] < 2 or frame.shape[1] < 3:
            continue
        result = friedmanchisquare(*(frame[column] for column in frame.columns))
        rows.append({
            "analysis_level": level,
            "n_blocks": frame.shape[0],
            "n_methods": frame.shape[1],
            "friedman_statistic": float(result.statistic),
            "friedman_p": float(result.pvalue),
        })
    return pd.DataFrame(rows)


def paired(final: pd.DataFrame):
    rows = []
    for (group, dataset), subset in final.groupby(["experiment_group", "dataset"]):
        available = set(subset.method)
        for method, baseline in PRIMARY_COMPARISONS:
            if method not in available or baseline not in available:
                continue
            left = subset[subset.method == method].set_index("seed")
            right = subset[subset.method == baseline].set_index("seed")
            seeds = left.index.intersection(right.index)
            dl = (left.loc[seeds, "test_loss"] - right.loc[seeds, "test_loss"]).to_numpy()
            da = (left.loc[seeds, "test_accuracy"] - right.loc[seeds, "test_accuracy"]).to_numpy()
            low, high = bootstrap_ci(dl)
            try: p_value = float(wilcoxon(dl, zero_method="zsplit").pvalue)
            except ValueError: p_value = 1.0
            sd = float(dl.std(ddof=1)) if len(dl) > 1 else 0.0
            rows.append({
                "experiment_group": group, "dataset": dataset, "method": method, "baseline": baseline,
                "n_pairs": len(seeds), "accuracy_difference_mean": float(da.mean()),
                "loss_difference_mean": float(dl.mean()), "loss_difference_ci95_low": low,
                "loss_difference_ci95_high": high, "paired_effect_dz": float(dl.mean()/sd) if sd else 0.0,
                "method_better_loss_count": int((dl < 0).sum()), "wilcoxon_loss_p": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty: result["wilcoxon_loss_p_holm"] = holm(result.wilcoxon_loss_p)
    return result


def ranks(final: pd.DataFrame):
    core = final[final.experiment_group == "core"].copy()
    core["loss_rank"] = core.groupby(["dataset", "seed"])["test_loss"].rank(method="average")
    return core.groupby("method", as_index=False).agg(mean_loss_rank=("loss_rank", "mean"), rank_std=("loss_rank", "std"), observations=("loss_rank", "size")).sort_values("mean_loss_rank")


def make_figures(
    history,
    final,
    summary,
    validation,
    relaxation_validation,
    overdamped_validation,
    rank_frame,
    aggregate,
    equivalence,
    output,
):
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(METHODS, plt.cm.tab20.colors))
    for dataset in sorted(history[history.experiment_group == "core"].dataset.unique()):
        subset = history[(history.experiment_group == "core") & (history.dataset == dataset)]
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        for method in [m for m in METHODS if m in set(subset.method)]:
            curve = subset[subset.method == method].groupby("epoch").validation_loss.mean()
            ax.plot(curve.index, curve.values, label=method, color=colors[method])
        ax.set(xlabel="Epoch", ylabel="Mean validation loss", title=dataset); ax.set_yscale("log"); ax.grid(alpha=.25)
        ax.legend(fontsize=6.5, ncol=2, frameon=False); fig.tight_layout()
        fig.savefig(figures/f"validation_curves_{dataset}.pdf", bbox_inches="tight"); fig.savefig(figures/f"validation_curves_{dataset}.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    if not validation.empty:
        grouped = validation.groupby(["affine_depth", "local_steps"], as_index=False).agg(error=("gradient_relative_error", "mean"), cosine=("gradient_cosine", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
        for depth, frame in grouped.groupby("affine_depth"):
            axes[0].plot(frame.local_steps, frame.error, marker="o", label=f"depth {depth}")
            axes[1].plot(frame.local_steps, frame.cosine, marker="o", label=f"depth {depth}")
        axes[0].set(xlabel="CAT steps", ylabel="Relative gradient error", title="Training-depth validation"); axes[0].set_yscale("log")
        axes[1].set(xlabel="CAT steps", ylabel="Gradient cosine", title="Direction alignment")
        for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(figures/"gradient_convergence_depth6.pdf", bbox_inches="tight"); fig.savefig(figures/"gradient_convergence_depth6.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    if not relaxation_validation.empty:
        selected = relaxation_validation[relaxation_validation.affine_depth == 6]
        grouped = selected.groupby(["credit_rule", "local_steps"], as_index=False).agg(
            error=("gradient_relative_error", "mean"),
            cosine=("gradient_cosine", "mean"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
        for rule, frame in grouped.groupby("credit_rule"):
            frame = frame.sort_values("local_steps")
            axes[0].plot(frame.local_steps, frame.error, marker="o", label=rule)
            axes[1].plot(frame.local_steps, frame.cosine, marker="o", label=rule)
        axes[0].set(
            xlabel="Equal local relaxation sweeps",
            ylabel="Relative gradient error",
            title="CAT versus Activation Relaxation",
        )
        axes[0].set_yscale("log")
        axes[1].set(
            xlabel="Equal local relaxation sweeps",
            ylabel="Gradient cosine",
            title="Direction alignment at depth 6",
        )
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures/"cat_ar_gradient_convergence.pdf", bbox_inches="tight")
        fig.savefig(figures/"cat_ar_gradient_convergence.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    if not overdamped_validation.empty:
        ordered = overdamped_validation.sort_values(
            ["affine_depth", "local_steps", "dataset", "seed"]
        ).reset_index(drop=True)
        numerical_floor = np.finfo(float).eps
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        ax.plot(
            np.arange(len(ordered)),
            np.maximum(ordered.gradient_max_absolute_difference, numerical_floor),
            marker="o",
            linestyle="none",
            markersize=3,
            color="#4c78a8",
        )
        ax.axhline(1e-12, color="#e45756", linestyle="--", linewidth=1, label="audit tolerance")
        ax.set(
            xlabel="Overdamped audit case",
            ylabel="Maximum absolute gradient difference",
            title="AR and overdamped CAT numerical identity",
        )
        ax.set_yscale("log")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures/"overdamped_limit_audit.pdf", bbox_inches="tight")
        fig.savefig(figures/"overdamped_limit_audit.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.3, 4.4)); ordered = rank_frame.sort_values("mean_loss_rank", ascending=False)
    rank_se = ordered.rank_std.fillna(0) / np.sqrt(ordered.observations.clip(lower=1))
    ax.barh(ordered.method, ordered.mean_loss_rank, xerr=rank_se, color="#4c78a8"); ax.set(xlabel="Mean core test-loss rank (error bar: SE)", title="Validation-selected models"); ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(figures/"average_ranks.pdf", bbox_inches="tight"); fig.savefig(figures/"average_ranks.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    core_summary = summary[summary.experiment_group == "core"]
    pivot = core_summary.pivot(index="dataset", columns="method", values="test_accuracy_mean")
    fig, ax = plt.subplots(figsize=(10.5, 4.5)); image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=max(0, float(np.nanmin(pivot.values))-.03), vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right"); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set_title("Validation-selected test accuracy"); fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(figures/"test_accuracy_heatmap.pdf", bbox_inches="tight"); fig.savefig(figures/"test_accuracy_heatmap.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    energy_matched = final[final.method == "CAT64-MWBP-local-EM"].copy()
    if not energy_matched.empty:
        calibration = energy_matched.groupby("dataset", as_index=False).agg(
            target=("calibration_target_energy", "mean"),
            achieved=("calibration_achieved_energy", "mean"),
            coupling=("calibrated_local_coupling", "mean"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        axes[0].scatter(calibration.target, calibration.achieved, color="#4c78a8")
        lower = min(calibration.target.min(), calibration.achieved.min()); upper = max(calibration.target.max(), calibration.achieved.max())
        axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#777777")
        axes[0].set(xlabel="Propagated target energy", ylabel="Local achieved energy", title="Train-only energy matching")
        axes[1].bar(calibration.dataset, calibration.coupling, color="#f58518")
        axes[1].set(ylabel="Calibrated local coupling", title="Frozen coupling by dataset")
        axes[1].tick_params(axis="x", rotation=35)
        for ax in axes: ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(figures/"energy_matched_calibration.pdf", bbox_inches="tight"); fig.savefig(figures/"energy_matched_calibration.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    scaling = summary[summary.experiment_group == "scaling"]
    if not scaling.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
        axes[0].bar(scaling.method, scaling.test_accuracy_mean, yerr=(scaling.test_accuracy_mean-scaling.test_accuracy_ci95_low, scaling.test_accuracy_ci95_high-scaling.test_accuracy_mean), color="#54a24b", capsize=3)
        axes[0].set(ylabel="Test accuracy", title="Large offline scaling dataset"); axes[0].set_ylim(max(0, scaling.test_accuracy_mean.min()-.05), min(1, scaling.test_accuracy_mean.max()+.03))
        axes[1].bar(scaling.method, scaling.elapsed_seconds_mean, yerr=(scaling.elapsed_seconds_mean-scaling.elapsed_seconds_ci95_low, scaling.elapsed_seconds_ci95_high-scaling.elapsed_seconds_mean), color="#b279a2", capsize=3)
        axes[1].set(ylabel="Mean runtime (s)", title="Scaling cost")
        for ax in axes: ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(figures/"scaling_results.pdf", bbox_inches="tight"); fig.savefig(figures/"scaling_results.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    if not aggregate.empty:
        selected = aggregate[aggregate.baseline == "BP-Momentum"].sort_values("loss_difference_mean")
        if not selected.empty:
            lower = selected.loss_difference_mean-selected.loss_difference_hierarchical_ci95_low
            upper = selected.loss_difference_hierarchical_ci95_high-selected.loss_difference_mean
            fig, ax = plt.subplots(figsize=(7.8, 4.5))
            ax.errorbar(selected.loss_difference_mean, selected.method, xerr=(lower, upper), fmt="o", color="#4c78a8", capsize=3)
            ax.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
            ax.set(xlabel="Mean test-loss difference vs BP-Momentum\n(dataset-hierarchical 95% CI)", title="Primary aggregate comparisons")
            ax.grid(axis="x", alpha=.25); fig.tight_layout()
            fig.savefig(figures/"aggregate_paired_differences.pdf", bbox_inches="tight"); fig.savefig(figures/"aggregate_paired_differences.png", dpi=180, bbox_inches="tight"); plt.close(fig)
        cat_ar = aggregate[
            (aggregate.method == "CAT64-Momentum")
            & (aggregate.baseline == "AR64-Momentum")
        ]
        if not cat_ar.empty:
            row = cat_ar.iloc[0]
            fig, ax = plt.subplots(figsize=(6.8, 2.6))
            ax.errorbar(
                [row.loss_difference_mean],
                ["CAT64 - AR64"],
                xerr=([
                    row.loss_difference_mean - row.loss_difference_hierarchical_ci95_low
                ], [
                    row.loss_difference_hierarchical_ci95_high - row.loss_difference_mean
                ]),
                fmt="o",
                color="#f58518",
                capsize=4,
            )
            ax.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
            ax.set(
                xlabel="Mean test-loss difference (dataset-hierarchical 95% CI)",
                title="Second-order CAT versus matched first-order AR",
            )
            ax.grid(axis="x", alpha=.25)
            fig.tight_layout()
            fig.savefig(figures/"cat_ar_test_loss.pdf", bbox_inches="tight")
            fig.savefig(figures/"cat_ar_test_loss.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    if not equivalence.empty:
        panels = (("test_accuracy", "Accuracy difference vs BP"), ("test_loss", "Loss difference vs BP"))
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
        drawn = False
        for ax, (metric, xlabel) in zip(axes, panels):
            frame = equivalence[equivalence.metric == metric].copy()
            if frame.empty:
                ax.set_visible(False)
                continue
            drawn = True
            margin = float(frame.margin.iloc[0])
            lower = frame.dataset_mean_difference-frame.difference_ci95_low
            upper = frame.difference_ci95_high-frame.dataset_mean_difference
            ax.errorbar(
                frame.dataset_mean_difference,
                frame.method,
                xerr=(lower, upper),
                fmt="o",
                color="#e45756",
                capsize=4,
            )
            ax.axvspan(-margin, margin, color="#54a24b", alpha=.16, label=f"margin ±{margin:g}")
            ax.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
            ax.set(xlabel=xlabel, title=metric.replace("test_", "").replace("_", " ").title())
            ax.legend(frameon=False)
            ax.grid(axis="x", alpha=.25)
        if drawn:
            fig.suptitle("Predeclared practical equivalence to BP")
            fig.tight_layout()
            fig.savefig(figures/"cat_bp_equivalence.pdf", bbox_inches="tight")
            fig.savefig(figures/"cat_bp_equivalence.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def latex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_tables(
    summary,
    comparisons,
    rank_frame,
    aggregate,
    equivalence,
    global_tests,
    relaxation_validation,
    overdamped_validation,
    output,
):
    directory = output / "tables"; directory.mkdir(parents=True, exist_ok=True)
    with (directory/"main_results.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Requires booktabs and graphicx.\n\\par\\noindent\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{llrrrrr}\n\\toprule\nDataset & Method & Accuracy & Macro-F1 & Loss & Best epoch & Time (s) \\\\\n\\midrule\n")
        for row in summary.itertuples(index=False):
            handle.write(f"{latex_escape(row.dataset)} & {latex_escape(row.method)} & {row.test_accuracy_mean:.4f} $\\pm$ {row.test_accuracy_std:.4f} & {row.macro_f1_mean:.4f} & {row.test_loss_mean:.4f} $\\pm$ {row.test_loss_std:.4f} & {row.best_epoch_mean:.1f} & {row.elapsed_seconds_mean:.1f} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}%\n}\\par\n")
    with (directory/"paired_comparisons.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Negative loss difference favors the method.\n\\par\\noindent\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lllrrrr}\n\\toprule\nDataset & Method & Baseline & $n$ & $\\Delta$ loss & 95\\% CI & Holm $p$ \\\\\n\\midrule\n")
        for row in comparisons.itertuples(index=False):
            handle.write(f"{latex_escape(row.dataset)} & {latex_escape(row.method)} & {latex_escape(row.baseline)} & {row.n_pairs} & {row.loss_difference_mean:+.4f} & [{row.loss_difference_ci95_low:+.4f}, {row.loss_difference_ci95_high:+.4f}] & {row.wilcoxon_loss_p_holm:.4f} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}%\n}\\par\n")
    with (directory/"average_ranks.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrr}\n\\toprule\nMethod & Mean rank & Observations \\\\\n\\midrule\n")
        for row in rank_frame.itertuples(index=False): handle.write(f"{latex_escape(row.method)} & {row.mean_loss_rank:.4f} & {row.observations} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
    aggregate.to_csv(directory/"aggregate_comparisons_table.csv", index=False)
    equivalence.to_csv(directory/"equivalence_tests_table.csv", index=False)
    global_tests.to_csv(directory/"global_tests_table.csv", index=False)
    with (directory/"aggregate_comparisons.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Dataset-hierarchical confidence intervals. Negative loss difference favors the method.\n")
        handle.write("\\par\\noindent\\resizebox{\\linewidth}{!}{%\n")
        handle.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        handle.write("Method & Baseline & $n$ & $\\Delta$ loss & Hierarchical 95\\% CI & Dataset wins & Holm $p$ \\\\\n\\midrule\n")
        for row in aggregate.itertuples(index=False):
            handle.write(
                f"{latex_escape(row.method)} & {latex_escape(row.baseline)} & {row.n_dataset_seed_pairs} & "
                f"{row.loss_difference_mean:+.5f} & [{row.loss_difference_hierarchical_ci95_low:+.5f}, "
                f"{row.loss_difference_hierarchical_ci95_high:+.5f}] & {row.dataset_mean_loss_win_count}/{row.n_datasets} & "
                f"{row.wilcoxon_loss_p_dataset_means_holm:.4f} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}%\n}\\par\n")
    with (directory/"equivalence_tests.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lllrrrr}\n\\toprule\n")
        handle.write("Method & Metric & Margin & Difference & 95\\% CI & TOST $p$ & Equivalent \\\\\n\\midrule\n")
        for row in equivalence.itertuples(index=False):
            label = "yes" if row.equivalent_at_0_05 else "no"
            handle.write(
                f"{latex_escape(row.method)} & {latex_escape(row.metric)} & {row.margin:.4f} & "
                f"{row.dataset_mean_difference:+.6f} & [{row.difference_ci95_low:+.6f}, {row.difference_ci95_high:+.6f}] & "
                f"{row.tost_p:.4f} & {label} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    with (directory/"global_tests.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        handle.write("Level & Blocks & Methods & Friedman $\\chi^2$ & $p$ \\\\\n\\midrule\n")
        for row in global_tests.itertuples(index=False):
            handle.write(
                f"{latex_escape(row.analysis_level)} & {row.n_blocks} & {row.n_methods} & "
                f"{row.friedman_statistic:.4f} & {row.friedman_p:.3g} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    if not relaxation_validation.empty:
        depth6 = relaxation_validation[relaxation_validation.affine_depth == 6]
        relaxation_table = depth6.groupby(
            ["credit_rule", "local_steps"], as_index=False
        ).agg(
            relative_error=("gradient_relative_error", "mean"),
            gradient_cosine=("gradient_cosine", "mean"),
            neighbor_actions=("neighbor_jacobian_actions", "first"),
        )
        relaxation_table.to_csv(directory/"cat_ar_relaxation_table.csv", index=False)
        with (directory/"cat_ar_relaxation.tex").open("w", encoding="utf-8") as handle:
            handle.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
            handle.write("Rule & Sweeps & Neighbor actions & Relative error & Cosine \\\\\n\\midrule\n")
            for row in relaxation_table.itertuples(index=False):
                handle.write(
                    f"{latex_escape(row.credit_rule)} & {int(row.local_steps)} & "
                    f"{int(row.neighbor_actions)} & {row.relative_error:.6g} & "
                    f"{row.gradient_cosine:.6f} \\\\\n"
                )
            handle.write("\\bottomrule\n\\end{tabular}\n")
    if not overdamped_validation.empty:
        audit = pd.DataFrame([{
            "cases": len(overdamped_validation),
            "configured_ar_rate": float(overdamped_validation.configured_ar_rate.iloc[0]),
            "derived_overdamped_rate": float(overdamped_validation.derived_overdamped_rate.iloc[0]),
            "maximum_rate_difference": float(overdamped_validation.rate_absolute_difference.max()),
            "maximum_gradient_relative_difference": float(overdamped_validation.gradient_relative_difference.max()),
            "maximum_gradient_absolute_difference": float(overdamped_validation.gradient_max_absolute_difference.max()),
            "minimum_gradient_cosine": float(overdamped_validation.gradient_cosine.min()),
        }])
        audit.to_csv(directory/"overdamped_limit_audit_table.csv", index=False)


def write_report(
    config,
    tasks,
    final,
    failures,
    rank_frame,
    validation,
    relaxation_validation,
    overdamped_validation,
    aggregate,
    equivalence,
    global_tests,
    output,
):
    depth6 = validation[(validation.affine_depth == 6) & (validation.local_steps.isin([40, 64, 100, 160]))].groupby("local_steps").agg(error=("gradient_relative_error", "mean"), cosine=("gradient_cosine", "mean")) if not validation.empty else pd.DataFrame()
    lines = ["# CAT-MWBP Neural Processing Letters reproducibility suite", "", f"- Completed runs: {len(final)} / {len(tasks)}", f"- Failed runs: {len(failures)}", f"- Configuration hash: `{fingerprint(config)}`", "- Test set used only after restoring the best validation checkpoint.", "- Feature scaler fitted only on the optimization-training partition.", "- Aggregate confidence intervals resample datasets first and seeds second.", "- Equivalence margins and the AR overdamped rate were fixed in the configuration before test evaluation.", "- CAT and AR use equal synchronous local sweeps and equal one-hop transpose-Jacobian action counts.", "", "## Aggregate result", ""]
    if not rank_frame.empty: lines.append(f"Lowest core test-loss rank: {rank_frame.iloc[0].method} ({rank_frame.iloc[0].mean_loss_rank:.3f}).")
    if not global_tests.empty:
        row = global_tests[global_tests.analysis_level == "dataset_mean"]
        if not row.empty:
            value = row.iloc[0]
            lines.append(f"Dataset-level Friedman test: chi-square={value.friedman_statistic:.4f}, p={value.friedman_p:.6g}.")
    if not aggregate.empty:
        lines.extend(["", "## Primary aggregate comparisons", ""])
        for row in aggregate.itertuples(index=False):
            lines.append(
                f"- {row.method} vs {row.baseline}: delta loss {row.loss_difference_mean:+.6g}, "
                f"hierarchical 95% CI [{row.loss_difference_hierarchical_ci95_low:+.6g}, "
                f"{row.loss_difference_hierarchical_ci95_high:+.6g}], dataset wins "
                f"{row.dataset_mean_loss_win_count}/{row.n_datasets}."
            )
    if not equivalence.empty:
        lines.extend(["", "## Practical equivalence to BP", ""])
        for row in equivalence.itertuples(index=False):
            decision = "supported" if row.equivalent_at_0_05 else "not supported"
            lines.append(
                f"- {row.method}, {row.metric}, margin +/-{row.margin:g}: {decision} "
                f"(TOST p={row.tost_p:.6g}; difference {row.dataset_mean_difference:+.6g})."
            )
    if not depth6.empty:
        lines.extend(["", "## Six-block gradient validation", ""])
        for steps, row in depth6.iterrows(): lines.append(f"- CAT{steps}: relative error {row.error:.6g}, cosine {row.cosine:.6f}.")
    if not relaxation_validation.empty:
        comparison = relaxation_validation[
            (relaxation_validation.affine_depth == 6)
            & (relaxation_validation.local_steps == config.ar_steps)
        ].groupby("credit_rule").agg(
            error=("gradient_relative_error", "mean"),
            cosine=("gradient_cosine", "mean"),
        )
        if not comparison.empty:
            lines.extend(["", "## Activation Relaxation comparison", ""])
            for rule, row in comparison.iterrows():
                lines.append(
                    f"- {rule}{config.ar_steps}: relative error {row.error:.6g}, "
                    f"cosine {row.cosine:.6f} at the same local sweep budget."
                )
        cat_ar = aggregate[
            (aggregate.method == "CAT64-Momentum")
            & (aggregate.baseline == "AR64-Momentum")
        ]
        if not cat_ar.empty:
            row = cat_ar.iloc[0]
            lines.append(
                f"- CAT64-Momentum vs AR64-Momentum: delta loss "
                f"{row.loss_difference_mean:+.6g}, hierarchical 95% CI "
                f"[{row.loss_difference_hierarchical_ci95_low:+.6g}, "
                f"{row.loss_difference_hierarchical_ci95_high:+.6g}], median runtime "
                f"ratio {row.runtime_ratio_median:.3f}."
            )
    if not overdamped_validation.empty:
        lines.extend([
            "",
            "## Overdamped-limit audit",
            "",
            f"- Configured AR rate: {overdamped_validation.configured_ar_rate.iloc[0]:.12g}.",
            f"- Derived CAT overdamped rate: {overdamped_validation.derived_overdamped_rate.iloc[0]:.12g}.",
            f"- Maximum absolute gradient difference across {len(overdamped_validation)} audit cases: "
            f"{overdamped_validation.gradient_max_absolute_difference.max():.6g}.",
            f"- Maximum relative gradient difference: "
            f"{overdamped_validation.gradient_relative_difference.max():.6g}.",
        ])
    em = final[final.method == "CAT64-MWBP-local-EM"]
    if not em.empty:
        relative = np.abs(em.calibration_achieved_energy-em.calibration_target_energy)/(em.calibration_target_energy+1e-15)
        lines.extend(["", "## Energy-matched ablation", "", f"- Mean calibrated local coupling: {em.calibrated_local_coupling.mean():.6f}.", f"- Mean relative calibration mismatch: {relative.mean():.6g}."])
    lines.extend(["", "## Interpretation guardrail", "", "Do not claim optimizer superiority from ranks alone. Report paired differences, confidence intervals, corrected p-values, runtime, early-stopping behavior, and the distinction between local credit transport and metric propagation."])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    draft = [
        "# Automatically generated NPL results draft",
        "",
        "This text is a numerical draft, not a claim-selection mechanism. Retain the limitations and verify every value against the generated CSV tables before submission.",
        "",
        f"The validation-controlled protocol completed {len(final)} of {len(tasks)} planned runs with {len(failures)} failures. Test observations were evaluated only after restoration of the minimum-validation-loss checkpoint, and feature scaling was fitted on the training partition.",
    ]
    cat64 = aggregate[(aggregate.method == "CAT64-Momentum") & (aggregate.baseline == "BP-Momentum")]
    if not cat64.empty:
        row = cat64.iloc[0]
        draft.append(
            f"Across the core datasets and seeds, CAT64-Momentum differed from BP-Momentum by {row.loss_difference_mean:+.6g} in test loss "
            f"(dataset-hierarchical 95% CI [{row.loss_difference_hierarchical_ci95_low:+.6g}, {row.loss_difference_hierarchical_ci95_high:+.6g}]) "
            f"and {row.accuracy_difference_mean:+.6g} in test accuracy. Its median runtime ratio was {row.runtime_ratio_median:.3f}."
        )
    ar64 = aggregate[(aggregate.method == "AR64-Momentum") & (aggregate.baseline == "BP-Momentum")]
    cat_ar = aggregate[(aggregate.method == "CAT64-Momentum") & (aggregate.baseline == "AR64-Momentum")]
    if not ar64.empty:
        row = ar64.iloc[0]
        draft.append(
            f"The matched first-order Activation Relaxation baseline differed from BP-Momentum by "
            f"{row.loss_difference_mean:+.6g} in test loss (dataset-hierarchical 95% CI "
            f"[{row.loss_difference_hierarchical_ci95_low:+.6g}, {row.loss_difference_hierarchical_ci95_high:+.6g}])."
        )
    if not cat_ar.empty:
        row = cat_ar.iloc[0]
        draft.append(
            f"At equal 64-sweep local communication budgets, second-order CAT differed from "
            f"first-order AR by {row.loss_difference_mean:+.6g} in test loss "
            f"(dataset-hierarchical 95% CI [{row.loss_difference_hierarchical_ci95_low:+.6g}, "
            f"{row.loss_difference_hierarchical_ci95_high:+.6g}])."
        )
    if not overdamped_validation.empty:
        draft.append(
            f"The independently implemented overdamped CAT update matched AR across "
            f"{len(overdamped_validation)} audit cases with maximum absolute gradient difference "
            f"{overdamped_validation.gradient_max_absolute_difference.max():.6g}."
        )
    propagated = aggregate[(aggregate.method == "CAT64-MWBP") & (aggregate.baseline == "CAT64-MWBP-local-EM")]
    if not propagated.empty:
        row = propagated.iloc[0]
        draft.append(
            f"The propagated metric variant differed from the train-only energy-matched local control by {row.loss_difference_mean:+.6g} in test loss "
            f"(dataset-hierarchical 95% CI [{row.loss_difference_hierarchical_ci95_low:+.6g}, {row.loss_difference_hierarchical_ci95_high:+.6g}])."
        )
    if not depth6.empty and 64 in depth6.index:
        row = depth6.loc[64]
        draft.append(f"At six affine blocks, 64 CAT steps produced mean relative gradient error {row.error:.6g} and cosine alignment {row.cosine:.6f}.")
    draft.append("The results treat CAT as a second-order extension of first-order Activation Relaxation rather than as the first local adjoint relaxation method. They support finite-step local credit transport with an exact-backpropagation equilibrium, but do not establish metric-wave optimizer superiority.")
    (output/"NPL_RESULTS_DRAFT.md").write_text("\n\n".join(value for value in draft if value)+"\n", encoding="utf-8")


def metadata(config, tasks, output, tests):
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception: commit = None
    value = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_hash": fingerprint(config), "config": config_payload(config), "expected_runs": len(tasks), "unit_tests": tests, "git_commit": commit, "platform": platform.platform(), "python": sys.version, "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__}}
    (output/"suite_metadata.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def manifest(output):
    rows=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.txt" and "checkpoints" not in path.parts:
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}")
    (output/"manifest_sha256.txt").write_text("\n".join(rows)+"\n", encoding="utf-8")


def run_tests(output):
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output/"unit_tests.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode: raise RuntimeError("unit tests failed")
    return "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "standard", "full"), default="quick")
    parser.add_argument("--output", type=Path, default=ROOT/"npl_results")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-gradient-validation", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1: parser.error("--jobs must be positive")
    config = profile_config(args.profile); tasks = task_plan(config); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    derived_rate = overdamped_relaxation_rate(
        config.adjoint_dt, config.adjoint_damping, config.adjoint_frequency
    )
    if not np.isclose(config.ar_relaxation_rate, derived_rate, rtol=0.0, atol=1e-15):
        raise SystemExit(
            f"AR rate {config.ar_relaxation_rate} does not match the predeclared "
            f"overdamped CAT rate {derived_rate}"
        )
    existing = output/"suite_metadata.json"
    if existing.exists() and not args.force:
        old = json.loads(existing.read_text()).get("config_hash")
        if old != fingerprint(config): raise SystemExit("output contains a different configuration")
    (output/"suite_config.json").write_text(json.dumps(config_payload(config), indent=2)+"\n", encoding="utf-8")
    test_status = "skipped" if args.skip_unit_tests else run_tests(output)
    history, final, failures = execute(tasks, output, args.jobs, args.force)
    if history.empty or final.empty: raise SystemExit("no successful runs")
    validation_path = output/"gradient_validation.csv"
    relaxation_path = output/"relaxation_validation.csv"
    overdamped_path = output/"overdamped_limit_validation.csv"
    if args.skip_gradient_validation:
        validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
        relaxation_validation = pd.read_csv(relaxation_path) if relaxation_path.exists() else pd.DataFrame()
        overdamped_validation = pd.read_csv(overdamped_path) if overdamped_path.exists() else pd.DataFrame()
    else:
        relaxation_validation = validate_relaxation_methods(
            config, quick=args.profile == "quick"
        )
        relaxation_validation.to_csv(relaxation_path, index=False)
        validation = relaxation_validation[
            relaxation_validation.credit_rule == "CAT"
        ].copy()
        validation.to_csv(validation_path, index=False)
        overdamped_validation = validate_overdamped_limit(
            config, quick=args.profile == "quick"
        )
        overdamped_validation.to_csv(overdamped_path, index=False)
    summary = summarize(final)
    comparisons = paired(final)
    rank_frame = ranks(final)
    aggregate = aggregate_paired(final, config)
    equivalence = equivalence_tests(final, config)
    global_tests = global_rank_tests(final)
    history.to_csv(output/"history.csv", index=False)
    final.to_csv(output/"final_runs.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    comparisons.to_csv(output/"paired_comparisons.csv", index=False)
    rank_frame.to_csv(output/"average_ranks.csv", index=False)
    aggregate.to_csv(output/"aggregate_comparisons.csv", index=False)
    equivalence.to_csv(output/"equivalence_tests.csv", index=False)
    global_tests.to_csv(output/"global_tests.csv", index=False)
    make_figures(
        history, final, summary, validation, relaxation_validation,
        overdamped_validation, rank_frame, aggregate, equivalence, output
    )
    write_tables(
        summary, comparisons, rank_frame, aggregate, equivalence, global_tests,
        relaxation_validation, overdamped_validation, output
    )
    write_report(
        config, tasks, final, failures, rank_frame, validation,
        relaxation_validation, overdamped_validation, aggregate, equivalence,
        global_tests, output
    )
    metadata(config, tasks, output, test_status)
    manifest(output)
    print(f"Completed {len(final)} / {len(tasks)} runs. Results: {output}", flush=True)
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
