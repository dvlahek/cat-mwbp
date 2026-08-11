"""Validation-controlled training without per-epoch access to the test set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .model import Array, MLP
from .optimizers import Optimizer
from .training import evaluate


GradientFunction = Callable[[MLP, Array, Array], Tuple[float, Sequence[Array], Dict[str, float]]]


@dataclass
class ValidationTrainingResult:
    history: List[Dict[str, float]]
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float
    early_stopped: bool


def train_with_validation(
    model: MLP,
    optimizer: Optimizer,
    gradient_function: GradientFunction,
    x_train: Array,
    y_train: Array,
    x_validation: Array,
    y_validation: Array,
    max_epochs: int = 150,
    patience: int = 20,
    min_delta: float = 1e-4,
    batch_size: int = 64,
    seed: int = 0,
) -> ValidationTrainingResult:
    """Train, select only on validation loss, and restore the best parameters."""
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("max_epochs, patience, and batch_size must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    rng = np.random.default_rng(seed)
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_blocks = [block.copy() for block in model.parameter_blocks()]
    stale_epochs = 0
    n = x_train.shape[0]
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n)
        batch_losses = []
        diagnostic_values: Dict[str, List[float]] = {}
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss, gradients, diagnostics = gradient_function(model, x_train[idx], y_train[idx])
            model.set_parameter_blocks(optimizer.step(model.parameter_blocks(), gradients))
            batch_losses.append(float(loss))
            for key, value in diagnostics.items():
                diagnostic_values.setdefault(key, []).append(float(value))
        train = evaluate(model, x_train, y_train)
        validation = evaluate(model, x_validation, y_validation)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        row.update({key: float(np.mean(values)) for key, values in diagnostic_values.items()})
        row.update(optimizer.diagnostics())
        history.append(row)
        if validation["loss"] < best_loss - min_delta:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_blocks = [block.copy() for block in model.parameter_blocks()]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.set_parameter_blocks(best_blocks)
    return ValidationTrainingResult(
        history=history,
        best_epoch=best_epoch,
        stopped_epoch=len(history),
        best_validation_loss=float(best_loss),
        early_stopped=len(history) < max_epochs,
    )
