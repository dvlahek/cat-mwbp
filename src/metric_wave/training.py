"""Training and evaluation utilities, including controlled gradient delays."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence

import numpy as np

from .model import MLP
from .optimizers import Optimizer
from .local_adjoint import local_adjoint_gradients


Array = np.ndarray


def evaluate(model: MLP, x: Array, y: Array) -> Dict[str, float]:
    loss = model.loss(x, y)
    accuracy = float(np.mean(model.predict(x) == y))
    return {"loss": loss, "accuracy": accuracy}


def train_model(
    model: MLP,
    optimizer: Optimizer,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    epochs: int = 80,
    batch_size: int = 64,
    seed: int = 0,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    history: List[Dict[str, float]] = []
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss, gradients = model.loss_and_gradients(x_train[idx], y_train[idx])
            parameters = optimizer.step(model.parameter_blocks(), gradients)
            model.set_parameter_blocks(parameters)
            batch_losses.append(loss)
        train = evaluate(model, x_train, y_train)
        test = evaluate(model, x_test, y_test)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
        }
        row.update(optimizer.diagnostics())
        history.append(row)
    return history


def train_model_local_adjoint(
    model: MLP,
    optimizer: Optimizer,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    adjoint_steps: int = 40,
    adjoint_dt: float = 0.04,
    adjoint_damping: float = 8.0,
    adjoint_frequency: float = 8.0,
    epochs: int = 80,
    batch_size: int = 64,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Train using locally relaxed adjoints instead of a hidden backward pass."""
    rng = np.random.default_rng(seed)
    history: List[Dict[str, float]] = []
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        batch_losses = []
        residuals = []
        adjoint_norms = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss, gradients, diagnostics = local_adjoint_gradients(
                model,
                x_train[idx],
                y_train[idx],
                steps=adjoint_steps,
                dt=adjoint_dt,
                damping=adjoint_damping,
                frequency=adjoint_frequency,
            )
            parameters = optimizer.step(model.parameter_blocks(), gradients)
            model.set_parameter_blocks(parameters)
            batch_losses.append(loss)
            residuals.append(diagnostics["adjoint_residual"])
            adjoint_norms.append(diagnostics["adjoint_norm"])
        train = evaluate(model, x_train, y_train)
        test = evaluate(model, x_test, y_test)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "adjoint_steps": float(adjoint_steps),
            "adjoint_residual": float(np.mean(residuals)),
            "adjoint_norm": float(np.mean(adjoint_norms)),
        }
        row.update(optimizer.diagnostics())
        history.append(row)
    return history


class BlockDelayBuffer:
    """Apply an exact, independently specified delay to each gradient block.

    A delay of ``d`` means that the block used at optimizer step ``t`` was
    computed at step ``t-d``. Missing warm-up entries are represented by
    zeros, so every optimizer receives the same controlled staleness pattern.
    """

    def __init__(self, delays: Sequence[int]):
        if any(int(delay) < 0 for delay in delays):
            raise ValueError("delays must be non-negative")
        self.delays = tuple(int(delay) for delay in delays)
        self.queues: List[Deque[Array]] = [deque() for _ in self.delays]

    def push(self, gradients: Sequence[Array]) -> List[Array]:
        if len(gradients) != len(self.delays):
            raise ValueError("one delay is required for each gradient block")
        delayed: List[Array] = []
        for queue, delay, gradient in zip(self.queues, self.delays, gradients):
            queue.append(np.array(gradient, copy=True))
            if len(queue) > delay:
                delayed.append(queue.popleft())
            else:
                delayed.append(np.zeros_like(gradient))
        return delayed


def train_model_delayed(
    model: MLP,
    optimizer: Optimizer,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    block_delays: Sequence[int],
    epochs: int = 100,
    batch_size: int = 64,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Train with layer-dependent stale gradients under a fixed schedule."""
    if len(block_delays) != model.n_layers:
        raise ValueError("block_delays must match the number of affine layers")
    rng = np.random.default_rng(seed)
    delay_buffer = BlockDelayBuffer(block_delays)
    history: List[Dict[str, float]] = []
    n = x_train.shape[0]
    optimizer_step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss, gradients = model.loss_and_gradients(x_train[idx], y_train[idx])
            stale_gradients = delay_buffer.push(gradients)
            parameters = optimizer.step(model.parameter_blocks(), stale_gradients)
            model.set_parameter_blocks(parameters)
            batch_losses.append(loss)
            optimizer_step += 1
        train = evaluate(model, x_train, y_train)
        test = evaluate(model, x_test, y_test)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "optimizer_step": float(optimizer_step),
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "max_block_delay": float(max(block_delays, default=0)),
            "mean_block_delay": float(np.mean(block_delays)),
        }
        row.update(optimizer.diagnostics())
        history.append(row)
    return history
