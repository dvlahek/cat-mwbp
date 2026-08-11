"""Local adjoint relaxation rules for feed-forward tanh networks.

The module contains both the second-order Causal Adjoint Transport (CAT)
scheme and a first-order Activation Relaxation (AR) reference.  The two rules
share the same output boundary, one-hop transpose-Jacobian action, synchronous
update schedule, and local gradient construction.  This makes the empirical
comparison isolate the order of the relaxation dynamics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .model import Array, LayerCache, MLP


def output_boundary(model: MLP, logits: Array, y: Array) -> Tuple[float, Array]:
    """Return cross-entropy and its local derivative at the output port."""
    probs = model._softmax(logits)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    delta = probs.copy()
    delta[np.arange(n), y] -= 1.0
    delta /= n
    return float(loss), delta


def local_target(model: MLP, cache: Sequence[LayerCache], deltas: Sequence[Array], k: int) -> Array:
    """One-hop transpose-Jacobian action from layer ``k+1`` to layer ``k``."""
    transported = deltas[k + 1] @ model.weights[k + 1].T
    derivative = 1.0 - np.tanh(cache[k].preactivation) ** 2
    return transported * derivative


def relax_local_adjoint(
    model: MLP,
    cache: Sequence[LayerCache],
    boundary: Array,
    steps: int = 40,
    dt: float = 0.04,
    damping: float = 8.0,
    frequency: float = 8.0,
    return_trace: bool = False,
) -> Tuple[List[Array], Dict[str, float], List[List[Array]]]:
    r"""Relax a damped local adjoint wave with the output port clamped.

    Hidden adjoints obey

    ``delta'' + 2*damping*delta' + frequency^2*(delta-target) = 0``.

    Every substep uses only the previous state of a layer and its immediate
    downstream neighbor. This synchronous update preserves exact finite-hop
    causality. At equilibrium, ``delta_k = J_{k+1}^T delta_{k+1}``.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0 or damping <= 0.0 or frequency <= 0.0:
        raise ValueError("dt, damping, and frequency must be positive")
    deltas = [np.zeros_like(layer.preactivation) for layer in cache]
    velocities = [np.zeros_like(layer.preactivation) for layer in cache]
    deltas[-1] = np.array(boundary, copy=True)
    trace: List[List[Array]] = []
    if return_trace:
        trace.append([value.copy() for value in deltas])
    for _ in range(steps):
        old = [value.copy() for value in deltas]
        next_deltas = [value.copy() for value in deltas]
        next_velocities = [value.copy() for value in velocities]
        for k in range(model.n_layers - 1):
            target = local_target(model, cache, old, k)
            residual = old[k] - target
            acceleration = -2.0 * damping * velocities[k] - frequency ** 2 * residual
            next_velocities[k] = velocities[k] + dt * acceleration
            next_deltas[k] = old[k] + dt * next_velocities[k]
        next_deltas[-1] = np.array(boundary, copy=True)
        next_velocities[-1].fill(0.0)
        deltas, velocities = next_deltas, next_velocities
        if return_trace:
            trace.append([value.copy() for value in deltas])

    residual_sq = 0.0
    adjoint_sq = 0.0
    for k in range(model.n_layers - 1):
        residual = deltas[k] - local_target(model, cache, deltas, k)
        residual_sq += float(np.sum(residual ** 2))
        adjoint_sq += float(np.sum(deltas[k] ** 2))
    adjoint_sq += float(np.sum(deltas[-1] ** 2))
    diagnostics = {
        "adjoint_residual": float(np.sqrt(residual_sq)),
        "adjoint_norm": float(np.sqrt(adjoint_sq)),
        "relaxation_sweeps": float(steps),
        "neighbor_jacobian_actions": float(steps * max(0, model.n_layers - 1)),
        "credit_dynamics_order": 2.0,
    }
    return deltas, diagnostics, trace


def overdamped_relaxation_rate(dt: float, damping: float, frequency: float) -> float:
    r"""Return the first-order rate implied by the overdamped CAT equation.

    Neglecting ``delta''`` in

    ``delta'' + 2*gamma*delta' + omega^2*(delta-target) = 0``

    gives ``delta' = omega^2/(2*gamma)*(target-delta)``.  One explicit Euler
    step of size ``dt`` therefore has rate ``omega^2*dt/(2*gamma)``.
    """
    if dt <= 0.0 or damping <= 0.0 or frequency <= 0.0:
        raise ValueError("dt, damping, and frequency must be positive")
    return float(frequency ** 2 * dt / (2.0 * damping))


def _first_order_diagnostics(
    model: MLP,
    cache: Sequence[LayerCache],
    deltas: Sequence[Array],
    steps: int,
    rate: float,
    label: str,
) -> Dict[str, float]:
    residual_sq = 0.0
    adjoint_sq = 0.0
    for k in range(model.n_layers - 1):
        residual = deltas[k] - local_target(model, cache, deltas, k)
        residual_sq += float(np.sum(residual ** 2))
        adjoint_sq += float(np.sum(deltas[k] ** 2))
    adjoint_sq += float(np.sum(deltas[-1] ** 2))
    return {
        "adjoint_residual": float(np.sqrt(residual_sq)),
        "adjoint_norm": float(np.sqrt(adjoint_sq)),
        "relaxation_sweeps": float(steps),
        "neighbor_jacobian_actions": float(steps * max(0, model.n_layers - 1)),
        "credit_dynamics_order": 1.0,
        "first_order_rate": float(rate),
        "first_order_rule_code": 1.0 if label == "ar" else 2.0,
    }


def relax_activation_adjoint(
    model: MLP,
    cache: Sequence[LayerCache],
    boundary: Array,
    steps: int = 64,
    rate: float = 0.16,
    return_trace: bool = False,
) -> Tuple[List[Array], Dict[str, float], List[List[Array]]]:
    r"""First-order Activation Relaxation reference with synchronous updates.

    The hidden fields obey ``delta' = target-delta`` after absorbing the time
    scale into ``rate``.  This implementation is an independently written,
    mechanistically matched baseline; it is not the original authors' code.
    Every sweep uses the old downstream state, so it has the same finite-hop
    schedule and the same number of one-hop Jacobian actions as CAT.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must lie in (0, 1]")
    deltas = [np.zeros_like(layer.preactivation) for layer in cache]
    deltas[-1] = np.array(boundary, copy=True)
    trace: List[List[Array]] = []
    if return_trace:
        trace.append([value.copy() for value in deltas])
    for _ in range(steps):
        old = [value.copy() for value in deltas]
        next_deltas = [value.copy() for value in deltas]
        for k in range(model.n_layers - 1):
            target = local_target(model, cache, old, k)
            next_deltas[k] = old[k] + rate * (target - old[k])
        next_deltas[-1] = np.array(boundary, copy=True)
        deltas = next_deltas
        if return_trace:
            trace.append([value.copy() for value in deltas])
    diagnostics = _first_order_diagnostics(model, cache, deltas, steps, rate, "ar")
    return deltas, diagnostics, trace


def relax_overdamped_adjoint(
    model: MLP,
    cache: Sequence[LayerCache],
    boundary: Array,
    steps: int = 64,
    dt: float = 0.04,
    damping: float = 8.0,
    frequency: float = 8.0,
    return_trace: bool = False,
) -> Tuple[List[Array], Dict[str, float], List[List[Array]]]:
    r"""Integrate the explicit first-order equation obtained from CAT.

    This path is intentionally separate from :func:`relax_activation_adjoint`
    so ``overdamped_limit_validation.csv`` can numerically audit the algebraic
    reduction instead of comparing a function with itself.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    rate = overdamped_relaxation_rate(dt, damping, frequency)
    if not 0.0 < rate <= 1.0:
        raise ValueError("the explicit overdamped rate must lie in (0, 1]")
    deltas = [np.zeros_like(layer.preactivation) for layer in cache]
    deltas[-1] = np.array(boundary, copy=True)
    trace: List[List[Array]] = []
    if return_trace:
        trace.append([value.copy() for value in deltas])
    coefficient = frequency ** 2 / (2.0 * damping)
    for _ in range(steps):
        old = [value.copy() for value in deltas]
        next_deltas = [value.copy() for value in deltas]
        for k in range(model.n_layers - 1):
            target = local_target(model, cache, old, k)
            derivative = coefficient * (target - old[k])
            next_deltas[k] = old[k] + dt * derivative
        next_deltas[-1] = np.array(boundary, copy=True)
        deltas = next_deltas
        if return_trace:
            trace.append([value.copy() for value in deltas])
    diagnostics = _first_order_diagnostics(model, cache, deltas, steps, rate, "cat_od")
    return deltas, diagnostics, trace


def gradients_from_local_adjoint(cache: Sequence[LayerCache], deltas: Sequence[Array]) -> List[Array]:
    """Construct each parameter gradient using only its activation and adjoint."""
    gradients: List[Array] = []
    for layer, delta in zip(cache, deltas):
        grad_w = layer.input.T @ delta
        grad_b = delta.sum(axis=0)
        gradients.append(np.concatenate((grad_w.ravel(), grad_b.ravel())))
    return gradients


def local_adjoint_gradients(
    model: MLP,
    x: Array,
    y: Array,
    steps: int = 40,
    dt: float = 0.04,
    damping: float = 8.0,
    frequency: float = 8.0,
) -> Tuple[float, List[Array], Dict[str, float]]:
    """Forward pass, local adjoint relaxation, and local parameter gradients."""
    logits, cache = model.forward(x)
    loss, boundary = output_boundary(model, logits, y)
    deltas, diagnostics, _ = relax_local_adjoint(
        model, cache, boundary, steps, dt, damping, frequency
    )
    return loss, gradients_from_local_adjoint(cache, deltas), diagnostics


def activation_relaxation_gradients(
    model: MLP,
    x: Array,
    y: Array,
    steps: int = 64,
    rate: float = 0.16,
) -> Tuple[float, List[Array], Dict[str, float]]:
    """Forward pass and the matched first-order AR credit baseline."""
    logits, cache = model.forward(x)
    loss, boundary = output_boundary(model, logits, y)
    deltas, diagnostics, _ = relax_activation_adjoint(
        model, cache, boundary, steps=steps, rate=rate
    )
    return loss, gradients_from_local_adjoint(cache, deltas), diagnostics


def overdamped_adjoint_gradients(
    model: MLP,
    x: Array,
    y: Array,
    steps: int = 64,
    dt: float = 0.04,
    damping: float = 8.0,
    frequency: float = 8.0,
) -> Tuple[float, List[Array], Dict[str, float]]:
    """Forward pass and the independently integrated overdamped CAT limit."""
    logits, cache = model.forward(x)
    loss, boundary = output_boundary(model, logits, y)
    deltas, diagnostics, _ = relax_overdamped_adjoint(
        model,
        cache,
        boundary,
        steps=steps,
        dt=dt,
        damping=damping,
        frequency=frequency,
    )
    return loss, gradients_from_local_adjoint(cache, deltas), diagnostics


def gradient_comparison(exact: Sequence[Array], approximate: Sequence[Array]) -> Dict[str, float]:
    """Relative error and cosine alignment, used only for validation."""
    exact_flat = np.concatenate([g.ravel() for g in exact])
    approx_flat = np.concatenate([g.ravel() for g in approximate])
    exact_norm = float(np.linalg.norm(exact_flat))
    approx_norm = float(np.linalg.norm(approx_flat))
    relative = float(np.linalg.norm(approx_flat - exact_flat) / (exact_norm + 1e-15))
    cosine = float(exact_flat @ approx_flat / (exact_norm * approx_norm + 1e-15))
    return {"gradient_relative_error": relative, "gradient_cosine": cosine}
