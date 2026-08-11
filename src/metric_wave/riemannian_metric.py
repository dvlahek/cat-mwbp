"""Layerwise pullback metrics and finite-hop output-factor transport.

The instantaneous optimizer uses ``G_k = J_k.T M J_k + mass*I``.  The
transported control does not claim to move the full, block-specific ``G_k``:
it transports a factor of the common output metric through the block graph and
combines the arrived factor with each block's own output Jacobian. This keeps
output-factor transport distinct from transport of a full block metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Array, MLP
from .optimizers import Optimizer


def layer_output_jacobians(model: MLP, x: Array) -> List[Array]:
    """Return per-sample logit Jacobians with shape ``(n, n_out, p_k)``."""
    logits, cache = model.forward(x)
    n, n_out = logits.shape
    deltas: List[Array] = [None] * model.n_layers  # type: ignore[list-item]
    deltas[-1] = np.broadcast_to(np.eye(n_out), (n, n_out, n_out)).copy()
    for k in range(model.n_layers - 2, -1, -1):
        upstream = deltas[k + 1] @ model.weights[k + 1].T
        derivative = 1.0 - np.tanh(cache[k].preactivation) ** 2
        deltas[k] = upstream * derivative[:, None, :]

    jacobians: List[Array] = []
    for layer, delta in zip(cache, deltas):
        grad_w = layer.input[:, None, :, None] * delta[:, :, None, :]
        grad_w = grad_w.reshape(n, n_out, -1)
        jacobians.append(np.concatenate((grad_w, delta), axis=2))
    return jacobians


def output_metric_factor(model: MLP, logits: Array, mode: str) -> Array:
    """Return an upper factor ``L`` satisfying ``M = L.T L``."""
    n_out = logits.shape[1]
    if mode == "identity":
        return np.eye(n_out)
    if mode == "gauss_newton":
        probs = model._softmax(logits)
        mean = probs.mean(axis=0)
        metric = np.diag(mean) - np.outer(mean, mean)
        metric += 1e-3 * np.eye(n_out)
        return np.linalg.cholesky(metric).T
    raise ValueError("output_metric must be 'identity' or 'gauss_newton'")


def woodbury_inverse_action(factor: Array, mass: float, vector: Array) -> Array:
    """Apply ``(mass*I + factor.T@factor)^-1`` without a parameter-size inverse."""
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if factor.shape[0] == 0:
        return vector / mass
    d_inv = 1.0 / mass
    inner = np.eye(factor.shape[0]) + d_inv * (factor @ factor.T)
    solved = np.linalg.solve(inner, factor @ vector)
    return d_inv * vector - d_inv**2 * (factor.T @ solved)


class _PullbackBase(Optimizer):
    def __init__(
        self,
        model: MLP,
        lr: float = 0.05,
        mass: float = 1.0,
        output_metric: str = "gauss_newton",
        metric_batch: int = 8,
        momentum: float = 0.9,
        seed: int = 0,
    ):
        if mass <= 0.0 or metric_batch < 1:
            raise ValueError("mass and metric_batch must be positive")
        self.model = model
        self.lr = float(lr)
        self.mass = float(mass)
        self.output_metric = output_metric
        self.metric_batch = int(metric_batch)
        self.momentum = float(momentum)
        self.rng = np.random.default_rng(seed)
        self.velocity: Optional[List[Array]] = None
        self._pending: Optional[Tuple[Array, Array]] = None
        self._last_diag: Dict[str, float] = {}

    def set_metric_batch(self, x: Array, y: Array) -> None:
        self._pending = (x, y)

    def _sample(self) -> Tuple[Array, List[Array], Array]:
        if self._pending is None:
            raise RuntimeError("call set_metric_batch(x, y) before step()")
        x, _ = self._pending
        count = min(self.metric_batch, x.shape[0])
        idx = self.rng.choice(x.shape[0], size=count, replace=False)
        xm = x[idx]
        logits, _ = self.model.forward(xm)
        factor = output_metric_factor(self.model, logits, self.output_metric)
        return xm, layer_output_jacobians(self.model, xm), factor

    def _forces(self, gradients: Sequence[Array]) -> List[Array]:
        if self.momentum <= 0.0:
            return [np.asarray(g) for g in gradients]
        if self.velocity is None:
            self.velocity = [np.zeros_like(g) for g in gradients]
        self.velocity = [
            self.momentum * v + (1.0 - self.momentum) * g
            for v, g in zip(self.velocity, gradients)
        ]
        return self.velocity

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._last_diag)


class RiemannianPullback(_PullbackBase):
    """Instantaneous layerwise pullback metric control."""

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        xm, jacobians, output_factor = self._sample()
        factors = []
        for jac in jacobians:
            scaled = np.einsum("ij,njp->nip", output_factor, jac)
            factors.append(scaled.reshape(-1, jac.shape[2]) / np.sqrt(xm.shape[0]))
        forces = self._forces(gradients)
        directions = [
            woodbury_inverse_action(factor, self.mass, force)
            for factor, force in zip(factors, forces)
        ]
        raw = np.sqrt(sum(float(g @ g) for g in forces))
        pre = np.sqrt(sum(float(d @ d) for d in directions))
        self._last_diag = {
            "pullback_force_norm": float(raw),
            "pullback_direction_norm": float(pre),
            "pullback_precondition_ratio": float(pre / (raw + 1e-15)),
            "transport_steps": float(self.model.n_layers - 1),
            "transport_reached_blocks": float(self.model.n_layers),
            "transport_all_blocks_reached": 1.0,
        }
        self._pending = None
        return [p - self.lr * d for p, d in zip(parameters, directions)]


class OutputFactorTransportPullback(_PullbackBase):
    """Finite-hop transport of a common output-metric factor.

    ``metric_steps=None`` uses the graph diameter, which is the minimum number
    of synchronous steps required for non-zero support at every block.
    """

    def __init__(
        self,
        *args,
        metric_steps: Optional[int] = None,
        wave_speed: float = 1.0,
        relax_rate: float = 0.6,
        require_full_reach: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if metric_steps is not None and metric_steps < 0:
            raise ValueError("metric_steps must be non-negative or None")
        if not 0.0 <= relax_rate <= 1.0:
            raise ValueError("relax_rate must lie in [0, 1]")
        self.metric_steps = metric_steps
        self.wave_speed = float(wave_speed)
        self.relax_rate = float(relax_rate)
        self.require_full_reach = bool(require_full_reach)

    @property
    def graph_diameter(self) -> int:
        return max(0, self.model.n_layers - 1)

    def resolved_steps(self) -> int:
        return self.graph_diameter if self.metric_steps is None else self.metric_steps

    def _transported_factors(self) -> Tuple[List[Array], List[float]]:
        xm, jacobians, true_factor = self._sample()
        steps = self.resolved_steps()
        if (
            self.require_full_reach
            and self.wave_speed > 0.0
            and steps < self.graph_diameter
        ):
            raise ValueError(
                f"metric_steps={steps} cannot reach all blocks; "
                f"the graph diameter is {self.graph_diameter}"
            )

        field = [np.zeros_like(true_factor) for _ in range(self.model.n_layers)]
        field[-1] = true_factor.copy()
        for _ in range(steps):
            old = [value.copy() for value in field]
            for k in range(self.model.n_layers - 1):
                target = self.wave_speed * old[k + 1]
                field[k] = old[k] + self.relax_rate * (target - old[k])
            field[-1] = true_factor.copy()

        factors: List[Array] = []
        norms: List[float] = []
        for block_factor, jac in zip(field, jacobians):
            scaled = np.einsum("ij,njp->nip", block_factor, jac)
            packed = scaled.reshape(-1, jac.shape[2]) / np.sqrt(xm.shape[0])
            factors.append(packed)
            norms.append(float(np.linalg.norm(packed)))

        reached = sum(norm > 1e-14 for norm in norms)
        gap = float(np.mean([np.linalg.norm(value - true_factor) for value in field]))
        self._last_diag = {
            "transport_steps": float(steps),
            "transport_graph_diameter": float(self.graph_diameter),
            "transport_reached_blocks": float(reached),
            "transport_all_blocks_reached": float(reached == self.model.n_layers),
            "transport_factor_gap": gap,
            "transport_min_factor_norm": float(min(norms)),
            "transport_max_factor_norm": float(max(norms)),
        }
        return factors, norms

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        factors, _ = self._transported_factors()
        forces = self._forces(gradients)
        directions = [
            woodbury_inverse_action(factor, self.mass, force)
            for factor, force in zip(factors, forces)
        ]
        self._pending = None
        return [p - self.lr * d for p, d in zip(parameters, directions)]


# Short alias retained for external probe scripts.
PropagatingPullback = OutputFactorTransportPullback
