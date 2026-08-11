"""Optimizers, including the weak-field Metric-Wave Backpropagation algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


Array = np.ndarray


class Optimizer:
    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, float]:
        return {}


class SGD(Optimizer):
    def __init__(self, lr: float = 0.03):
        self.lr = float(lr)

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        return [p - self.lr * g for p, g in zip(parameters, gradients)]


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.02, beta: float = 0.9):
        self.lr, self.beta = float(lr), float(beta)
        self.velocity: Optional[List[Array]] = None

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        if self.velocity is None:
            self.velocity = [np.zeros_like(g) for g in gradients]
        self.velocity = [self.beta * v + (1.0 - self.beta) * g for v, g in zip(self.velocity, gradients)]
        return [p - self.lr * v for p, v in zip(parameters, self.velocity)]


class Adam(Optimizer):
    def __init__(self, lr: float = 0.003, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = map(float, (lr, beta1, beta2, eps))
        self.m: Optional[List[Array]] = None
        self.v: Optional[List[Array]] = None
        self.t = 0

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        if self.m is None:
            self.m = [np.zeros_like(g) for g in gradients]
            self.v = [np.zeros_like(g) for g in gradients]
        self.t += 1
        self.m = [self.beta1 * m + (1.0 - self.beta1) * g for m, g in zip(self.m, gradients)]
        self.v = [self.beta2 * v + (1.0 - self.beta2) * (g * g) for v, g in zip(self.v, gradients)]
        result = []
        for p, m, v in zip(parameters, self.m, self.v):
            mhat = m / (1.0 - self.beta1 ** self.t)
            vhat = v / (1.0 - self.beta2 ** self.t)
            result.append(p - self.lr * mhat / (np.sqrt(vhat) + self.eps))
        return result


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.004, beta: float = 0.99, eps: float = 1e-8):
        self.lr, self.beta, self.eps = float(lr), float(beta), float(eps)
        self.v: Optional[List[Array]] = None

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        if self.v is None:
            self.v = [np.zeros_like(g) for g in gradients]
        self.v = [self.beta * v + (1.0 - self.beta) * (g * g) for v, g in zip(self.v, gradients)]
        return [p - self.lr * g / (np.sqrt(v) + self.eps) for p, g, v in zip(parameters, gradients, self.v)]


class CovariantMoment(Optimizer):
    """Low-rank full second-moment metric baseline inspired by covariant GD."""

    def __init__(self, lr: float = 0.02, beta1: float = 0.9, beta2: float = 0.98, rank: int = 6,
                 eps: float = 0.1, seed: int = 0):
        self.lr, self.beta1, self.beta2 = float(lr), float(beta1), float(beta2)
        self.rank, self.eps = int(rank), float(eps)
        self.rng = np.random.default_rng(seed)
        self.projectors: Optional[List[Array]] = None
        self.first: Optional[List[Array]] = None
        self.second: Optional[List[Array]] = None

    def _initialize(self, gradients: Sequence[Array]) -> None:
        q = min(self.rank, min(g.size for g in gradients))
        self.projectors, self.first, self.second = [], [], []
        for grad in gradients:
            raw = self.rng.normal(size=(grad.size, q))
            orthonormal, _ = np.linalg.qr(raw)
            self.projectors.append(orthonormal[:, :q].T)
            self.first.append(np.zeros_like(grad))
            self.second.append(np.zeros((q, q)))

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        if self.projectors is None:
            self._initialize(gradients)
        assert self.projectors is not None and self.first is not None and self.second is not None
        result = []
        for idx, (p, g, qmat) in enumerate(zip(parameters, gradients, self.projectors)):
            self.first[idx] = self.beta1 * self.first[idx] + (1.0 - self.beta1) * g
            sketch = qmat @ g
            self.second[idx] = self.beta2 * self.second[idx] + (1.0 - self.beta2) * np.outer(sketch, sketch)
            eigvals, eigvecs = np.linalg.eigh(self.second[idx] + self.eps * np.eye(sketch.size))
            invsqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            force = self.first[idx]
            direction = force + qmat.T @ ((invsqrt - np.eye(sketch.size)) @ (qmat @ force))
            result.append(p - self.lr * direction)
        return result


@dataclass
class MetricWaveState:
    metric: Array
    velocity: Array
    projector: Array
    anchor: Array


class MetricWave(Optimizer):
    r"""Low-rank tensor Metric-Wave Backpropagation.

    For layer block ``k``, a symmetric metric perturbation ``H_k`` follows

        H'' + 2 gamma H' + omega^2 H + c^2 (L H)_k = kappa S_k,

    where ``S_k`` is the trace-free outer product of a sketched local gradient.
    The induced block metric is

        M_k = I + Q_k^T [exp(rho H_k) - I] Q_k,

    so its inverse is available without constructing a full parameter-space
    matrix.  ``wave_speed=0`` is the non-propagating metric ablation.
    """

    def __init__(
        self,
        lr: float = 0.02,
        beta: float = 0.9,
        rank: int = 6,
        rho: float = 0.35,
        coupling: float = 0.8,
        damping: float = 1.2,
        restoring: float = 1.0,
        wave_speed: float = 0.8,
        dt: float = 0.08,
        substeps: int = 2,
        max_metric_eigenvalue: float = 2.5,
        tensor: bool = True,
        source_mode: str = "output",
        frame_mode: str = "fixed",
        seed: int = 0,
    ):
        self.lr = float(lr)
        self.beta = float(beta)
        self.rank = int(rank)
        self.rho = float(rho)
        self.coupling = float(coupling)
        self.damping = float(damping)
        self.restoring = float(restoring)
        self.wave_speed = float(wave_speed)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.max_metric_eigenvalue = float(max_metric_eigenvalue)
        self.tensor = bool(tensor)
        if source_mode not in {"output", "all"}:
            raise ValueError("source_mode must be 'output' or 'all'")
        self.source_mode = source_mode
        if frame_mode not in {"fixed", "gradient_aligned"}:
            raise ValueError("frame_mode must be 'fixed' or 'gradient_aligned'")
        self.frame_mode = frame_mode
        self.rng = np.random.default_rng(seed)
        self.states: Optional[List[MetricWaveState]] = None
        self.momentum: Optional[List[Array]] = None
        self._last_diag: Dict[str, float] = {}

    def _initialize(self, gradients: Sequence[Array]) -> None:
        q = 1 if not self.tensor else min(self.rank, min(g.size for g in gradients))
        self.states = []
        for grad in gradients:
            raw = self.rng.normal(size=(grad.size, q))
            orthonormal, _ = np.linalg.qr(raw)
            projector = orthonormal[:, :q].T
            self.states.append(MetricWaveState(
                np.zeros((q, q)), np.zeros((q, q)), projector, projector.copy()
            ))
        self.momentum = [np.zeros_like(g) for g in gradients]

    @staticmethod
    def _gradient_aligned_projector(state: MetricWaveState, vector: Array) -> Array:
        """Return a row-orthonormal frame whose first row follows ``vector``."""
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-14:
            return state.projector
        first = vector / norm
        candidates = np.column_stack((first, state.anchor.T))
        orthonormal, _ = np.linalg.qr(candidates)
        return orthonormal[:, : state.projector.shape[0]].T

    @staticmethod
    def _line_laplacian(values: Sequence[Array]) -> List[Array]:
        result: List[Array] = []
        n = len(values)
        for k, value in enumerate(values):
            lap = np.zeros_like(value)
            if k > 0:
                lap += value - values[k - 1]
            if k + 1 < n:
                lap += value - values[k + 1]
            result.append(lap)
        return result

    def _metric_laplacian(self, values: Sequence[Array]) -> List[Array]:
        """Return the graph Laplacian in the metric-field coordinates.

        The base implementation identifies neighbouring sketch coordinates.
        Subclasses with explicit inter-block gauges override this method and
        transport a neighbouring tensor before taking its difference.
        """
        return self._line_laplacian(values)

    def _source(self, projected_gradient: Array) -> Array:
        if projected_gradient.size == 1:
            magnitude = np.tanh(np.log1p(abs(float(projected_gradient[0]))))
            return np.array([[magnitude]], dtype=float)
        norm2 = float(projected_gradient @ projected_gradient)
        outer = np.outer(projected_gradient, projected_gradient) / (norm2 / projected_gradient.size + 1e-12)
        source = outer - np.eye(projected_gradient.size) * np.trace(outer) / projected_gradient.size
        spectral = max(1.0, float(np.max(np.abs(np.linalg.eigvalsh(source)))))
        return source / spectral

    def _clip_symmetric(self, matrix: Array) -> Array:
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, -self.max_metric_eigenvalue, self.max_metric_eigenvalue)
        return (eigvecs * eigvals) @ eigvecs.T

    def _inverse_metric_action(self, state: MetricWaveState, vector: Array) -> Array:
        eigvals, eigvecs = np.linalg.eigh(state.metric)
        exp_inverse = (eigvecs * np.exp(-self.rho * eigvals)) @ eigvecs.T
        projected = state.projector @ vector
        return vector + state.projector.T @ ((exp_inverse - np.eye(exp_inverse.shape[0])) @ projected)

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> List[Array]:
        if self.states is None:
            self._initialize(gradients)
        assert self.states is not None and self.momentum is not None
        self.momentum = [self.beta * m + (1.0 - self.beta) * g for m, g in zip(self.momentum, gradients)]
        if self.frame_mode == "gradient_aligned":
            for state, force in zip(self.states, self.momentum):
                state.projector = self._gradient_aligned_projector(state, force)
        local_sources = [self._source(s.projector @ g) for s, g in zip(self.states, self.momentum)]
        if self.source_mode == "output":
            sources = [np.zeros_like(source) for source in local_sources]
            sources[-1] = local_sources[-1]
        else:
            sources = local_sources

        for _ in range(self.substeps):
            metrics = [s.metric for s in self.states]
            laplacian = self._metric_laplacian(metrics)
            for state, lap, source in zip(self.states, laplacian, sources):
                acceleration = (
                    self.coupling * source
                    - 2.0 * self.damping * state.velocity
                    - (self.restoring ** 2) * state.metric
                    - (self.wave_speed ** 2) * lap
                )
                state.velocity = state.velocity + self.dt * acceleration
                state.metric = self._clip_symmetric(state.metric + self.dt * state.velocity)

        directions = [self._inverse_metric_action(state, grad) for state, grad in zip(self.states, self.momentum)]
        all_eigs = np.concatenate([np.linalg.eigvalsh(s.metric) for s in self.states])
        wave_energy = sum(float(np.sum(s.velocity ** 2)) for s in self.states) / 2.0
        metric_energy = sum(float(np.sum(s.metric ** 2)) for s in self.states) / 2.0
        edge_energy = sum(float(np.sum((self.states[k + 1].metric - self.states[k].metric) ** 2)) for k in range(len(self.states) - 1)) / 2.0
        self._last_diag = {
            "wave_energy": wave_energy,
            "metric_energy": metric_energy,
            "edge_energy": edge_energy,
            "metric_eig_min": float(all_eigs.min()),
            "metric_eig_max": float(all_eigs.max()),
            "metric_condition_bound": float(np.exp(self.rho * (all_eigs.max() - all_eigs.min()))),
        }
        return [p - self.lr * direction for p, direction in zip(parameters, directions)]

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._last_diag)
