"""Metric wave with explicit orthogonal transport between block gauges.

Every block frame is anchored to one common probe-output space.  Adjacent
coordinate systems are related by a measured Procrustes map, and neighbouring
metric tensors are transported as ``R H R.T`` before the graph difference is
taken. The diagnostic is evaluated in a shared probe-output space rather than
by comparing ``Q Q.T`` matrices, which are identities for row-orthonormal
frames.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .model import Array, MLP
from .optimizers import MetricWave, MetricWaveState
from .riemannian_metric import layer_output_jacobians


def _symmetric_inverse_sqrt(matrix: Array, floor: float = 1e-10) -> Array:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _procrustes_map(target_basis: Array, source_basis: Array) -> Tuple[Array, float]:
    """Return ``R`` minimizing ``||target_basis R - source_basis||_F``.

    Thus ``R`` maps source coordinates into target coordinates.
    Both bases live in the same probe-output space and have shape ``(r, q)``.
    """
    cross = target_basis.T @ source_basis
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_t
    residual = np.linalg.norm(target_basis @ rotation - source_basis)
    residual /= np.linalg.norm(source_basis) + 1e-15
    return rotation, float(residual)


class TransportedGaugeMetricWave(MetricWave):
    """MetricWave using common-space frames and explicit edge transports."""

    def __init__(self, *args, gauge_batch: int = 16, **kwargs):
        kwargs["frame_mode"] = "fixed"
        super().__init__(*args, **kwargs)
        if gauge_batch < 1:
            raise ValueError("gauge_batch must be positive")
        self.gauge_batch = int(gauge_batch)
        self._gauge_source: Optional[Tuple[MLP, Array]] = None
        self._responses: Optional[List[Array]] = None
        self._edge_maps: List[Array] = []
        self._gauge_diag = {}

    def set_gauge_batch(self, model: MLP, x: Array) -> None:
        self._gauge_source = (model, x)

    def _common_space_frames(self, model: MLP, x: Array, q: int):
        count = min(self.gauge_batch, x.shape[0])
        jacobians = layer_output_jacobians(model, x[:count])
        flat = [jac.reshape(-1, jac.shape[2]) for jac in jacobians]
        common_dim = flat[0].shape[0]
        covariance = np.zeros((common_dim, common_dim))
        for jac in flat:
            scale = max(1.0, float(np.linalg.norm(jac, ord="fro") ** 2))
            covariance += (jac @ jac.T) / scale
        values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        common = vectors[:, np.argsort(values)[-q:]]

        frames: List[Array] = []
        responses: List[Array] = []
        orthogonality = []
        for jac in flat:
            anchored = common.T @ jac
            frame = _symmetric_inverse_sqrt(anchored @ anchored.T) @ anchored
            frames.append(frame)
            responses.append(jac @ frame.T)
            orthogonality.append(float(np.linalg.norm(frame @ frame.T - np.eye(q))))
        return frames, responses, max(orthogonality)

    def _install_gauge(self, gradients: Sequence[Array]) -> None:
        if self._gauge_source is None:
            raise RuntimeError("call set_gauge_batch(model, x) before step()")
        assert self.states is not None
        model, x = self._gauge_source
        q = self.states[0].projector.shape[0]
        frames, responses, orthogonality = self._common_space_frames(model, x, q)

        coordinate_residuals = []
        if self._responses is not None:
            for state, new_response, old_response in zip(
                self.states, responses, self._responses
            ):
                change, residual = _procrustes_map(new_response, old_response)
                state.metric = change @ state.metric @ change.T
                state.velocity = change @ state.velocity @ change.T
                coordinate_residuals.append(residual)

        for index, (state, frame) in enumerate(zip(self.states, frames)):
            if state.projector.shape != frame.shape:
                self.states[index] = MetricWaveState(
                    np.zeros((q, q)), np.zeros((q, q)), frame, frame.copy()
                )
            else:
                state.projector = frame
                state.anchor = frame.copy()

        self._edge_maps = []
        edge_residuals = []
        for current, downstream in zip(responses[:-1], responses[1:]):
            transport, residual = _procrustes_map(current, downstream)
            self._edge_maps.append(transport)
            edge_residuals.append(residual)

        captures = []
        for state, gradient in zip(self.states, gradients):
            projected = state.projector @ gradient
            captures.append(float(projected @ projected) / float(gradient @ gradient + 1e-15))

        self._responses = [value.copy() for value in responses]
        self._gauge_diag = {
            "gauge_capture": float(np.mean(captures)),
            "gauge_transport_residual": float(np.mean(edge_residuals)),
            "gauge_transport_residual_max": float(max(edge_residuals)),
            "gauge_coordinate_change_residual": float(
                np.mean(coordinate_residuals) if coordinate_residuals else 0.0
            ),
            "gauge_frame_orthogonality_error": float(orthogonality),
            "gauge_explicit_transport": 1.0,
        }

    def _metric_laplacian(self, values: Sequence[Array]) -> List[Array]:
        if len(self._edge_maps) != max(0, len(values) - 1):
            raise RuntimeError("edge transports are not initialized")
        result: List[Array] = []
        for k, value in enumerate(values):
            lap = np.zeros_like(value)
            if k > 0:
                from_upstream = self._edge_maps[k - 1].T
                transported = from_upstream @ values[k - 1] @ from_upstream.T
                lap += value - transported
            if k + 1 < len(values):
                from_downstream = self._edge_maps[k]
                transported = from_downstream @ values[k + 1] @ from_downstream.T
                lap += value - transported
            result.append(0.5 * (lap + lap.T))
        return result

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]):
        if self.states is None:
            self._initialize(gradients)
        self._install_gauge(gradients)
        result = super().step(parameters, gradients)
        self._gauge_source = None
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(self._gauge_diag)
        return result


__all__ = ["TransportedGaugeMetricWave"]
