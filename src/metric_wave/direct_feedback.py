"""Direct Feedback Alignment baseline for the feed-forward reference model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .local_adjoint import gradients_from_local_adjoint, output_boundary
from .model import Array, MLP


class DirectFeedbackAlignment:
    """Broadcast the output error through fixed random feedback matrices.

    DFA is included as a direct local-learning control. It avoids a hidden
    transpose-Jacobian traversal, but unlike CAT it broadcasts the output error
    to every hidden layer and has no exact-backpropagation equilibrium.
    """

    def __init__(self, seed: int = 0, feedback_scale: float = 1.0):
        if feedback_scale <= 0.0:
            raise ValueError("feedback_scale must be positive")
        self.rng = np.random.default_rng(seed)
        self.feedback_scale = float(feedback_scale)
        self.feedback: Optional[List[Array]] = None

    def _initialize(self, model: MLP) -> None:
        output_width = model.widths[-1]
        self.feedback = []
        for hidden_width in model.widths[1:-1]:
            raw = self.rng.normal(size=(output_width, hidden_width))
            raw /= np.sqrt(max(1, output_width))
            self.feedback.append(self.feedback_scale * raw)

    def gradients(self, model: MLP, x: Array, y: Array) -> Tuple[float, List[Array], Dict[str, float]]:
        logits, cache = model.forward(x)
        loss, boundary = output_boundary(model, logits, y)
        if self.feedback is None:
            self._initialize(model)
        assert self.feedback is not None
        deltas: List[Array] = []
        for layer, matrix in zip(cache[:-1], self.feedback):
            derivative = 1.0 - np.tanh(layer.preactivation) ** 2
            deltas.append((boundary @ matrix) * derivative)
        deltas.append(boundary)
        diagnostics = {
            "dfa_feedback_norm": float(np.sqrt(sum(np.sum(matrix ** 2) for matrix in self.feedback))),
            "dfa_adjoint_norm": float(np.sqrt(sum(np.sum(delta ** 2) for delta in deltas))),
        }
        return loss, gradients_from_local_adjoint(cache, deltas), diagnostics


class NeighborFeedbackAlignment:
    """Propagate errors backward through fixed neighbor-local random matrices.

    This is the conventional Feedback Alignment control. It retains a serial
    hidden error traversal, but removes weight transport and therefore helps
    separate CAT's finite-speed relaxation from random-feedback learning.
    """

    def __init__(self, seed: int = 0, feedback_scale: float = 1.0):
        if feedback_scale <= 0.0:
            raise ValueError("feedback_scale must be positive")
        self.rng = np.random.default_rng(seed)
        self.feedback_scale = float(feedback_scale)
        self.feedback: Optional[List[Array]] = None

    def _initialize(self, model: MLP) -> None:
        self.feedback = []
        for downstream_width, upstream_width in zip(model.widths[2:], model.widths[1:-1]):
            matrix = self.rng.normal(size=(downstream_width, upstream_width))
            matrix *= self.feedback_scale / np.sqrt(max(1, downstream_width))
            self.feedback.append(matrix)

    def gradients(self, model: MLP, x: Array, y: Array) -> Tuple[float, List[Array], Dict[str, float]]:
        logits, cache = model.forward(x)
        loss, boundary = output_boundary(model, logits, y)
        if self.feedback is None:
            self._initialize(model)
        assert self.feedback is not None
        deltas: List[Array] = [np.zeros_like(layer.preactivation) for layer in cache]
        deltas[-1] = boundary
        for k in range(model.n_layers - 2, -1, -1):
            derivative = 1.0 - np.tanh(cache[k].preactivation) ** 2
            deltas[k] = (deltas[k + 1] @ self.feedback[k]) * derivative
        diagnostics = {
            "fa_feedback_norm": float(np.sqrt(sum(np.sum(matrix ** 2) for matrix in self.feedback))),
            "fa_adjoint_norm": float(np.sqrt(sum(np.sum(delta ** 2) for delta in deltas))),
        }
        return loss, gradients_from_local_adjoint(cache, deltas), diagnostics
