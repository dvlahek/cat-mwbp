"""Layer-local auxiliary-classifier baseline for the reference MLP.

The hidden layers receive the class label but no downstream activation error.
Each hidden representation is connected to a fixed random classifier.  The
resulting auxiliary cross-entropy produces a gradient using only the layer's
input, preactivation, fixed local head, and label.  The network output retains
its ordinary softmax cross-entropy gradient.

This control is intentionally named descriptively rather than presented as an
implementation of a particular published algorithm.  It tests label-local
training without a backward traversal, weight transport, or output-error
broadcast through the trainable network.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .local_adjoint import gradients_from_local_adjoint, output_boundary
from .model import Array, MLP


class LocalAuxiliaryHeads:
    """Train hidden blocks through fixed random layer-local classifiers."""

    def __init__(self, seed: int = 0, head_scale: float = 1.0):
        if head_scale <= 0.0:
            raise ValueError("head_scale must be positive")
        self.rng = np.random.default_rng(seed)
        self.head_scale = float(head_scale)
        self.heads: Optional[List[Array]] = None

    def _initialize(self, model: MLP) -> None:
        classes = model.widths[-1]
        self.heads = []
        for width in model.widths[1:-1]:
            matrix = self.rng.normal(size=(width, classes))
            matrix *= self.head_scale / np.sqrt(max(1, width))
            self.heads.append(matrix)

    def gradients(self, model: MLP, x: Array, y: Array) -> Tuple[float, List[Array], Dict[str, float]]:
        logits, cache = model.forward(x)
        loss, output_delta = output_boundary(model, logits, y)
        if self.heads is None:
            self._initialize(model)
        assert self.heads is not None

        n = x.shape[0]
        deltas: List[Array] = []
        local_losses: List[float] = []
        for layer, head in zip(cache[:-1], self.heads):
            hidden = np.tanh(layer.preactivation)
            local_logits = hidden @ head
            probabilities = model._softmax(local_logits)
            local_losses.append(float(-np.log(probabilities[np.arange(n), y] + 1e-12).mean()))
            error = probabilities.copy()
            error[np.arange(n), y] -= 1.0
            error /= n
            derivative = 1.0 - hidden ** 2
            deltas.append((error @ head.T) * derivative)
        deltas.append(output_delta)

        diagnostics = {
            "local_head_loss": float(np.mean(local_losses)) if local_losses else loss,
            "local_head_norm": float(np.sqrt(sum(np.sum(head ** 2) for head in self.heads))),
            "local_head_adjoint_norm": float(np.sqrt(sum(np.sum(delta ** 2) for delta in deltas))),
        }
        return loss, gradients_from_local_adjoint(cache, deltas), diagnostics
