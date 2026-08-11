"""A small NumPy multilayer perceptron used by the reproducibility experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class LayerCache:
    input: Array
    preactivation: Array


class MLP:
    """Fully connected tanh network with softmax cross-entropy output.

    Each affine layer is one parameter block.  This block structure is also the
    computational graph on which the metric wave propagates.
    """

    def __init__(self, widths: Sequence[int], seed: int = 0):
        if len(widths) < 2:
            raise ValueError("widths must include input and output dimensions")
        self.widths = tuple(int(x) for x in widths)
        rng = np.random.default_rng(seed)
        self.weights: List[Array] = []
        self.biases: List[Array] = []
        for fan_in, fan_out in zip(self.widths[:-1], self.widths[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=float))

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def forward(self, x: Array) -> Tuple[Array, List[LayerCache]]:
        h = np.asarray(x, dtype=float)
        cache: List[LayerCache] = []
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = h @ weight + bias
            cache.append(LayerCache(h, z))
            h = z if idx == self.n_layers - 1 else np.tanh(z)
        return h, cache

    @staticmethod
    def _softmax(logits: Array) -> Array:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def loss_and_gradients(self, x: Array, y: Array) -> Tuple[float, List[Array]]:
        logits, cache = self.forward(x)
        probs = self._softmax(logits)
        n = x.shape[0]
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        delta = probs
        delta[np.arange(n), y] -= 1.0
        delta /= n
        grad_w: List[Array] = [np.empty_like(w) for w in self.weights]
        grad_b: List[Array] = [np.empty_like(b) for b in self.biases]
        for idx in range(self.n_layers - 1, -1, -1):
            grad_w[idx] = cache[idx].input.T @ delta
            grad_b[idx] = delta.sum(axis=0)
            if idx > 0:
                delta = (delta @ self.weights[idx].T) * (1.0 - np.tanh(cache[idx - 1].preactivation) ** 2)
        return float(loss), [self._pack_pair(w, b) for w, b in zip(grad_w, grad_b)]

    def loss(self, x: Array, y: Array) -> float:
        """Cross-entropy without constructing hidden-layer gradients."""
        logits, _ = self.forward(x)
        probs = self._softmax(logits)
        return float(-np.log(probs[np.arange(x.shape[0]), y] + 1e-12).mean())

    def predict(self, x: Array) -> Array:
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)

    def parameter_blocks(self) -> List[Array]:
        return [self._pack_pair(w, b) for w, b in zip(self.weights, self.biases)]

    def set_parameter_blocks(self, blocks: Sequence[Array]) -> None:
        if len(blocks) != self.n_layers:
            raise ValueError("one parameter block is required for each affine layer")
        for idx, block in enumerate(blocks):
            expected = self.weights[idx].size + self.biases[idx].size
            if block.size != expected:
                raise ValueError(f"block {idx} has size {block.size}; expected {expected}")
            cut = self.weights[idx].size
            self.weights[idx][...] = block[:cut].reshape(self.weights[idx].shape)
            self.biases[idx][...] = block[cut:]

    @staticmethod
    def _pack_pair(weight: Array, bias: Array) -> Array:
        return np.concatenate((weight.ravel(), bias.ravel()))
