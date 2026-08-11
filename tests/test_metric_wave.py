import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metric_wave.model import MLP
from metric_wave.local_adjoint import (
    gradient_comparison,
    local_adjoint_gradients,
    output_boundary,
    relax_activation_adjoint,
    relax_local_adjoint,
)
from metric_wave.optimizers import MetricWave
from metric_wave.training import BlockDelayBuffer


class MetricWaveTests(unittest.TestCase):
    def test_metric_action_is_positive(self):
        optimizer = MetricWave(rank=3, seed=7)
        gradients = [np.arange(1.0, 13.0), np.arange(1.0, 9.0)]
        optimizer._initialize(gradients)
        for state in optimizer.states:
            random = np.random.default_rng(1).normal(size=state.metric.shape)
            state.metric = 0.4 * (random + random.T)
        for state, gradient in zip(optimizer.states, gradients):
            direction = optimizer._inverse_metric_action(state, gradient)
            self.assertGreater(float(gradient @ direction), 0.0)

    def test_zero_coupling_recovers_momentum(self):
        model = MLP((2, 4, 2), seed=1)
        x = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1])
        _, gradients = model.loss_and_gradients(x, y)
        parameters = model.parameter_blocks()
        optimizer = MetricWave(lr=0.1, beta=0.0, coupling=0.0, seed=3)
        updated = optimizer.step(parameters, gradients)
        for old, new, grad in zip(parameters, updated, gradients):
            np.testing.assert_allclose(new, old - 0.1 * grad, atol=1e-10)

    def test_one_step_is_finite_and_symmetric(self):
        model = MLP((3, 5, 2), seed=2)
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(12, 3)), rng.integers(0, 2, size=12)
        _, gradients = model.loss_and_gradients(x, y)
        optimizer = MetricWave(rank=3, seed=5)
        updated = optimizer.step(model.parameter_blocks(), gradients)
        self.assertTrue(all(np.isfinite(block).all() for block in updated))
        for state in optimizer.states:
            np.testing.assert_allclose(state.metric, state.metric.T, atol=1e-12)

    def test_block_delay_buffer_exact_timing(self):
        buffer = BlockDelayBuffer((0, 2))
        first = buffer.push((np.array([1.0]), np.array([10.0])))
        second = buffer.push((np.array([2.0]), np.array([20.0])))
        third = buffer.push((np.array([3.0]), np.array([30.0])))
        self.assertEqual(first[0][0], 1.0)
        self.assertEqual(second[0][0], 2.0)
        self.assertEqual(first[1][0], 0.0)
        self.assertEqual(second[1][0], 0.0)
        self.assertEqual(third[1][0], 10.0)

    def test_gradient_aligned_frame_is_orthonormal(self):
        gradients = [np.arange(1.0, 13.0), np.arange(1.0, 9.0)]
        optimizer = MetricWave(rank=3, frame_mode="gradient_aligned", seed=9)
        optimizer._initialize(gradients)
        state = optimizer.states[0]
        frame = optimizer._gradient_aligned_projector(state, gradients[0])
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        self.assertGreater(abs(float(frame[0] @ gradients[0])) / np.linalg.norm(gradients[0]), 1.0 - 1e-12)

    def test_local_adjoint_converges_to_backprop_gradient(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(24, 5))
        y = rng.integers(0, 3, size=24)
        model = MLP((5, 8, 7, 6, 3), seed=13)
        _, exact = model.loss_and_gradients(x, y)
        _, approximate, _ = local_adjoint_gradients(model, x, y, steps=100)
        comparison = gradient_comparison(exact, approximate)
        self.assertLess(comparison["gradient_relative_error"], 2e-4)
        self.assertGreater(comparison["gradient_cosine"], 1.0 - 1e-8)

    def test_local_adjoint_has_finite_hop_support(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(10, 4))
        y = rng.integers(0, 2, size=10)
        model = MLP((4, 6, 5, 4, 2), seed=15)
        logits, cache = model.forward(x)
        _, boundary = output_boundary(model, logits, y)
        _, _, trace = relax_local_adjoint(model, cache, boundary, steps=2, return_trace=True)
        self.assertEqual(float(np.linalg.norm(trace[1][0])), 0.0)
        self.assertEqual(float(np.linalg.norm(trace[1][1])), 0.0)
        self.assertGreater(float(np.linalg.norm(trace[1][2])), 0.0)
        self.assertEqual(float(np.linalg.norm(trace[2][0])), 0.0)
        self.assertGreater(float(np.linalg.norm(trace[2][1])), 0.0)

    def test_activation_relaxation_has_the_same_finite_hop_schedule(self):
        rng = np.random.default_rng(141)
        x = rng.normal(size=(10, 4))
        y = rng.integers(0, 2, size=10)
        model = MLP((4, 6, 5, 4, 2), seed=142)
        logits, cache = model.forward(x)
        _, boundary = output_boundary(model, logits, y)
        _, _, trace = relax_activation_adjoint(
            model, cache, boundary, steps=2, rate=0.16, return_trace=True
        )
        self.assertEqual(float(np.linalg.norm(trace[1][0])), 0.0)
        self.assertEqual(float(np.linalg.norm(trace[1][1])), 0.0)
        self.assertGreater(float(np.linalg.norm(trace[1][2])), 0.0)
        self.assertEqual(float(np.linalg.norm(trace[2][0])), 0.0)
        self.assertGreater(float(np.linalg.norm(trace[2][1])), 0.0)

    def test_local_adjoint_does_not_call_global_backward(self):
        rng = np.random.default_rng(16)
        x = rng.normal(size=(8, 3))
        y = rng.integers(0, 2, size=8)
        model = MLP((3, 5, 4, 2), seed=17)

        def forbidden(*args, **kwargs):
            raise AssertionError("global hidden-layer backward was called")

        model.loss_and_gradients = forbidden
        loss, gradients, _ = local_adjoint_gradients(model, x, y, steps=8)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(gradients), model.n_layers)


if __name__ == "__main__":
    unittest.main()
