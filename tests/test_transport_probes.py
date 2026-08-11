import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metric_wave.model import MLP
from metric_wave.riemannian_metric import (
    OutputFactorTransportPullback,
    layer_output_jacobians,
    woodbury_inverse_action,
)
from metric_wave.transported_gauge import TransportedGaugeMetricWave


class TransportProbeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1201)
        self.x = self.rng.normal(size=(20, 5))
        self.y = self.rng.integers(0, 3, size=20)
        self.model = MLP((5, 12, 11, 10, 9, 8, 3), seed=1202)
        _, self.gradients = self.model.loss_and_gradients(self.x, self.y)

    def test_layer_output_jacobian_shapes(self):
        jacobians = layer_output_jacobians(self.model, self.x[:4])
        self.assertEqual(len(jacobians), self.model.n_layers)
        for jacobian, block in zip(jacobians, self.model.parameter_blocks()):
            self.assertEqual(jacobian.shape, (4, 3, block.size))

    def test_woodbury_matches_dense_inverse(self):
        factor = self.rng.normal(size=(7, 13))
        vector = self.rng.normal(size=13)
        mass = 0.7
        actual = woodbury_inverse_action(factor, mass, vector)
        expected = np.linalg.solve(mass * np.eye(13) + factor.T @ factor, vector)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def _factor_diagnostics(self, steps, require=False):
        optimizer = OutputFactorTransportPullback(
            self.model,
            metric_steps=steps,
            wave_speed=1.0,
            relax_rate=0.6,
            require_full_reach=require,
            metric_batch=6,
            seed=1203,
        )
        optimizer.set_metric_batch(self.x, self.y)
        optimizer._transported_factors()
        return optimizer.diagnostics()

    def test_three_hops_are_explicitly_partial(self):
        diagnostics = self._factor_diagnostics(3)
        self.assertEqual(diagnostics["transport_graph_diameter"], 5.0)
        self.assertEqual(diagnostics["transport_reached_blocks"], 4.0)
        self.assertEqual(diagnostics["transport_all_blocks_reached"], 0.0)
        self.assertEqual(diagnostics["transport_min_factor_norm"], 0.0)

    def test_graph_diameter_reaches_every_block(self):
        diagnostics = self._factor_diagnostics(None, require=True)
        self.assertEqual(diagnostics["transport_steps"], 5.0)
        self.assertEqual(diagnostics["transport_reached_blocks"], 6.0)
        self.assertEqual(diagnostics["transport_all_blocks_reached"], 1.0)
        self.assertGreater(diagnostics["transport_min_factor_norm"], 0.0)

    def test_full_reach_guard_rejects_too_few_hops(self):
        optimizer = OutputFactorTransportPullback(
            self.model,
            metric_steps=3,
            wave_speed=1.0,
            require_full_reach=True,
            metric_batch=6,
        )
        optimizer.set_metric_batch(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "cannot reach all blocks"):
            optimizer.step(self.model.parameter_blocks(), self.gradients)

    def test_transported_gauge_has_explicit_nontrivial_edge_maps(self):
        optimizer = TransportedGaugeMetricWave(
            rank=3,
            source_mode="output",
            wave_speed=1.5,
            gauge_batch=8,
            seed=1204,
        )
        optimizer.set_gauge_batch(self.model, self.x)
        updated = optimizer.step(self.model.parameter_blocks(), self.gradients)
        diagnostics = optimizer.diagnostics()
        self.assertTrue(all(np.isfinite(block).all() for block in updated))
        self.assertEqual(diagnostics["gauge_explicit_transport"], 1.0)
        self.assertGreaterEqual(diagnostics["gauge_transport_residual"], 0.0)
        self.assertLess(diagnostics["gauge_frame_orthogonality_error"], 1e-6)
        self.assertEqual(len(optimizer._edge_maps), self.model.n_layers - 1)
        for rotation in optimizer._edge_maps:
            np.testing.assert_allclose(
                rotation.T @ rotation,
                np.eye(rotation.shape[0]),
                atol=1e-10,
                rtol=0.0,
            )

        # A second batch exercises state transport into the changed gauge.
        _, gradients = self.model.loss_and_gradients(self.x[::-1], self.y[::-1])
        optimizer.set_gauge_batch(self.model, self.x[::-1])
        optimizer.step(self.model.parameter_blocks(), gradients)
        self.assertTrue(np.isfinite(optimizer.diagnostics()["gauge_coordinate_change_residual"]))


if __name__ == "__main__":
    unittest.main()

