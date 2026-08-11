import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from metric_wave.data import load_dataset_three_way
from metric_wave.direct_feedback import DirectFeedbackAlignment, NeighborFeedbackAlignment
from metric_wave.local_heads import LocalAuxiliaryHeads
from metric_wave.local_adjoint import (
    activation_relaxation_gradients,
    gradient_comparison,
    overdamped_adjoint_gradients,
    overdamped_relaxation_rate,
)
from metric_wave.model import MLP
from metric_wave.optimizers import Momentum
from metric_wave.training import evaluate
from metric_wave.validated_training import train_with_validation
from run_journal_suite import calibrate_energy_matched_local, make_gradient_function, profile_config, task_plan, tost


class JournalSuiteTests(unittest.TestCase):
    def test_npl_profile_run_counts_are_fixed(self):
        self.assertEqual(len(task_plan(profile_config("quick"))), 18)
        self.assertEqual(len(task_plan(profile_config("standard"))), 434)
        self.assertEqual(len(task_plan(profile_config("full"))), 861)

    def test_ar_rate_is_the_predeclared_overdamped_cat_rate(self):
        config = profile_config("full")
        derived = overdamped_relaxation_rate(
            config.adjoint_dt, config.adjoint_damping, config.adjoint_frequency
        )
        self.assertAlmostEqual(derived, 0.16, places=15)
        self.assertAlmostEqual(config.ar_relaxation_rate, derived, places=15)

    def test_ar_and_overdamped_cat_are_numerically_identical(self):
        rng = np.random.default_rng(401)
        x = rng.normal(size=(18, 5))
        y = rng.integers(0, 3, size=18)
        model = MLP((5, 9, 8, 7, 3), seed=402)
        _, ar_gradients, _ = activation_relaxation_gradients(
            model, x, y, steps=64, rate=0.16
        )
        _, overdamped_gradients, _ = overdamped_adjoint_gradients(
            model, x, y, steps=64, dt=0.04, damping=8.0, frequency=8.0
        )
        comparison = gradient_comparison(ar_gradients, overdamped_gradients)
        self.assertLess(comparison["gradient_relative_error"], 1e-14)
        self.assertGreater(comparison["gradient_cosine"], 1.0 - 1e-12)
        for ar_block, overdamped_block in zip(ar_gradients, overdamped_gradients):
            np.testing.assert_allclose(ar_block, overdamped_block, atol=1e-14, rtol=0.0)

    def test_ar_uses_no_global_hidden_backward_and_keeps_output_gradient(self):
        rng = np.random.default_rng(403)
        x = rng.normal(size=(17, 4))
        y = rng.integers(0, 2, size=17)
        reference = MLP((4, 8, 7, 2), seed=404)
        _, exact = reference.loss_and_gradients(x, y)
        model = MLP((4, 8, 7, 2), seed=404)
        model.loss_and_gradients = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global backward called")
        )
        loss, approximate, diagnostics = activation_relaxation_gradients(
            model, x, y, steps=16, rate=0.16
        )
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(exact[-1], approximate[-1], atol=1e-12)
        self.assertEqual(diagnostics["credit_dynamics_order"], 1.0)
        self.assertEqual(diagnostics["neighbor_jacobian_actions"], 32.0)

    def test_tost_detects_inside_and_outside_equivalence_margin(self):
        inside = tost(np.array([-0.0001, 0.0, 0.0001, 0.0, 0.00005]), 0.001)
        outside = tost(np.array([0.0020, 0.0021, 0.0019, 0.0022, 0.0018]), 0.001)
        self.assertTrue(inside[-1])
        self.assertFalse(outside[-1])

    def test_three_way_split_is_disjoint_sized_and_train_scaled(self):
        x_train, x_validation, x_test, y_train, y_validation, y_test = load_dataset_three_way("moons", 41)
        self.assertEqual(len(x_train) + len(x_validation) + len(x_test), 1200)
        self.assertEqual((len(x_train), len(x_validation), len(x_test)), (720, 180, 300))
        np.testing.assert_allclose(x_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(x_train.std(axis=0), 1.0, atol=1e-12)
        self.assertEqual(set(np.unique(y_train)), set(np.unique(y_validation)))
        self.assertEqual(set(np.unique(y_train)), set(np.unique(y_test)))

    def test_large_offline_dataset_is_deterministic(self):
        first = load_dataset_three_way("synthetic_large", 7)
        second = load_dataset_three_way("synthetic_large", 7)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)
        self.assertEqual(first[0].shape[1], 100)
        self.assertEqual(len(np.unique(first[3])), 10)

    def test_dfa_does_not_call_global_backward(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 4))
        y = rng.integers(0, 3, size=12)
        model = MLP((4, 8, 7, 3), seed=43)
        model.loss_and_gradients = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global backward called")
        )
        loss, gradients, diagnostics = DirectFeedbackAlignment(seed=44).gradients(model, x, y)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(gradients), model.n_layers)
        self.assertGreater(diagnostics["dfa_feedback_norm"], 0.0)

    def test_dfa_output_gradient_matches_backprop_output_block(self):
        rng = np.random.default_rng(45)
        x = rng.normal(size=(16, 5))
        y = rng.integers(0, 2, size=16)
        model = MLP((5, 9, 7, 2), seed=46)
        _, exact = model.loss_and_gradients(x, y)
        _, approximate, _ = DirectFeedbackAlignment(seed=47).gradients(model, x, y)
        np.testing.assert_allclose(exact[-1], approximate[-1], atol=1e-12)

    def test_neighbor_fa_is_finite_and_keeps_exact_output_block(self):
        rng = np.random.default_rng(471)
        x = rng.normal(size=(14, 6))
        y = rng.integers(0, 3, size=14)
        model = MLP((6, 9, 8, 3), seed=472)
        _, exact = model.loss_and_gradients(x, y)
        _, approximate, diagnostics = NeighborFeedbackAlignment(seed=473).gradients(model, x, y)
        np.testing.assert_allclose(exact[-1], approximate[-1], atol=1e-12)
        self.assertTrue(all(np.isfinite(block).all() for block in approximate))
        self.assertGreater(diagnostics["fa_feedback_norm"], 0.0)

    def test_local_heads_use_no_global_backward_and_keep_exact_output_block(self):
        rng = np.random.default_rng(474)
        x = rng.normal(size=(15, 6))
        y = rng.integers(0, 3, size=15)
        reference = MLP((6, 10, 8, 3), seed=475)
        _, exact = reference.loss_and_gradients(x, y)
        model = MLP((6, 10, 8, 3), seed=475)
        model.loss_and_gradients = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global backward called")
        )
        loss, approximate, diagnostics = LocalAuxiliaryHeads(seed=476).gradients(model, x, y)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(block).all() for block in approximate))
        np.testing.assert_allclose(exact[-1], approximate[-1], atol=1e-12)
        self.assertGreater(diagnostics["local_head_norm"], 0.0)

    def test_validation_training_restores_best_checkpoint(self):
        x_train, x_validation, _, y_train, y_validation, _ = load_dataset_three_way("moons", 48)
        model = MLP((2, 8, 2), seed=49)

        def bp(current, x, y):
            loss, gradients = current.loss_and_gradients(x, y)
            return loss, gradients, {}

        result = train_with_validation(
            model,
            Momentum(lr=0.03, beta=0.0),
            bp,
            x_train,
            y_train,
            x_validation,
            y_validation,
            max_epochs=4,
            patience=1,
            min_delta=10.0,
            batch_size=64,
            seed=50,
        )
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.stopped_epoch, 2)
        self.assertAlmostEqual(evaluate(model, x_validation, y_validation)["loss"], result.best_validation_loss, places=12)

    def test_energy_match_calibration_uses_finite_positive_coupling(self):
        config = profile_config("quick")
        x_train, _, _, y_train, _, _ = load_dataset_three_way("moons", 51)
        model = MLP((2, 10, 8, 2), seed=52)
        gradient_function = make_gradient_function("bp", 0, 53, config)
        coupling, target, achieved = calibrate_energy_matched_local(
            model, gradient_function, x_train, y_train, 54, config
        )
        self.assertGreater(coupling, 0.0)
        self.assertLessEqual(coupling, 3.0)
        self.assertGreater(target, 0.0)
        self.assertLess(abs(achieved - target) / target, 0.08)


if __name__ == "__main__":
    unittest.main()
