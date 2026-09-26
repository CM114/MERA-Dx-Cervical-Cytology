"""Tests for temperature scaling module (pure logic, no PyTorch)."""

import math
import unittest

import numpy as np

from experiments.tbs.temperature_scaling import (
    apply_temperature,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    high_group_brier,
    nll,
    per_class_ece,
    reliability_data,
    stratified_ece_high_grade,
    stratified_ece_screening,
)


class TestFitTemperature(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.logits = rng.randn(200, 5).astype(np.float64)
        self.labels = rng.randint(0, 5, 200).astype(np.int64)

    def test_returns_positive_temperature(self):
        result = fit_temperature(self.logits, self.labels)
        self.assertGreater(result["temperature"], 0.0)
        self.assertLess(result["temperature"], 50.0)
        self.assertTrue(result["converged"])

    def test_nll_improves_or_stays_same(self):
        result = fit_temperature(self.logits, self.labels)
        self.assertLessEqual(result["nll_after"], result["nll_before"] + 1e-6)

    def test_temperature_reasonable_range(self):
        """Temperature should be positive and finite."""
        logits = np.random.RandomState(0).randn(500, 3).astype(np.float64)
        labels = (logits + np.random.RandomState(1).randn(500, 3) * 0.1).argmax(axis=1)
        result = fit_temperature(logits, labels)
        self.assertGreater(result["temperature"], 0.0)
        self.assertLess(result["temperature"], 50.0)
        self.assertTrue(math.isfinite(result["temperature"]))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fit_temperature(self.logits, self.labels[:10])

    def test_overconfident_logits(self):
        """Overconfident logits should yield T > 1."""
        logits = np.zeros((100, 3), dtype=np.float64)
        logits[:, 0] = 10.0  # very confident class 0
        labels = np.array([0] * 70 + [1] * 15 + [2] * 15, dtype=np.int64)
        result = fit_temperature(logits, labels)
        self.assertGreater(result["temperature"], 1.0)


class TestApplyTemperature(unittest.TestCase):
    def test_argmax_invariant(self):
        logits = np.random.RandomState(0).randn(50, 5).astype(np.float64)
        probs = apply_temperature(logits, 2.5)
        self.assertTrue(np.array_equal(logits.argmax(axis=1), probs.argmax(axis=1)))

    def test_rows_sum_to_one(self):
        logits = np.random.RandomState(0).randn(50, 5).astype(np.float64)
        probs = apply_temperature(logits, 1.3)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


class TestECE(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.labels = rng.randint(0, 3, 300).astype(np.int64)
        # Perfectly calibrated: probs = one-hot
        one_hot = np.eye(3)[self.labels]
        self.perfect_probs = one_hot + rng.randn(*one_hot.shape) * 0.001
        self.perfect_probs = np.clip(self.perfect_probs, 0, 1)
        self.perfect_probs /= self.perfect_probs.sum(axis=1, keepdims=True)

    def test_ece_perfect_is_low(self):
        result = expected_calibration_error(self.perfect_probs, self.labels, n_bins=10)
        self.assertLess(result["ece"], 0.1)

    def test_ece_overconfident_is_high(self):
        """Overconfident probabilities → high ECE."""
        overconfident = np.clip(self.perfect_probs * 1.5, 0, 1)
        overconfident /= overconfident.sum(axis=1, keepdims=True)
        result = expected_calibration_error(
            np.clip(overconfident, 1e-6, 1.0), self.labels, n_bins=10
        )
        self.assertGreater(result["ece"], 0.0)

    def test_equal_mass_bins(self):
        result = expected_calibration_error(self.perfect_probs, self.labels, n_bins=10, bin_strategy="equal_mass")
        self.assertIn("ece", result)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            expected_calibration_error(self.perfect_probs, self.labels, bin_strategy="invalid")


class TestStratifiedECE(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 200
        self.labels = rng.randint(0, 5, n).astype(np.int64)
        one_hot = np.eye(5)[self.labels].astype(np.float64)
        self.probs = one_hot + rng.randn(*one_hot.shape) * 0.01
        self.probs = np.clip(self.probs, 1e-6, 1.0)
        self.probs /= self.probs.sum(axis=1, keepdims=True)

    def test_screening_ece(self):
        result = stratified_ece_screening(self.probs, self.labels)
        self.assertIn("ece", result)

    def test_high_grade_ece(self):
        result = stratified_ece_high_grade(self.probs, self.labels)
        self.assertIn("ece", result)

    def test_per_class_ece(self):
        result = per_class_ece(self.probs, self.labels)
        self.assertEqual(len(result), 5)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 100
        self.labels = rng.randint(0, 3, n).astype(np.int64)
        self.probs = np.eye(3)[self.labels].astype(np.float64)
        self.probs += rng.randn(*self.probs.shape) * 0.01
        self.probs = np.clip(self.probs, 1e-6, 1.0)
        self.probs /= self.probs.sum(axis=1, keepdims=True)

    def test_brier_zero_for_perfect(self):
        bs = brier_score(np.eye(3)[self.labels].astype(np.float64), self.labels)
        self.assertAlmostEqual(bs, 0.0, delta=0.02)

    def test_nll_finite(self):
        self.assertTrue(math.isfinite(nll(self.probs, self.labels)))

    def test_high_group_brier(self):
        probs = np.eye(5)[[0, 3, 4, 0, 3]]  # 1 Normal, 2 ASC-H, 2 HSIL
        labels = np.array([0, 3, 4, 0, 3], dtype=np.int64)
        bs = high_group_brier(probs, labels)
        self.assertAlmostEqual(bs, 0.0, delta=0.01)


class TestBootstrap(unittest.TestCase):
    def test_returns_ci(self):
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1, 1, 1, 1, 1], 100).astype(np.float64)
        labels = rng.randint(0, 5, 100).astype(np.int64)
        ci = bootstrap_ci(probs, labels, lambda p, l: float(np.mean(p.argmax(axis=1) == l)))
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)
        self.assertLess(ci["ci_lower"], ci["ci_upper"])


class TestReliabilityData(unittest.TestCase):
    def test_returns_lists(self):
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1, 1, 1], 200).astype(np.float64)
        labels = rng.randint(0, 3, 200).astype(np.int64)
        conf, acc = reliability_data(probs, labels, n_bins=10)
        self.assertEqual(len(conf), 10)
        self.assertEqual(len(acc), 10)
