"""Tests for M7 risk control module (pure logic)."""

import unittest

import numpy as np
import pandas as pd

from experiments.tbs.risk_control import (
    COST_MATRIX,
    CLASS_NAMES,
    decide,
    expected_risk,
    compute_metrics,
    search_thresholds,
    risk_coverage_curve,
)


class TestExpectedRisk(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.probs = rng.dirichlet([1, 1, 1, 1, 1], 50).astype(np.float64)

    def test_output_shape(self):
        risks = expected_risk(self.probs)
        self.assertEqual(risks.shape, (50, 5))

    def test_risks_nonnegative(self):
        risks = expected_risk(self.probs)
        self.assertTrue((risks >= 0).all())

    def test_normal_correct_is_zero_cost(self):
        """When true=Normal and action=Normal, cost should be 0."""
        probs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        risks = expected_risk(probs)
        self.assertAlmostEqual(risks[0, 0], 0.0)

    def test_hsil_high_risk_for_normal_action(self):
        """When P(HSIL)=1.0, R(Normal|HSIL) = C(HSIL,Normal) = 6."""
        probs = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        risks = expected_risk(probs)
        self.assertAlmostEqual(risks[0, 0], 6.0, delta=0.5)


class TestDecide(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.probs = rng.dirichlet([1, 1, 1, 1, 1], 200).astype(np.float64)
        self.labels = rng.randint(0, 5, 200).astype(np.int64)
        self.risks = expected_risk(self.probs)

    def test_returns_dataframe(self):
        dec = decide(self.risks, self.labels, 0.5, 0.8)
        self.assertIsInstance(dec, pd.DataFrame)
        self.assertEqual(len(dec), 200)
        self.assertIn("action", dec.columns)
        self.assertIn("prediction_set", dec.columns)
        self.assertIn("is_rejected", dec.columns)

    def test_very_high_threshold_accepts_all(self):
        dec = decide(self.risks, self.labels, tau_normal=100.0, tau_abnormal=100.0)
        self.assertEqual(dec["is_rejected"].sum(), 0)

    def test_very_low_threshold_rejects_all(self):
        dec = decide(self.risks, self.labels, tau_normal=0.0, tau_abnormal=0.0)
        self.assertEqual(dec["is_rejected"].sum(), 200)

    def test_rejected_samples_have_prediction_sets(self):
        dec = decide(self.risks, self.labels, tau_normal=0.3, tau_abnormal=0.5)
        rejected = dec[dec["is_rejected"]]
        if len(rejected) > 0:
            for ps in rejected["prediction_set"]:
                self.assertIn("{", ps)


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 300
        probs = rng.dirichlet([1, 1, 1, 1, 1], n).astype(np.float64)
        self.labels = rng.randint(0, 5, n).astype(np.int64)
        risks = expected_risk(probs)
        self.decisions = decide(risks, self.labels, 0.5, 0.8)

    def test_coverage_between_zero_and_one(self):
        m = compute_metrics(self.decisions)
        self.assertGreaterEqual(m["coverage"], 0.0)
        self.assertLessEqual(m["coverage"], 1.0)

    def test_rejection_rate_plus_coverage_equals_one(self):
        m = compute_metrics(self.decisions)
        self.assertAlmostEqual(m["coverage"] + m["rejection_rate"], 1.0)

    def test_per_class_accept_rate_keys(self):
        m = compute_metrics(self.decisions)
        for name in CLASS_NAMES:
            self.assertIn(name, m["per_class_accept_rate"])

    def test_no_rejection_gives_undercall_raw(self):
        """With tau=inf, no rejection; undercall = raw M0 error rate."""
        probs = np.eye(5)[[3, 0, 4, 0, 3, 1, 2, 0, 4, 3]].astype(np.float64)
        labels = np.array([3, 0, 4, 0, 3, 1, 2, 0, 4, 3], dtype=np.int64)
        risks = expected_risk(probs)
        dec = decide(risks, labels, tau_normal=100.0, tau_abnormal=100.0)
        m = compute_metrics(dec)
        self.assertEqual(m["rejected"], 0)
        self.assertEqual(m["high_grade_undercall_rate"], 0.0)


class TestSearchThresholds(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 500
        probs = rng.dirichlet([1, 1, 1, 1, 1], n).astype(np.float64)
        self.labels = rng.randint(0, 5, n).astype(np.int64)
        self.risks = expected_risk(probs)

    def test_returns_dict_with_keys(self):
        result = search_thresholds(self.risks, self.labels,
                                   tau_normal_range=np.linspace(0.1, 1.0, 5),
                                   tau_abnormal_range=np.linspace(0.1, 2.0, 6))
        self.assertIn("tau_normal", result)
        self.assertIn("tau_abnormal", result)
        self.assertIn("metrics", result)

    def test_tau_values_in_range(self):
        result = search_thresholds(self.risks, self.labels,
                                   tau_normal_range=np.linspace(0.2, 1.5, 4),
                                   tau_abnormal_range=np.linspace(0.2, 2.5, 5))
        self.assertGreaterEqual(result["tau_normal"], 0.2)
        self.assertLessEqual(result["tau_normal"], 1.5)
        self.assertGreaterEqual(result["tau_abnormal"], 0.2)
        self.assertLessEqual(result["tau_abnormal"], 2.5)


class TestRiskCoverageCurve(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1, 1, 1, 1, 1], 100).astype(np.float64)
        self.labels = rng.randint(0, 5, 100).astype(np.int64)
        self.risks = expected_risk(probs)

    def test_curve_monotonic(self):
        """As tau increases, coverage should be non-decreasing."""
        curve = risk_coverage_curve(self.risks, self.labels, n_tau=20)
        coverages = [c["coverage"] for c in curve]
        for i in range(1, len(coverages)):
            self.assertGreaterEqual(coverages[i], coverages[i - 1] - 0.01)


class TestCostMatrix(unittest.TestCase):
    def test_diagonal_is_zero(self):
        for i in range(5):
            self.assertEqual(COST_MATRIX[i, i], 0)

    def test_hsil_to_normal_max_cost(self):
        """HSIL→Normal should be the highest cost."""
        self.assertGreater(COST_MATRIX[4, 0], COST_MATRIX[4, 1])
        self.assertGreater(COST_MATRIX[4, 0], COST_MATRIX[4, 3])

    def test_adjacent_boundaries_lower(self):
        """ASC-US↔LSIL cost should be lower than ASC-US→ASC-H."""
        self.assertLess(COST_MATRIX[1, 2], COST_MATRIX[1, 3])  # ASC-US→LSIL < ASC-US→ASC-H
        self.assertLess(COST_MATRIX[3, 4], COST_MATRIX[3, 0])  # ASC-H→HSIL < ASC-H→Normal
