import unittest

import numpy as np

from experiments.C3MHSC.pair import (
    apply_pair_sets,
    evaluate_pair_sets,
    finite_pair_quantile,
    fit_pair_calibration,
)


def _toy_arrays(n=100):
    labels = np.arange(n, dtype=int) % 5
    probs = np.full((n, 5), 0.02, dtype=float)
    for i, y in enumerate(labels):
        probs[i, y] = 0.92
    return probs, labels, np.asarray([f"group-{i}" for i in range(n)], dtype=object)


class C3MHSCPairTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_conformal_rank_without_truncation(self):
        q, rank = finite_pair_quantile(np.asarray([0.95, 0.95, 0.95]), 0.95)
        self.assertEqual(rank, 4)
        self.assertEqual(q, 1.0)

        q, rank = finite_pair_quantile(np.arange(20, dtype=float) / 20.0, 0.95)
        self.assertEqual(rank, 20)
        self.assertAlmostEqual(q, 0.95)

    def test_fit_contains_class_specific_quantiles_and_group_hash(self):
        probs, labels, ids = _toy_arrays()
        calibration = fit_pair_calibration(probs, labels, ids, nominal=0.95)
        for name in ("quantile_ascus", "quantile_lsil", "quantile_asch", "quantile_hsil"):
            self.assertIn(name, calibration)
        self.assertEqual(calibration["calibration_rows"], 100)
        self.assertEqual(len(calibration["calibration_group_hash"]), 64)
        self.assertTrue(calibration["dev_accessed"])

    def test_empty_raw_sets_are_counted_and_have_nonempty_final_fallback(self):
        probs, labels, ids = _toy_arrays()
        calibration = fit_pair_calibration(probs, labels, ids, nominal=0.95)
        for name in ("quantile_ascus", "quantile_lsil", "quantile_asch", "quantile_hsil"):
            calibration[name] = 0.0
        result = apply_pair_sets(probs[:10], calibration)
        self.assertGreater(result["n_empty_raw_low"], 0)
        self.assertGreater(result["n_empty_raw_high"], 0)
        self.assertTrue(all(result["final_low_sets"]))
        self.assertTrue(all(result["final_high_sets"]))
        self.assertEqual(len(result["final_low_sets"]), 10)

    def test_pair_evaluation_uses_pairwise_and_class_conditional_coverage(self):
        probs, labels, ids = _toy_arrays()
        calibration = fit_pair_calibration(probs, labels, ids, nominal=0.95)
        sets = apply_pair_sets(probs, calibration)
        metrics = evaluate_pair_sets(labels, sets)
        self.assertIn("low_pair_coverage", metrics)
        self.assertIn("high_pair_coverage", metrics)
        self.assertEqual(metrics["n_low"], 40)
        self.assertEqual(metrics["n_high"], 40)
        self.assertFalse(metrics["formal_eligible"])
        self.assertFalse(metrics["formal_promotion"])


if __name__ == "__main__":
    unittest.main()
