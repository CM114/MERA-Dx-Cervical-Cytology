import unittest

import numpy as np

from experiments.C3MHSC.matched_budget import compare_fold, score_arrays


class C3MatchedBudgetTests(unittest.TestCase):
    def test_scores_are_finite_and_have_expected_direction(self):
        p = np.asarray([[0.9, 0.05, 0.02, 0.02, 0.01], [0.2] * 5], dtype=float)
        scores = score_arrays(p)
        self.assertTrue(all(np.isfinite(value).all() for value in scores.values()))
        self.assertGreater(scores["direct_p_high"][0], scores["direct_p_high"][1])

    def test_all_methods_use_exact_sentinel_budget(self):
        p = np.asarray([
            [0.70, 0.20, 0.05, 0.03, 0.02],
            [0.65, 0.20, 0.05, 0.05, 0.05],
            [0.10, 0.20, 0.25, 0.25, 0.20],
            [0.05, 0.10, 0.15, 0.40, 0.30],
        ], dtype=float)
        y = np.asarray([3, 4, 1, 0])
        ids = np.asarray(["d", "c", "b", "a"])
        sentinel = np.asarray([True, False, False, True])
        rows = compare_fold(probabilities=p, labels=y, sample_ids=ids, sentinel_review_mask=sentinel)
        self.assertTrue(all(row["review_count"] == 2 for row in rows))
        self.assertTrue(all(row["c1_prediction_changes"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
