import unittest

import numpy as np

from experiments.C2Semantic.geometry_audit import bootstrap_gap_ci, factor_geometry_gaps


class C2SemanticGeometryTests(unittest.TestCase):
    def test_factor_geometry_gaps_follow_controlled_pairs(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.9, 0.1],  # ASC-US / LSIL: morphology positive
            [0.0, 1.0], [0.1, 0.9],  # ASC-H / HSIL: morphology positive
        ], dtype=float)
        labels = np.asarray([1, 2, 3, 4], dtype=int)
        result = factor_geometry_gaps(embeddings, labels)
        self.assertGreater(result["positive_mean"], result["negative_mean"])
        self.assertGreater(result["gap"], 0.0)

    def test_bootstrap_ci_is_deterministic(self):
        values = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=float)
        first = bootstrap_gap_ci(values, np.asarray([0.1, 0.1, 0.2, 0.2]), reps=100, seed=42)
        second = bootstrap_gap_ci(values, np.asarray([0.1, 0.1, 0.2, 0.2]), reps=100, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(first["gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
