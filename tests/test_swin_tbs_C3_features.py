import unittest

import numpy as np

from experiments.C3.features import build_c3_features, build_factor_targets, js_divergence


class C3FeatureTests(unittest.TestCase):
    def test_js_is_symmetric_and_bounded(self):
        a = np.asarray([[0.8, 0.2], [0.5, 0.5]])
        b = np.asarray([[0.2, 0.8], [0.5, 0.5]])
        self.assertTrue(np.allclose(js_divergence(a, b), js_divergence(b, a)))
        self.assertTrue(np.all(js_divergence(a, b) >= 0.0))
        self.assertTrue(np.all(js_divergence(a, b) <= 1.0))

    def test_factor_features_and_targets_are_finite_and_ordered(self):
        c1 = np.asarray([
            [0.05, 0.70, 0.10, 0.10, 0.05],
            [0.02, 0.05, 0.05, 0.80, 0.08],
            [0.90, 0.03, 0.02, 0.03, 0.02],
            [0.05, 0.10, 0.75, 0.05, 0.05],
        ], dtype=float)
        semantic = {
            "labels": np.asarray([1, 3, 0, 2]),
            "sample_ids": np.asarray(["a", "b", "c", "d"]),
            "q_morph": np.tile([[0.7, 0.3]], (4, 1)),
            "q_evidence": np.tile([[0.6, 0.4]], (4, 1)),
            "q_proto": np.tile([[0.5, 0.2, 0.2, 0.1]], (4, 1)),
            "morph_assignments": np.tile([[[0.7, 0.3], [0.4, 0.6]]], (4, 1, 1)),
            "evidence_assignments": np.tile([[[0.6, 0.4], [0.5, 0.5]]], (4, 1, 1)),
            "morph_nearest_distance": np.tile([[0.2, 0.8]], (4, 1)),
            "evidence_nearest_distance": np.tile([[0.3, 0.7]], (4, 1)),
        }
        result = build_c3_features(c1, semantic)
        self.assertEqual(result["morph_features"].shape, (4, 6))
        self.assertEqual(result["evidence_features"].shape, (4, 6))
        self.assertEqual(result["overall_features"].shape[1], 14)
        self.assertTrue(np.isfinite(result["overall_features"]).all())
        targets = build_factor_targets(np.argmax(c1, axis=1), semantic["labels"])
        self.assertEqual(targets["t_m"].tolist(), [0, 0, 0, 0])
        self.assertEqual(targets["t_e"].tolist(), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
