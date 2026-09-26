import unittest

import numpy as np

from experiments.C3AGR.features import AGR_FEATURE_NAMES, build_agr_features, r1_anchor


class C3AGRFeatureTests(unittest.TestCase):
    def test_uses_exact_state_geometry_residual_features(self):
        c1 = np.asarray([
            [0.02, 0.40, 0.30, 0.20, 0.08],
            [0.02, 0.10, 0.08, 0.55, 0.25],
            [0.80, 0.05, 0.05, 0.05, 0.05],
        ])
        semantic = {
            "q_morph": np.asarray([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]),
            "q_evidence": np.asarray([[0.7, 0.3], [0.8, 0.2], [0.5, 0.5]]),
            "sample_ids": np.asarray(["a", "b", "c"]),
        }
        result = build_agr_features(c1, semantic)
        self.assertEqual(result["features"].shape, (3, 8))
        self.assertEqual(result["feature_names"], AGR_FEATURE_NAMES)
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertTrue(np.allclose(result["features"][2], 0.0))
        self.assertAlmostEqual(result["features"][0, 0], np.log(0.8 / 0.2), places=6)
        self.assertAlmostEqual(result["features"][0, 1], np.log(0.7 / 0.3), places=6)

    def test_r1_anchor_matches_locked_entropy_margin_definition(self):
        c1 = np.asarray([[0.02, 0.40, 0.30, 0.20, 0.08]])
        r1, s1, entropy, uncertainty = r1_anchor(c1)
        self.assertEqual(r1.shape, (1,))
        self.assertTrue(np.isfinite(s1).all())
        self.assertAlmostEqual(float(r1[0]), float(0.5 * (entropy[0] + uncertainty[0])), places=8)


if __name__ == "__main__":
    unittest.main()
