import unittest

import numpy as np

from experiments.C3FHS.features import FHS_FEATURE_NAMES, build_fhs_features
from experiments.C3FHS.protocol import C3FHS_CONFIG, c3fhs_root
from experiments.C3FHS.sentinel import crossfit_sentinel, finite_sample_high_threshold
from experiments.C3FHS.sentinel import fit_sentinel
from experiments.C3FHS.policy import apply_fhs_policy
from experiments.C3HCCS.calibration import fit_hccs_calibration


class C3FHSTests(unittest.TestCase):
    def _bundle(self, n=60):
        p_b0 = np.tile(np.asarray([.05, .20, .15, .30, .30]), (n, 1))
        p_final = np.tile(np.asarray([.05, .55, .15, .15, .10]), (n, 1))
        morph = np.tile(np.asarray([.35, .65]), (n, 1))
        evidence = np.tile(np.asarray([.40, .60]), (n, 1))
        eta = np.linspace(-.2, .2, n)
        labels = np.asarray([0, 1, 2, 3, 4] * (n // 5))
        return {
            "p_b0": p_b0,
            "p_final": p_final,
            "morph_probs": morph,
            "evidence_probs": evidence,
            "eta": eta,
            "labels": labels,
            "sample_ids": np.asarray([f"s{i}" for i in range(n)]),
        }

    def test_exact_features_and_nonhigh_candidate_mask(self):
        result = build_fhs_features(self._bundle())
        self.assertEqual(result["features"].shape, (60, 8))
        self.assertEqual(result["feature_names"], FHS_FEATURE_NAMES)
        self.assertEqual(result["candidate_mask"].dtype, bool)
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertTrue(np.all(result["candidate_mask"] == ~np.isin(result["prediction_c1"], [3, 4])))

    def test_crossfit_sentinel_and_threshold_are_deterministic(self):
        data = self._bundle(100)
        features = build_fhs_features(data)
        target = np.isin(data["labels"], [3, 4]).astype(int)
        candidate = features["candidate_mask"]
        fold_ids = np.arange(100) % 5
        result = crossfit_sentinel(features["features"][candidate], target[candidate], fold_ids[candidate])
        self.assertEqual(result["oof_scores"].shape[0], int(candidate.sum()))
        high_scores = result["oof_scores"][target[candidate] == 1]
        threshold = finite_sample_high_threshold(high_scores, 0.95)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(result["feature_names"], FHS_FEATURE_NAMES)

    def test_protocol_isolated(self):
        self.assertEqual(C3FHS_CONFIG["feature_count"], 8)
        self.assertEqual(C3FHS_CONFIG["calibration_target"], 0.95)
        self.assertTrue(str(c3fhs_root("/tmp/project")).endswith("C3_factor_discordance_high_sentinel_v1"))

    def test_sentinel_only_adds_review_and_preserves_c1_top1(self):
        data = self._bundle(100)
        features = build_fhs_features(data)
        candidate = features["candidate_mask"]
        head = fit_sentinel(features["features"][candidate], features["target_high_undercall"][candidate])
        calibration = fit_hccs_calibration(data["p_final"], data["labels"])
        rows, _ = apply_fhs_policy(data["p_final"], data["labels"], features, head, 0.0, calibration)
        np.testing.assert_array_equal(rows["top1"], features["prediction_c1"])
        self.assertEqual(len(rows["review_flag"]), len(data["labels"]))


if __name__ == "__main__":
    unittest.main()
