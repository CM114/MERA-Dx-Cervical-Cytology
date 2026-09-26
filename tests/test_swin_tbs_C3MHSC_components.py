import unittest

import numpy as np

from experiments.C3MHSC.components import audit_c3a, audit_c3b


class C3MHSCComponentTests(unittest.TestCase):
    def test_c3a_uses_frozen_hccs_screening_only(self):
        probabilities = np.asarray(
            [[0.95, 0.0125, 0.0125, 0.0125, 0.0125], [0.01, 0.90, 0.04, 0.03, 0.02]],
            dtype=float,
        )
        labels = np.asarray([0, 1], dtype=int)
        calibration = {
            "screen_threshold_0": 1.0,
            "screen_threshold_1": 1.0,
        }
        result = audit_c3a(probabilities, labels, calibration, source_artifact={"route": "C3-HCCS-v1"})
        self.assertEqual(result["component"], "C3-A")
        self.assertEqual(result["screen_coverage"], 1.0)
        self.assertTrue(result["development_gate_pass"])
        self.assertFalse(result["formal_eligible"])

    def test_c3b_matching_locked_sentinel_passes_and_mismatch_aborts(self):
        sentinel = {
            "feature_names": ["a"],
            "mean": [0.0],
            "scale": [1.0],
            "coef": [[1.0]],
            "intercept": 0.0,
            "high_undercall_threshold": 0.5,
        }
        metrics = {
            "high_undercall_review_recall": 0.95,
            "silent_high_undercall_rate": 0.02,
            "silent_abnormal_to_normal_rate": 0.001,
            "c1_parameter_drift": 0.0,
            "c1_prediction_changes": 0,
        }
        result = audit_c3b(sentinel, sentinel, metrics, source_artifact={"route": "C3-FHS-v1"})
        self.assertTrue(result["development_gate_pass"])
        self.assertEqual(result["component"], "C3-B")
        mismatch = dict(sentinel)
        mismatch["high_undercall_threshold"] = 0.6
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_PROVENANCE_MISMATCH"):
            audit_c3b(mismatch, sentinel, metrics, source_artifact={"route": "C3-FHS-v1"})


if __name__ == "__main__":
    unittest.main()
