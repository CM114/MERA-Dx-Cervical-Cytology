import unittest

import numpy as np

from experiments.C3MHSC.routing import route_risk_outputs, summarize_risk_outputs


def _calibration():
    # A deliberately permissive frozen screening artifact: Normal rows can be
    # flagged when the abnormal mass is non-negligible.
    return {
        "screen_threshold_0": 0.60,
        "screen_threshold_1": 0.60,
    }


class C3MHSCRoutingTests(unittest.TestCase):
    def test_high_undercall_flag_does_not_change_c1_top1_or_diagnosis_set(self):
        probabilities = np.array([[0.05, 0.55, 0.25, 0.10, 0.05]], dtype=float)
        pair_rows = {
            "final_low_sets": [[1]],
            "final_high_sets": [[3]],
        }
        result = route_risk_outputs(
            probabilities,
            pair_rows,
            sentinel_scores=np.array([0.97]),
            sentinel_threshold=0.90,
            hccs_calibration=_calibration(),
        )

        self.assertEqual(result["top1"].tolist(), [1])
        self.assertEqual(result["prediction_sets"], [[1]])
        self.assertEqual(result["risk_reason"].tolist(), ["HIGH_GRADE_UNDERCALL_RISK"])
        self.assertEqual(result["review_priority"].tolist(), ["high"])
        self.assertTrue(bool(result["high_undercall_flag"][0]))
        self.assertTrue(bool(result["abstain_flag"][0]))

    def test_normal_screening_conflict_is_high_priority_without_changing_top1(self):
        probabilities = np.array([[0.55, 0.25, 0.10, 0.05, 0.05]], dtype=float)
        pair_rows = {
            "final_low_sets": [[1]],
            "final_high_sets": [[3]],
        }
        result = route_risk_outputs(
            probabilities,
            pair_rows,
            sentinel_scores=np.array([0.01]),
            sentinel_threshold=0.90,
            hccs_calibration=_calibration(),
        )

        self.assertEqual(result["top1"].tolist(), [0])
        self.assertEqual(result["prediction_sets"], [[0]])
        self.assertEqual(result["risk_reason"].tolist(), ["ABNORMAL_SCREENING_CONFLICT"])
        self.assertEqual(result["review_priority"].tolist(), ["high"])
        self.assertTrue(bool(result["screening_flag"][0]))
        self.assertTrue(bool(result["abstain_flag"][0]))

    def test_pair_ambiguity_is_medium_priority_and_keeps_set_explicit(self):
        probabilities = np.array([[0.02, 0.49, 0.48, 0.005, 0.005]], dtype=float)
        pair_rows = {
            "final_low_sets": [[1, 2]],
            "final_high_sets": [[3]],
        }
        result = route_risk_outputs(
            probabilities,
            pair_rows,
            sentinel_scores=np.array([0.01]),
            sentinel_threshold=0.90,
            hccs_calibration={"screen_threshold_0": 0.01, "screen_threshold_1": 0.01},
        )

        self.assertEqual(result["prediction_sets"], [[1, 2]])
        self.assertEqual(result["risk_reason"].tolist(), ["ADJACENT_DIAGNOSIS_AMBIGUITY"])
        self.assertEqual(result["review_priority"].tolist(), ["medium"])
        self.assertTrue(bool(result["pair_ambiguity_flag"][0]))
        self.assertFalse(bool(result["abstain_flag"][0]))

    def test_summary_reports_structured_review_rates(self):
        rows = {
            "top1": np.array([0, 1]),
            "prediction_sets": [[0], [1, 2]],
            "p_abnormal": np.array([0.2, 0.9]),
            "p_high": np.array([0.1, 0.2]),
            "u_screen": np.array([0.3, 0.2]),
            "u_severity": np.array([0.1, 0.4]),
            "u_pair": np.array([0.0, 0.9]),
            "screening_flag": np.array([True, False]),
            "high_undercall_flag": np.array([False, True]),
            "pair_ambiguity_flag": np.array([False, True]),
            "review_priority": np.array(["high", "high"]),
            "abstain_flag": np.array([True, True]),
            "risk_reason": np.array(["ABNORMAL_SCREENING_CONFLICT", "HIGH_GRADE_UNDERCALL_RISK"]),
            "sentinel_score": np.array([0.1, 0.9]),
        }
        summary = summarize_risk_outputs(np.array([0, 4]), rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["high_undercall_count"], 1)
        self.assertEqual(summary["review_count"], 2)
        self.assertFalse(summary["formal_eligible"])
        self.assertFalse(summary["formal_promotion"])


if __name__ == "__main__":
    unittest.main()
