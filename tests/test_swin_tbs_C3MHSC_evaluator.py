import unittest

import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.C3FHS.protocol import FHS_FEATURE_NAMES
from experiments.C3MHSC.evaluator import group_split_audit
from experiments.C3MHSC.summary import summarize_component_status
from experiments.C3MHSC.evaluator import evaluate_fold


class C3MHSCEvaluatorTests(unittest.TestCase):
    def test_group_split_audit_requires_zero_overlap(self):
        audit = group_split_audit(["a", "b"], ["c", "d"])
        self.assertEqual(audit["overlap_count"], 0)
        self.assertEqual(len(audit["calibration_group_hash"]), 64)
        with self.assertRaisesRegex(ValueError, "calibration/evaluation group overlap"):
            group_split_audit(["a", "b"], ["b", "c"])

    def test_summary_has_generic_failure_route_and_formal_flags_false(self):
        result = summarize_component_status(
            [
                {"component": "C3-A", "development_gate_pass": True},
                {"component": "C3-B", "development_gate_pass": True},
                {"component": "C3-C", "development_gate_pass": False},
            ],
            fold_count=1,
        )
        self.assertEqual(result["route"], "C3MHSC_EXPLORATORY_COMPONENT_FAILURE")
        self.assertFalse(result["formal_eligible"])
        self.assertFalse(result["formal_promotion"])

    def test_single_fold_writes_all_component_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.arange(100, dtype=int) % 5
            probs = np.full((100, 5), 0.02, dtype=float)
            for i, label in enumerate(labels):
                probs[i, label] = 0.92
            eval_labels = np.arange(50, dtype=int) % 5
            eval_probs = np.full((50, 5), 0.02, dtype=float)
            for i, label in enumerate(eval_labels):
                eval_probs[i, label] = 0.92
            calibration_npz = root / "calibration.npz"
            evaluation_npz = root / "evaluation.npz"
            np.savez(calibration_npz, p_final=probs, labels=labels, sample_ids=np.asarray([f"cal-{i}" for i in range(100)]))
            np.savez(
                evaluation_npz,
                p_b0=eval_probs,
                p_final=eval_probs,
                morph_probs=np.tile(np.asarray([[0.4, 0.6]], dtype=float), (50, 1)),
                evidence_probs=np.tile(np.asarray([[0.6, 0.4]], dtype=float), (50, 1)),
                eta=np.zeros(50, dtype=float),
                labels=eval_labels,
                sample_ids=np.asarray([f"eval-{i}" for i in range(50)]),
            )
            hccs = root / "hccs.json"
            hccs.write_text(json.dumps({"screen_threshold_0": 1.0, "screen_threshold_1": 1.0}), encoding="utf-8")
            sentinel = {
                "feature_names": list(FHS_FEATURE_NAMES),
                "mean": [0.0] * 8,
                "scale": [1.0] * 8,
                "coef": [[0.0] * 8],
                "intercept": 0.0,
                "high_undercall_threshold": 0.5,
            }
            sentinel_path = root / "sentinel.json"
            sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")
            fhs = root / "fhs.json"
            fhs.write_text(json.dumps({
                "high_undercall_review_recall": 0.95,
                "silent_high_undercall_rate": 0.01,
                "silent_abnormal_to_normal_rate": 0.001,
                "c1_parameter_drift": 0.0,
                "c1_prediction_changes": 0,
            }), encoding="utf-8")
            result = evaluate_fold(calibration_npz, evaluation_npz, sentinel_path, sentinel_path, hccs, root / "out", 0, fhs_metrics_json=fhs)
            self.assertIn(result["route"], {"C3MHSC_EXPLORATORY_COMPLETE_NO_FORMAL_PROMOTION", "C3MHSC_EXPLORATORY_COMPONENT_FAILURE"})
            self.assertTrue((root / "out" / "fold_summary.json").exists())
            self.assertTrue((root / "out" / "c3c_predictions.csv").exists())
            self.assertTrue((root / "out" / "risk_summary.json").exists())
            self.assertTrue((root / "out" / "risk_outputs.csv").exists())
            self.assertFalse(result["formal_eligible"])


if __name__ == "__main__":
    unittest.main()
