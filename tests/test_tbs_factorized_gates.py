import json
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_tbs_factorized_gates import _read_metrics, evaluate_gate


def metrics(**overrides):
    values = {
        "macro_f1": 0.80,
        "low_grade_pair_macro_f1": 0.80,
        "high_grade_pair_macro_f1": 0.80,
        "screen_sensitivity": 0.99,
        "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
        "morph_auroc": 0.80,
        "evidence_auroc": 0.80,
    }
    values.update(overrides)
    return values


class FactorizedGateTests(unittest.TestCase):
    def test_c0_requires_global_gain_and_boundary_protection(self):
        result = evaluate_gate("c0", metrics(), metrics(macro_f1=0.806))
        self.assertTrue(result["passed"])
        failed = evaluate_gate("c0", metrics(), metrics(macro_f1=0.806, high_grade_pair_macro_f1=0.79))
        self.assertFalse(failed["passed"])

    def test_c1_requires_semantic_learning_and_a_boundary_gain(self):
        result = evaluate_gate(
            "c1",
            metrics(),
            metrics(
                macro_f1=0.799,
                low_grade_pair_macro_f1=0.81,
                morph_auroc=0.70,
                evidence_auroc=0.71,
            ),
        )
        self.assertTrue(result["passed"])
        failed = evaluate_gate(
            "c1", metrics(), metrics(macro_f1=0.799, morph_auroc=0.70, evidence_auroc=0.71)
        )
        self.assertFalse(failed["passed"])

    def test_c2_requires_boundary_gain_without_high_grade_risk_regression(self):
        result = evaluate_gate(
            "c2",
            metrics(),
            metrics(
                macro_f1=0.81,
                low_grade_pair_macro_f1=0.82,
                high_grade_pair_macro_f1=0.81,
            ),
        )
        self.assertTrue(result["passed"])
        failed = evaluate_gate(
            "c2",
            metrics(),
            metrics(
                macro_f1=0.81,
                low_grade_pair_macro_f1=0.82,
                asc_h_hsil_to_normal_lowgrade_rate=0.06,
            ),
        )
        self.assertFalse(failed["passed"])

    def test_gate_output_is_json_serializable(self):
        result = evaluate_gate("c0", metrics(), metrics(macro_f1=0.806))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["passed"])

    def test_metrics_can_be_recomputed_from_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev_predictions.csv"
            rows = [
                "image_path,true_label,prob_Normal,prob_ASC-US,prob_LSIL,prob_ASC-H,prob_HSIL",
                "0.jpg,0,1,0,0,0,0",
                "1.jpg,1,0,1,0,0,0",
                "2.jpg,2,0,0,1,0,0",
                "3.jpg,3,0,0,0,1,0",
                "4.jpg,4,0,0,0,0,1",
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            loaded = _read_metrics(path)
            self.assertEqual(loaded["macro_f1"], 1.0)
            self.assertEqual(loaded["low_grade_pair_macro_f1"], 1.0)
            self.assertEqual(loaded["high_grade_pair_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
