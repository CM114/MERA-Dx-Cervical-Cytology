import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.singleview_protocol import (
    evaluate_singleview_gate,
    make_stagewise_train_splits,
    metrics_from_prediction_csv,
)


def gate_metrics(**overrides):
    values = {
        "macro_f1": 0.78,
        "low_grade_pair_macro_f1": 0.77,
        "high_grade_pair_macro_f1": 0.81,
        "screen_sensitivity": 1.0,
        "asc_h_hsil_to_normal_lowgrade_rate": 0.06,
        "morph_auroc": 0.75,
        "evidence_auroc": 0.75,
    }
    values.update(overrides)
    return values


class StagewiseSplitTests(unittest.TestCase):
    def test_stagewise_splits_are_deterministic_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in range(5):
                for index in range(40):
                    rows.append(
                        {
                            "image_path": f"/train/{label}_{index}.jpg",
                            "diagnosis_label": label,
                            "content_sha256": f"{label:02d}{index:04d}",
                        }
                    )
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)

            first = make_stagewise_train_splits(source, root / "a", 0.1, 42)
            second = make_stagewise_train_splits(source, root / "b", 0.1, 42)
            third = make_stagewise_train_splits(source, root / "c", 0.1, 7)

            for left, right in zip(first, second):
                self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertNotEqual(first[1].read_bytes(), third[1].read_bytes())
            fit, select_s0, select_s1 = (pd.read_csv(path) for path in first)
            sets = [set(frame["image_path"]) for frame in (fit, select_s0, select_s1)]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            self.assertEqual(set(select_s0["diagnosis_label"]), set(range(5)))
            self.assertEqual(set(select_s1["diagnosis_label"]), set(range(5)))
            self.assertEqual(sum(len(frame) for frame in (fit, select_s0, select_s1)), 200)

    def test_duplicate_content_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.csv"
            pd.DataFrame(
                [
                    {"image_path": "/a.jpg", "diagnosis_label": 0, "content_sha256": "same"},
                    {"image_path": "/b.jpg", "diagnosis_label": 0, "content_sha256": "same"},
                ]
            ).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate content"):
                make_stagewise_train_splits(source, root / "splits", 0.1, 42)


class SingleViewGateTests(unittest.TestCase):
    def test_prediction_csv_metrics_include_locked_pair_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            rows = []
            for label in range(5):
                for index in range(4):
                    probabilities = [0.01] * 5
                    probabilities[label] = 0.96
                    row = {"true_label": label, "image_path": f"/{label}_{index}.jpg"}
                    row.update(
                        {
                            f"prob_{name}": probabilities[class_index]
                            for class_index, name in enumerate(
                                ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
                            )
                        }
                    )
                    rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            metrics = metrics_from_prediction_csv(path)
            self.assertEqual(metrics["low_grade_pair_macro_f1"], 1.0)
            self.assertEqual(metrics["high_grade_pair_macro_f1"], 1.0)

    def test_s0_must_remain_close_to_historical_m0(self):
        baseline = gate_metrics(macro_f1=0.7845)
        self.assertTrue(
            evaluate_singleview_gate("s0", baseline, gate_metrics(macro_f1=0.77))["passed"]
        )
        self.assertFalse(
            evaluate_singleview_gate("s0", baseline, gate_metrics(macro_f1=0.74))["passed"]
        )

    def test_s1_requires_semantic_and_boundary_gain_without_risk_regression(self):
        baseline = gate_metrics()
        candidate = gate_metrics(
            macro_f1=0.781,
            low_grade_pair_macro_f1=0.78,
            morph_auroc=0.70,
            evidence_auroc=0.72,
        )
        self.assertTrue(evaluate_singleview_gate("s1", baseline, candidate)["passed"])
        candidate["asc_h_hsil_to_normal_lowgrade_rate"] = 0.07
        self.assertFalse(evaluate_singleview_gate("s1", baseline, candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
