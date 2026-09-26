import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_case_predictions import aggregate_case_predictions


class AggregateCasePredictionsTests(unittest.TestCase):
    def test_aggregates_patch_probabilities_by_case_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "dev_predictions.csv"
            manifest = root / "dev.csv"
            rows = [
                {
                    "image_path": "/data/case_a_1.jpg",
                    "true_label": "0",
                    "prob_Normal": "0.8",
                    "prob_ASC-US": "0.1",
                    "prob_LSIL": "0.05",
                    "prob_ASC-H": "0.03",
                    "prob_HSIL": "0.02",
                },
                {
                    "image_path": "/data/case_a_2.jpg",
                    "true_label": "0",
                    "prob_Normal": "0.6",
                    "prob_ASC-US": "0.2",
                    "prob_LSIL": "0.1",
                    "prob_ASC-H": "0.05",
                    "prob_HSIL": "0.05",
                },
                {
                    "image_path": "/data/case_b_1.jpg",
                    "true_label": "1",
                    "prob_Normal": "0.1",
                    "prob_ASC-US": "0.7",
                    "prob_LSIL": "0.1",
                    "prob_ASC-H": "0.05",
                    "prob_HSIL": "0.05",
                },
                {
                    "image_path": "/data/case_b_2.jpg",
                    "true_label": "1",
                    "prob_Normal": "0.2",
                    "prob_ASC-US": "0.5",
                    "prob_LSIL": "0.15",
                    "prob_ASC-H": "0.1",
                    "prob_HSIL": "0.05",
                },
            ]
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("image_path", "case_id", "diagnosis_label"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "image_path": "/data/case_a_1.jpg",
                            "case_id": "case_a",
                            "diagnosis_label": "0",
                        },
                        {
                            "image_path": "/data/case_a_2.jpg",
                            "case_id": "case_a",
                            "diagnosis_label": "0",
                        },
                        {
                            "image_path": "/data/case_b_1.jpg",
                            "case_id": "case_b",
                            "diagnosis_label": "1",
                        },
                        {
                            "image_path": "/data/case_b_2.jpg",
                            "case_id": "case_b",
                            "diagnosis_label": "1",
                        },
                    ]
                )

            out_dir = root / "case_level"
            summary = aggregate_case_predictions(predictions, manifest, out_dir)

            self.assertEqual(summary["patch_count"], 4)
            self.assertEqual(summary["case_count"], 2)
            self.assertTrue(summary["image_manifest_join_passed"])
            self.assertTrue((out_dir / "case_predictions.csv").is_file())
            self.assertTrue((out_dir / "case_metrics.json").is_file())

            case_rows = list(
                csv.DictReader(
                    (out_dir / "case_predictions.csv").open(
                        newline="", encoding="utf-8"
                    )
                )
            )
            self.assertEqual([row["case_id"] for row in case_rows], ["case_a", "case_b"])
            self.assertEqual(case_rows[0]["patch_count"], "2")
            self.assertEqual(case_rows[0]["pred_label"], "0")
            self.assertEqual(case_rows[1]["pred_label"], "1")

            metrics = json.loads(
                (out_dir / "case_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
