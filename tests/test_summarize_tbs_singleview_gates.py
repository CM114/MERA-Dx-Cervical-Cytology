import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.summarize_tbs_singleview_gates import (
    read_metrics,
    validate_prediction_alignment,
)


class SingleViewGateCliTests(unittest.TestCase):
    def test_json_metrics_are_read_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            expected = {"macro_f1": 0.78, "low_grade_pair_macro_f1": 0.76}
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(read_metrics(path), expected)

    def test_only_json_or_prediction_csv_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.txt"
            path.write_text("not metrics", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON or prediction CSV"):
                read_metrics(path)

    def test_prediction_comparison_requires_identical_samples_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            pd.DataFrame(
                {"image_path": ["a.jpg", "b.jpg"], "true_label": [0, 1]}
            ).to_csv(baseline, index=False)
            pd.DataFrame(
                {"image_path": ["b.jpg", "a.jpg"], "true_label": [1, 0]}
            ).to_csv(candidate, index=False)
            validate_prediction_alignment(baseline, candidate)
            pd.DataFrame(
                {"image_path": ["a.jpg", "b.jpg"], "true_label": [0, 2]}
            ).to_csv(candidate, index=False)
            with self.assertRaisesRegex(ValueError, "labels"):
                validate_prediction_alignment(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
