import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.xudata_gain_common import (
    compute_locked_candidate_metrics,
    make_internal_train_split,
    reject_forbidden_data_path,
    write_safety_artifacts,
)


class XudataGainCommonTests(unittest.TestCase):
    def test_rejects_calibration_and_test_components(self):
        with self.assertRaises(ValueError):
            reject_forbidden_data_path(Path("/run/calibration/dev.csv"))
        with self.assertRaises(ValueError):
            reject_forbidden_data_path(Path("/run/test/dev.csv"))

    def test_internal_split_is_deterministic_and_class_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in range(5):
                for index in range(10):
                    rows.append(
                        {
                            "image_path": f"/xudata/train/{label}_{index}.jpg",
                            "diagnosis_label": label,
                            "content_sha256": f"{label:02d}{index:02d}",
                        }
                    )
            train_csv = root / "train_xudata.csv"
            pd.DataFrame(rows).to_csv(train_csv, index=False)

            fit_a, select_a = make_internal_train_split(train_csv, root / "a")
            fit_b, select_b = make_internal_train_split(train_csv, root / "b")

            self.assertEqual(fit_a.read_bytes(), fit_b.read_bytes())
            self.assertEqual(select_a.read_bytes(), select_b.read_bytes())
            self.assertEqual(
                set(pd.read_csv(select_a)["diagnosis_label"]),
                {0, 1, 2, 3, 4},
            )

    def test_locked_metrics_include_pair_macro_f1(self):
        metrics = compute_locked_candidate_metrics(
            np.array([0, 1, 2, 3, 4]),
            np.eye(5),
        )
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["abnormal_macro_f1"], 1.0)
        self.assertEqual(metrics["low_grade_pair_macro_f1"], 1.0)
        self.assertEqual(metrics["high_grade_pair_macro_f1"], 1.0)

    def test_safety_artifacts_mark_sealed_splits_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            write_safety_artifacts(
                out_dir,
                {"seed": 42},
                {"route": "XUDATA_ONLY"},
            )
            decision = pd.read_json(out_dir / "decision.json", typ="series")
            self.assertFalse(bool(decision["calibration_used"]))
            self.assertFalse(bool(decision["test_used"]))
            self.assertFalse(bool(decision["raw_data_modified"]))
            self.assertTrue((out_dir / "artifact_manifest.json").is_file())
            self.assertTrue((out_dir / "completed.json").is_file())


if __name__ == "__main__":
    unittest.main()
