import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.C3.splits import build_risk_splits


class C3SplitTests(unittest.TestCase):
    def test_group_is_never_split_and_every_row_is_held_out_once(self):
        rows = []
        for group in range(12):
            for j in range(2):
                rows.append({"patient_id": f"p{group}", "label": (group + j) % 5, "image_path": f"g{group}_{j}.png"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = build_risk_splits(source, root / "splits", n_folds=3, seed=42)
            heldout = []
            for fold in range(3):
                train = pd.read_csv(result["folds"][fold]["train_csv"])
                test = pd.read_csv(result["folds"][fold]["heldout_csv"])
                self.assertTrue(set(train.patient_id).isdisjoint(set(test.patient_id)))
                heldout.extend(test.image_path.tolist())
            self.assertEqual(sorted(heldout), sorted(pd.DataFrame(rows).image_path.tolist()))

    def test_tbs_manifest_diagnosis_label_is_normalized_to_label(self):
        rows = [{"patient_id": f"p{i}", "diagnosis_label": i % 5, "image_path": f"x{i}.png"} for i in range(9)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = build_risk_splits(source, root / "splits", n_folds=3, seed=42)
            heldout = pd.read_csv(result["folds"][0]["heldout_csv"])
            self.assertIn("label", heldout.columns)
            self.assertIn("diagnosis_label", heldout.columns)
            self.assertEqual(heldout["label"].astype(int).tolist(), heldout["diagnosis_label"].astype(int).tolist())

    def test_missing_content_hash_does_not_create_one_giant_group(self):
        rows = [{"patient_id": None, "slide_id": None, "content_sha256": None, "image_path": f"x{i}.png", "diagnosis_label": i % 5} for i in range(12)]
        rows[0]["content_sha256"] = "real_hash"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = build_risk_splits(source, root / "splits", n_folds=3, seed=42)
            self.assertEqual(result["group_source"], "image_path")
            sizes = [result["folds"][fold]["n_heldout"] for fold in range(3)]
            self.assertTrue(all(size >= 3 for size in sizes))


if __name__ == "__main__":
    unittest.main()
