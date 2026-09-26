import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.audit_tbs_fold_structure import audit_fold_structure


class FoldStructureAuditTests(unittest.TestCase):
    def test_audit_writes_counts_and_optional_provenance_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "folds"
            for fold in range(5):
                fold_dir = root / f"fold_{fold}"
                fold_dir.mkdir(parents=True)
                frame = pd.DataFrame({
                    "image_path": [f"/data/case_{fold}/a.png", f"/data/case_{fold}/b.png"],
                    "diagnosis_label": [0, 1],
                    "case_id": [f"case_{fold}", f"case_{fold}"],
                })
                frame.to_csv(fold_dir / "train.csv", index=False)
                frame.iloc[:1].to_csv(fold_dir / "val.csv", index=False)
            out = Path(tmp) / "audit"
            report = audit_fold_structure(root, out)
            self.assertEqual(report["fold_count"], 5)
            self.assertEqual(report["folds"]["0"]["train_rows"], 2)
            self.assertTrue((out / "report.json").is_file())
            self.assertEqual(json.loads((out / "report.json").read_text())["fold_count"], 5)


if __name__ == "__main__":
    unittest.main()
