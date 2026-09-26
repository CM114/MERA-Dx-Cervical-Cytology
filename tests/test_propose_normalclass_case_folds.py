import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.propose_normalclass_case_folds import propose_case_folds


class NormalClassCaseFoldTests(unittest.TestCase):
    @staticmethod
    def _write_audit(directory):
        labels = {
            "NIML": 5,
            "ASC-US": 5,
            "LSIL": 5,
            "ASC-H": 5,
            "HSIL": 5,
        }
        rows = []
        for label, count in labels.items():
            for index in range(count):
                rows.append(
                    {
                        "member_path": (
                            f"root/normalClassDataset_{label}/case_{label}_{index}/p.jpg"
                        ),
                        "label_candidate": label,
                        "case_id_candidate": f"case_{label}_{index}",
                        "case_key_candidate": (
                            f"normalClassDataset_{label}/case_{label}_{index}"
                        ),
                        "size_bytes": "10",
                        "crc_size_key": f"{label}_{index}:10",
                        "filename_key": "p",
                        "status": "ok",
                    }
                )
        with (directory / "case_inventory.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "unknown_label_image_count": 0,
                    "unknown_case_image_count": 0,
                    "case_label_conflict_count": 0,
                    "cross_case_duplicate_content_key_count": 0,
                    "cross_label_duplicate_content_key_count": 0,
                    "training_manifest_generated": False,
                }
            ),
            encoding="utf-8",
        )

    def test_five_fold_assignment_keeps_cases_together_and_covers_each_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            self._write_audit(audit_dir)
            out_dir = Path(tmp) / "folds"

            summary = propose_case_folds(
                audit_dir=audit_dir,
                out_dir=out_dir,
                seed=42,
                folds=5,
            )

            self.assertEqual(summary["route"], "CASE_FOLD_PROPOSAL_REVIEW_REQUIRED")
            self.assertFalse(summary["training_manifest_generated"])
            with (out_dir / "case_folds.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            by_case = {}
            for row in rows:
                by_case.setdefault(row["case_key"], set()).add(row["fold"])
            self.assertTrue(all(len(folds) == 1 for folds in by_case.values()))
            for fold in range(5):
                held_out_labels = {
                    row["label"] for row in rows if int(row["fold"]) == fold
                }
                self.assertEqual(
                    held_out_labels,
                    {"NIML", "ASC-US", "LSIL", "ASC-H", "HSIL"},
                )
            self.assertTrue((out_dir / "fold_summary.csv").is_file())
            self.assertTrue((out_dir / "report.md").is_file())

    def test_fold_proposal_rejects_integrity_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            self._write_audit(audit_dir)
            summary_path = audit_dir / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["unknown_case_image_count"] = 1
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "integrity gate"):
                propose_case_folds(
                    audit_dir=audit_dir,
                    out_dir=Path(tmp) / "folds",
                    seed=42,
                    folds=5,
                )


if __name__ == "__main__":
    unittest.main()
