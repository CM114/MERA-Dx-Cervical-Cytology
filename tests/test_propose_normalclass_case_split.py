import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.propose_normalclass_case_split import propose_case_split


class NormalClassCaseSplitTests(unittest.TestCase):
    @staticmethod
    def _write_audit(directory, labels):
        rows = []
        for label, case_count in labels.items():
            for index in range(case_count):
                rows.append(
                    {
                        "member_path": (
                            f"normalClassDataSet/normalClassDataset_{label}/"
                            f"case_{label}_{index}/patch_{index}.jpg"
                        ),
                        "label_candidate": label,
                        "case_id_candidate": f"case_{label}_{index}",
                        "case_key_candidate": (
                            f"normalClassDataset_{label}/case_{label}_{index}"
                        ),
                        "size_bytes": "10",
                        "crc_size_key": f"{index:08x}:10",
                        "filename_key": f"patch_{index}",
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
                    "route": "CASE_INTEGRITY_REVIEW_REQUIRED",
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

    def test_proposal_assigns_each_case_once_and_preserves_label_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            self._write_audit(
                audit_dir,
                {"NIML": 3, "ASC-US": 4, "LSIL": 4, "ASC-H": 5, "HSIL": 3},
            )
            out_dir = Path(tmp) / "split"

            summary = propose_case_split(
                audit_dir=audit_dir,
                out_dir=out_dir,
                seed=42,
            )

            self.assertEqual(summary["route"], "CASE_SPLIT_PROPOSAL_REVIEW_REQUIRED")
            self.assertFalse(summary["training_manifest_generated"])
            with (out_dir / "case_split.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len({row["case_key"] for row in rows}))
            self.assertEqual(
                {
                    (row["label"], row["split"])
                    for row in rows
                    if row["label"] == "ASC-H"
                },
                {("ASC-H", "train"), ("ASC-H", "dev"), ("ASC-H", "test")},
            )
            self.assertTrue((out_dir / "split_summary.csv").is_file())
            self.assertTrue((out_dir / "report.md").is_file())

    def test_proposal_rejects_failed_integrity_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            audit_dir.mkdir()
            self._write_audit(audit_dir, {"NIML": 3})
            payload = json.loads((audit_dir / "summary.json").read_text())
            payload["cross_case_duplicate_content_key_count"] = 1
            (audit_dir / "summary.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "integrity gate"):
                propose_case_split(
                    audit_dir=audit_dir,
                    out_dir=Path(tmp) / "split",
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()
