import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from experiments.deep_audit_cervical_candidates import (
    EXPECTED_LABELS,
    audit_source,
    infer_group_key,
    infer_label,
    infer_split,
    run_audit,
)


class DeepCandidateAuditTests(unittest.TestCase):
    @staticmethod
    def _zip_bytes(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_label_split_and_group_inference_are_conservative(self):
        self.assertEqual(infer_label("train/ASC-US/case_01_cell_2.png"), "ASC-US")
        self.assertEqual(infer_label("test/normal/case_02.png"), "NIML")
        self.assertEqual(infer_split("dataset/validation/HSIL/case.png"), "dev")
        self.assertEqual(infer_split("ASC-US/case.png"), "unknown")
        self.assertEqual(
            infer_group_key("train/HSIL/patient_0007/cell_03.png"),
            "patient_0007",
        )
        self.assertEqual(infer_group_key("HSIL/cell_03.png"), "unknown")
        self.assertEqual(
            set(EXPECTED_LABELS), {"NIML", "ASC-US", "LSIL", "ASC-H", "HSIL"}
        )

    def test_nested_zip_inventory_counts_labels_and_cross_split_content(self):
        shared = b"same-image"
        nested = self._zip_bytes(
            [
                ("train/NIML/patient_01/cell_01.png", shared),
                ("test/HSIL/patient_02/cell_02.png", shared),
                ("test/HSIL/patient_02/cell_03.jpg", b"different"),
            ]
        )
        outer = self._zip_bytes(
            [
                ("release/images.zip", nested),
                ("README.md", b"cervical cytology"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "candidate.zip"
            source.write_bytes(outer)

            result = audit_source(source, "candidate.zip", max_entries=100)

        records = result["records"]
        self.assertEqual(Counter(row["role"] for row in records)["image"], 3)
        self.assertEqual(
            Counter(row["label_candidate"] for row in records if row["role"] == "image"),
            Counter({"NIML": 1, "HSIL": 2}),
        )
        summary = result["summary"]
        self.assertEqual(summary["nested_archive_count"], 1)
        self.assertEqual(summary["cross_split_content_duplicate_key_count"], 1)
        self.assertEqual(summary["missing_expected_labels"], ["ASC-US", "LSIL", "ASC-H"])
        self.assertFalse(summary["five_class_candidate"])

    def test_run_audit_writes_review_artifacts_and_safety_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            source = root / "AICCS .zip"
            source.write_bytes(
                self._zip_bytes(
                    [("train/NIML/patient.png", b"image")]
                )
            )
            out_dir = Path(tmp) / "audit"

            summary = run_audit(
                root=root,
                out_dir=out_dir,
                sources=["AICCS .zip"],
                max_entries_per_source=100,
            )

            self.assertEqual(summary["route"], "DEEP_AUDIT_COMPLETE_REVIEW_REQUIRED")
            self.assertFalse(summary["archives_extracted"])
            self.assertFalse(summary["model_trained"])
            self.assertTrue((out_dir / "deep_inventory.csv").is_file())
            self.assertTrue((out_dir / "candidate_summary.csv").is_file())
            self.assertTrue((out_dir / "split_summary.csv").is_file())
            self.assertTrue((out_dir / "report.md").is_file())
            payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
