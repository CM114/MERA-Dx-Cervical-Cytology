import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from experiments.deep_audit_candidate_archives import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    run_audit,
)


class DeepCandidateArchiveAuditTests(unittest.TestCase):
    @staticmethod
    def _zip_bytes(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_nested_zip_members_are_scanned_without_disk_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources"
            root.mkdir()
            nested = self._zip_bytes(
                [
                    ("train/Normal/n1.jpg", b"normal"),
                    ("train/ASC-US/a1.jpg", b"asc-us"),
                    ("dev/HSIL/h1.jpg", b"hsil"),
                    (
                        "metadata.csv",
                        b"patient_id,label\nP1,Normal\nP2,HSIL\n",
                    ),
                ]
            )
            outer = root / "AICCS.zip"
            with zipfile.ZipFile(outer, "w") as archive:
                archive.writestr("release/images.zip", nested)
                archive.writestr("README.md", "case-level cervical cytology")

            out_dir = Path(tmp) / "results"
            summary = run_audit(
                root=root,
                out_dir=out_dir,
                sources=["AICCS.zip"],
            )

            self.assertEqual(summary["route"], "AUDIT_COMPLETE")
            self.assertEqual(summary["source_count"], 1)
            self.assertEqual(summary["nested_archive_count"], 1)
            self.assertEqual(summary["image_count"], 3)
            self.assertEqual(
                set(summary["label_candidates"]),
                {"Normal", "ASC-US", "HSIL"},
            )
            classes = pd.read_csv(out_dir / "deep_class_distribution.csv")
            self.assertIn("Normal", set(classes["five_class_candidate"]))
            self.assertIn("ASC-US", set(classes["five_class_candidate"]))
            nested_rows = pd.read_csv(out_dir / "deep_nested_archives.csv")
            self.assertEqual(len(nested_rows), 1)
            self.assertEqual(nested_rows.loc[0, "status"], "scanned")
            self.assertFalse((root / "release").exists())
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                set(ARTIFACT_FILENAMES) | {OWNER_FILENAME},
            )

    def test_unowned_output_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources"
            root.mkdir()
            (root / "source.txt").write_text("source", encoding="utf-8")
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            sentinel = out_dir / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "unowned"):
                run_audit(root, out_dir, sources=["source.txt"], overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_completion_artifact_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources"
            root.mkdir()
            (root / "source.txt").write_text("source", encoding="utf-8")
            out_dir = Path(tmp) / "results"

            run_audit(root, out_dir, sources=["source.txt"])

            completed = json.loads(
                (out_dir / "completed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("completed.json", completed["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
