import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.audit_wsl_class_dataset import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    prepare_output_directory,
    run_audit,
)


class WslClassDatasetAuditTests(unittest.TestCase):
    def test_audit_groups_cases_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "WSL_class_DataSet"
            (root / "NIML" / "HospitalA" / "P001" / "normal").mkdir(parents=True)
            (root / "LISL" / "HospitalB" / "P002" / "result").mkdir(parents=True)
            (root / "NIML" / "HospitalA" / "P001" / "normal" / "a.jpg").write_bytes(b"a")
            (root / "NIML" / "HospitalA" / "P001" / "normal" / "b.txt").write_text("x", encoding="utf-8")
            (root / "LISL" / "HospitalB" / "P002" / "result" / "wsi_classification.json").write_text("{}", encoding="utf-8")
            (root / "LISL" / "HospitalB" / "P002" / "result" / "f.npy").write_bytes(b"npy")
            out_dir = Path(tmp) / "audit"

            summary = run_audit(root, out_dir, progress_every=0)

            self.assertEqual(summary["route"], "AUDIT_COMPLETE")
            self.assertEqual(summary["total_files"], 4)
            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["raw_data_modified"], False)
            self.assertEqual(summary["training_manifest_generated"], False)
            self.assertTrue((out_dir / OWNER_FILENAME).is_file())
            for filename in ARTIFACT_FILENAMES:
                self.assertTrue((out_dir / filename).is_file(), filename)

            with (out_dir / "label_summary.csv").open(newline="", encoding="utf-8") as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual({row["raw_label"] for row in labels}, {"NIML", "LISL"})
            self.assertEqual(
                json.loads((out_dir / "completed.json").read_text(encoding="utf-8"))["status"],
                "completed",
            )

    def test_audit_reports_truncation_and_skips_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            case_dir = root / "NIML" / "H" / "P"
            case_dir.mkdir(parents=True)
            for index in range(3):
                (case_dir / f"{index}.jpg").write_bytes(b"x")
            linked = root / "linked_case"
            try:
                linked.symlink_to(case_dir, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            summary = run_audit(
                root,
                Path(tmp) / "audit",
                max_files=2,
                progress_every=0,
            )

            self.assertEqual(summary["route"], "AUDIT_COMPLETE_WITH_TRUNCATION")
            self.assertEqual(summary["total_files"], 2)
            self.assertGreaterEqual(summary["skipped_symlink_count"], 1)

    def test_output_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside input source"):
                prepare_output_directory(root / "audit", root, overwrite=False)

    def test_linked_owner_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            out_dir = root / "audit"
            prepare_output_directory(out_dir, source, overwrite=False)
            marker = out_dir / OWNER_FILENAME
            marker.unlink()
            external = root / "external-owner.json"
            external.write_text('{"keep": true}\n', encoding="utf-8")
            try:
                marker.symlink_to(external)
            except OSError:
                os.link(external, marker)

            with self.assertRaisesRegex(FileExistsError, "ownership marker"):
                prepare_output_directory(out_dir, source, overwrite=True)

            self.assertEqual(external.read_text(encoding="utf-8"), '{"keep": true}\n')


if __name__ == "__main__":
    unittest.main()
