import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from experiments.package_tbs_audit_reports import (
    PACKAGE_FILENAMES,
    package_reports,
    validate_audit_directory,
)


class TbsAuditReportPackagingTests(unittest.TestCase):
    def _make_completed_audit(self, root):
        audit_dir = Path(root) / "audit"
        audit_dir.mkdir()
        (audit_dir / ".tbs_data_source_audit_owner").write_text(
            json.dumps({"schema": "tbs-candidate-data-source-audit-v1"}) + "\n",
            encoding="utf-8",
        )
        contents = {
            "report.md": "# audit\n",
            "source_scorecard.csv": "source_name,provisional_score\nA,80\n",
            "metadata_field_summary.csv": "source_name,field_name\nA,patient_id\n",
            "split_overlap_summary.csv": "source_name,exact_overlap_scope\nA,partial\n",
        }
        hashes = {}
        for filename, content in contents.items():
            path = audit_dir / filename
            path.write_text(content, encoding="utf-8", newline="")
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        (audit_dir / "completed.json").write_text(
            json.dumps(
                {
                    "schema": "tbs-candidate-data-source-audit-v1",
                    "status": "completed",
                    "artifact_sha256": hashes,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return audit_dir

    def test_valid_audit_is_verified_and_packages_exact_four_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = self._make_completed_audit(tmp)
            output = Path(tmp) / "delivery" / "tbs_reports.tar.gz"

            validation = validate_audit_directory(audit_dir)
            result = package_reports(audit_dir, output)

            self.assertEqual(validation["status"], "completed")
            self.assertEqual(result["members"], list(PACKAGE_FILENAMES))
            with tarfile.open(output, "r:gz") as archive:
                self.assertEqual(archive.getnames(), list(PACKAGE_FILENAMES))

    def test_hash_mismatch_is_rejected_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = self._make_completed_audit(tmp)
            (audit_dir / "report.md").write_text("tampered\n", encoding="utf-8")
            output = Path(tmp) / "delivery.tar.gz"

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                package_reports(audit_dir, output)
            self.assertFalse(output.exists())

    def test_incomplete_marker_and_unowned_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit_unowned"
            audit_dir.mkdir()
            (audit_dir / "completed.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ownership marker"):
                validate_audit_directory(audit_dir)

            owned = self._make_completed_audit(tmp)
            (owned / "completed.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "status"):
                validate_audit_directory(owned)

    def test_output_must_be_outside_audit_and_existing_file_needs_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = self._make_completed_audit(tmp)
            with self.assertRaisesRegex(ValueError, "outside audit"):
                package_reports(audit_dir, audit_dir / "delivery.tar.gz")

            output = Path(tmp) / "delivery.tar.gz"
            package_reports(audit_dir, output)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                package_reports(audit_dir, output)
            package_reports(audit_dir, output, overwrite=True)
            self.assertTrue(output.is_file())

    def test_repeated_packaging_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = self._make_completed_audit(tmp)
            output = Path(tmp) / "delivery.tar.gz"

            package_reports(audit_dir, output)
            first = output.read_bytes()
            package_reports(audit_dir, output, overwrite=True)

            self.assertEqual(first, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
