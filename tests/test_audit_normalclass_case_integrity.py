import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.audit_normalclass_case_integrity import (
    infer_case_key,
    infer_label,
    run_integrity_audit,
)


class NormalClassCaseIntegrityTests(unittest.TestCase):
    @staticmethod
    def _zip_bytes(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_case_and_label_inference_uses_case_directory(self):
        path = (
            "normalClassDataSet/normalClassDataset_HSIL/"
            "case_2025_001/8192_34242_256_256.jpg"
        )
        self.assertEqual(infer_label(path), "HSIL")
        self.assertEqual(infer_case_key(path), "normalClassDataset_HSIL/case_2025_001")
        self.assertEqual(infer_case_key("normalClassDataSet/HSIL/image.jpg"), "unknown")

    def test_audit_detects_cross_case_and_cross_label_content_duplicates(self):
        payload = b"same-patch"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "normalClassDataSet.zip"
            source.write_bytes(
                self._zip_bytes(
                    [
                        (
                            "normalClassDataSet/normalClassDataset_HSIL/case_a/a.jpg",
                            payload,
                        ),
                        (
                            "normalClassDataSet/normalClassDataset_HSIL/case_b/b.jpg",
                            payload,
                        ),
                        (
                            "normalClassDataSet/normalClassDataset_LSIL/case_c/c.jpg",
                            payload,
                        ),
                    ]
                )
            )

            result = run_integrity_audit(
                source=source,
                out_dir=Path(tmp) / "audit",
            )

        self.assertEqual(result["image_count"], 3)
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["cross_case_duplicate_content_key_count"], 1)
        self.assertEqual(result["cross_label_duplicate_content_key_count"], 1)
        self.assertEqual(result["case_label_conflict_count"], 0)

    def test_audit_writes_case_artifacts_and_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "normalClassDataSet.zip"
            source.write_bytes(
                self._zip_bytes(
                    [
                        (
                            "normalClassDataSet/normalClassDataset_NIML/case_a/a.jpg",
                            b"image",
                        )
                    ]
                )
            )
            out_dir = Path(tmp) / "audit"
            result = run_integrity_audit(source=source, out_dir=out_dir)

            payload = json.loads(
                (out_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["route"], "CASE_INTEGRITY_REVIEW_REQUIRED")
            self.assertFalse(payload["archives_extracted"])
            self.assertFalse(payload["training_manifest_generated"])
            self.assertTrue((out_dir / "case_inventory.csv").is_file())
            self.assertTrue((out_dir / "content_duplicate_summary.csv").is_file())
            self.assertTrue((out_dir / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
