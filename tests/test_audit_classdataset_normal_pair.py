import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from experiments.audit_classdataset_normal_pair import (
    audit_zip,
    infer_label,
    run_pair_audit,
)


class ClassDatasetNormalPairAuditTests(unittest.TestCase):
    @staticmethod
    def _zip_bytes(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_label_inference_does_not_treat_abnormal_as_normal(self):
        self.assertEqual(
            infer_label("abnormalClassDataset/processed_dataset/1_ASC-US/a.jpg"),
            "ASC-US",
        )
        self.assertEqual(
            infer_label("normalClassDataSet/processed_dataset/normal/a.jpg"),
            "NIML",
        )
        self.assertEqual(infer_label("processed_dataset/unknown/a.jpg"), "unknown")

    def test_audit_zip_counts_labels_and_exact_screening_keys(self):
        payload = b"same"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ClassDataset.zip"
            path.write_bytes(
                self._zip_bytes(
                    [
                        (
                            "abnormalClassDataset/processed_dataset/1_ASC-US/a.jpg",
                            payload,
                        ),
                        (
                            "abnormalClassDataset/processed_dataset/4_HSIL/b.jpg",
                            b"hsil",
                        ),
                        ("README.md", b"metadata"),
                    ]
                )
            )

            result = audit_zip(path, "ClassDataset.zip")

        self.assertEqual(result["summary"]["image_count"], 2)
        self.assertEqual(
            Counter(
                row["label_candidate"]
                for row in result["records"]
                if row["role"] == "image"
            ),
            Counter({"ASC-US": 1, "HSIL": 1}),
        )
        self.assertEqual(result["summary"]["missing_expected_labels"], [
            "NIML",
            "LSIL",
            "ASC-H",
        ])

    def test_pair_audit_reports_cross_source_content_overlap(self):
        payload = b"same-image"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            (root / "ClassDataset.zip").write_bytes(
                self._zip_bytes(
                    [("processed/1_ASC-US/cell_01.jpg", payload)]
                )
            )
            (root / "normalClassDataSet.zip").write_bytes(
                self._zip_bytes(
                    [("processed/normal/cell_01.jpg", payload)]
                )
            )
            result = run_pair_audit(
                root=root,
                out_dir=Path(tmp) / "audit",
                class_source="ClassDataset.zip",
                normal_source="normalClassDataSet.zip",
            )

            self.assertEqual(result["cross_source_content_overlap_key_count"], 1)
            self.assertEqual(result["cross_source_filename_overlap_key_count"], 1)
            self.assertFalse(result["raw_data_modified"])
            self.assertFalse(result["archives_extracted"])
            payload = json.loads(
                (Path(tmp) / "audit" / "pair_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["route"], "PAIR_AUDIT_COMPLETE_REVIEW_REQUIRED")
            self.assertTrue((Path(tmp) / "audit" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
