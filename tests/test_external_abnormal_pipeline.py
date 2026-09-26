import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.audit_classdataset_hsil import audit_source
from experiments.prepare_external_pretrain_manifest import prepare_manifest
from experiments.tbs.external_dataset import ExternalAbnormalDataset


def make_zip(root, members):
    path = Path(root) / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class ExternalAbnormalPipelineTests(unittest.TestCase):
    def test_hsil_audit_reports_labels_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_zip(tmp, {
                "train/ASC-H/case1/a.png": image_bytes("red"),
                "dev/HSIL/case2/b.png": image_bytes("blue"),
                "README.md": b"review",
            })
            before = source.read_bytes()
            output = Path(tmp) / "audit"
            summary = audit_source(source, output)

            self.assertEqual(summary["observed_labels"], ["ASC-H", "HSIL"])
            self.assertEqual(summary["image_count"], 2)
            self.assertFalse(summary["raw_data_modified"])
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "report.md").is_file())

    def test_hsil_audit_reads_nested_zip_members_without_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested_buffer = io.BytesIO()
            with zipfile.ZipFile(nested_buffer, "w") as nested:
                nested.writestr("HSIL/case1/a.png", image_bytes("red"))
            source = make_zip(tmp, {"nested.zip": nested_buffer.getvalue()})
            summary = audit_source(source, Path(tmp) / "audit")
            self.assertEqual(summary["nested_archive_count"], 1)
            self.assertEqual(summary["image_count"], 1)
            self.assertEqual(summary["observed_labels"], ["HSIL"])

    def test_prepare_manifest_creates_deterministic_four_class_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_zip(tmp, {
                "processed/1_ASC-US/case1/a.png": image_bytes("red"),
                "processed/1_ASC-US/case5/e.png": image_bytes("orange"),
                "processed/2_LSIL/case2/b.png": image_bytes("green"),
                "processed/2_LSIL/case6/f.png": image_bytes("yellow"),
                "processed/3_ASC-H/case3/c.png": image_bytes("blue"),
                "processed/3_ASC-H/case7/g.png": image_bytes("purple"),
                "processed/4_HSIL/case4/d.png": image_bytes("white"),
                "processed/4_HSIL/case8/h.png": image_bytes("black"),
            })
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first_summary = prepare_manifest(
                source, Path(tmp) / "stage1", first, dev_fraction=0.5, seed=42
            )
            second_summary = prepare_manifest(
                source, Path(tmp) / "stage2", second, dev_fraction=0.5, seed=42
            )

            self.assertEqual(first_summary["label_order"], ["ASC-US", "LSIL", "ASC-H", "HSIL"])
            self.assertEqual(first_summary["source_zip_sha256"], second_summary["source_zip_sha256"])
            first_rows = pd.read_csv(first / "train.csv")
            second_rows = pd.read_csv(second / "train.csv")
            first_rows["image_path"] = first_rows["source_archive_member"]
            second_rows["image_path"] = second_rows["source_archive_member"]
            pd.testing.assert_frame_equal(first_rows, second_rows)
            frame = ExternalAbnormalDataset(first / "train.csv", transform=None)
            self.assertEqual(len(frame), 4)
            self.assertEqual(frame[0]["label"], 0)

    def test_prepare_manifest_rejects_normal_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_zip(tmp, {"processed/Normal/case/a.png": image_bytes("red")})
            with self.assertRaisesRegex(ValueError, "unknown labels"):
                prepare_manifest(source, Path(tmp) / "stage", Path(tmp) / "out")

    def test_prepare_manifest_marks_direct_class_patches_as_group_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            members = {}
            for label_index, label in enumerate(("ASC-US", "LSIL", "ASC-H", "HSIL"), 1):
                members[f"processed/{label_index}_{label}/patch_{label_index}.png"] = image_bytes((label_index * 40, 10, 10))
                members[f"processed/{label_index}_{label}/patch_{label_index}_b.png"] = image_bytes((10, label_index * 40, 10))
            source = make_zip(tmp, members)
            summary = prepare_manifest(
                source, Path(tmp) / "stage", Path(tmp) / "out", dev_fraction=0.5
            )
            self.assertEqual(summary["grouping_policy"], "unknown_image_level_split")
            self.assertIsNone(summary["case_count"])

    def test_prepare_manifest_rejects_crc_size_overlap_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = image_bytes("red")
            members = {}
            for label_index, label in enumerate(("ASC-US", "LSIL", "ASC-H", "HSIL"), 1):
                members[f"processed/{label_index}_{label}/a.png"] = shared
                members[f"processed/{label_index}_{label}/b.png"] = image_bytes((label_index * 20, 5, 5))
            source = make_zip(tmp, members)
            with self.assertRaisesRegex(ValueError, "CRC-size content overlap"):
                prepare_manifest(source, Path(tmp) / "stage", Path(tmp) / "out", dev_fraction=0.5)

    def test_prepare_manifest_rejects_unsafe_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_zip(tmp, {
                "processed/1_ASC-US/../escape.png": image_bytes("red"),
                "processed/2_LSIL/b.png": image_bytes("green"),
                "processed/3_ASC-H/c.png": image_bytes("blue"),
                "processed/4_HSIL/d.png": image_bytes("white"),
            })
            with self.assertRaisesRegex(ValueError, "Unsafe archive member path"):
                prepare_manifest(source, Path(tmp) / "stage", Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
