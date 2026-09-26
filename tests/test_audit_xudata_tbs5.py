import tempfile
import unittest
from pathlib import Path

from experiments.audit_xudata_tbs5 import audit_manifests
from experiments.build_xudata_tbs5_csv import build_manifests


class ManifestAuditTests(unittest.TestCase):
    @staticmethod
    def make_image(root, source_split, diagnosis, name):
        path = root / source_split / diagnosis / "2_Parabasal" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-image")

    def test_accepts_valid_generated_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "xudata"
            out_dir = base / "csv_files"
            diagnoses = (
                "0_NIML",
                "1_ASC-US",
                "2_LSIL",
                "4_ASC-H",
                "5_HSIL",
            )
            for diagnosis in diagnoses:
                for index in range(10):
                    self.make_image(root, "train", diagnosis, f"train_{index}.jpg")
                for index in range(3):
                    self.make_image(root, "val", diagnosis, f"test_{index}.jpg")

            build_manifests(root, out_dir, 0.8, 0.1, 0.1, 20260728)
            report = audit_manifests(out_dir, data_root=root, check_files=True)

            self.assertEqual(report["total"], 65)
            self.assertEqual(report["split_counts"]["test"], 15)
            self.assertEqual(report["errors"], [])

    def test_detects_duplicate_paths_across_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "xudata"
            out_dir = base / "csv_files"
            for diagnosis in (
                "0_NIML",
                "1_ASC-US",
                "2_LSIL",
                "4_ASC-H",
                "5_HSIL",
            ):
                for index in range(10):
                    self.make_image(root, "train", diagnosis, f"train_{index}.jpg")
                for index in range(3):
                    self.make_image(root, "val", diagnosis, f"test_{index}.jpg")

            build_manifests(root, out_dir, 0.8, 0.1, 0.1, 20260728)
            train_csv = out_dir / "train_xudata_tbs5.csv"
            dev_csv = out_dir / "dev_xudata_tbs5.csv"
            with train_csv.open(encoding="utf-8") as handle:
                duplicate_row = handle.readlines()[1]
            with dev_csv.open("a", encoding="utf-8") as handle:
                handle.write(duplicate_row)

            report = audit_manifests(out_dir, data_root=root, check_files=False)
            self.assertTrue(
                any("Duplicate image_path" in error for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
