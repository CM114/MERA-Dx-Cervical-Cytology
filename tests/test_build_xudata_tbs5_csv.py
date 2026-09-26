import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.build_xudata_tbs5_csv import (
    build_manifests,
    partition_training_rows,
    scan_source_split,
)
from experiments.tbs.labels import labels_for_folders


class LabelSchemaTests(unittest.TestCase):
    def test_maps_all_five_diagnoses_to_tbs_semantics(self):
        cases = {
            "0_NIML": (0, "Normal", 0, -1, -1, 0),
            "1_ASC-US": (1, "ASC-US", 1, 0, 0, 1),
            "2_LSIL": (2, "LSIL", 1, 0, 1, 1),
            "4_ASC-H": (3, "ASC-H", 1, 1, 0, 1),
            "5_HSIL": (4, "HSIL", 1, 1, 1, 1),
        }

        for folder, expected in cases.items():
            with self.subTest(folder=folder):
                labels = labels_for_folders(folder, "2_Parabasal")
                actual = (
                    labels["diagnosis_label"],
                    labels["diagnosis_name"],
                    labels["screen_label"],
                    labels["morph_label"],
                    labels["evidence_label"],
                    labels["semantic_mask"],
                )
                self.assertEqual(actual, expected)

    def test_keeps_maturity_as_metadata(self):
        expected = {
            "0_Superficial": (0, "Superficial"),
            "1_Intermediate": (1, "Intermediate"),
            "2_Parabasal": (2, "Parabasal"),
        }
        for folder, maturity in expected.items():
            with self.subTest(folder=folder):
                labels = labels_for_folders("1_ASC-US", folder)
                self.assertEqual(
                    (labels["maturity_label"], labels["maturity_name"]),
                    maturity,
                )

    def test_rejects_unknown_folders(self):
        with self.assertRaisesRegex(ValueError, "Unknown diagnosis folder"):
            labels_for_folders("3_UNKNOWN", "2_Parabasal")
        with self.assertRaisesRegex(ValueError, "Unknown maturity folder"):
            labels_for_folders("1_ASC-US", "3_UNKNOWN")


class ManifestBuilderTests(unittest.TestCase):
    @staticmethod
    def make_image(root, source_split, diagnosis, maturity, name):
        path = root / source_split / diagnosis / maturity / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-image")
        return path

    def test_scan_source_split_reads_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "xudata"
            image = self.make_image(
                root, "train", "2_LSIL", "1_Intermediate", "cell.jpg"
            )
            rows = scan_source_split(root, "train")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image_path"], str(image.resolve()))
            self.assertEqual(rows[0]["diagnosis_name"], "LSIL")
            self.assertEqual(rows[0]["maturity_name"], "Intermediate")

    def test_partition_is_deterministic_and_stratified(self):
        rows = []
        for diagnosis_label in (0, 1):
            for index in range(20):
                rows.append(
                    {
                        "image_path": f"/{diagnosis_label}/{index}.jpg",
                        "diagnosis_label": diagnosis_label,
                        "maturity_label": 2,
                    }
                )

        first = partition_training_rows(rows, 0.8, 0.1, 0.1, seed=20260728)
        second = partition_training_rows(rows, 0.8, 0.1, 0.1, seed=20260728)
        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            {"train": 32, "dev": 4, "calibration": 4},
        )

    def test_build_writes_four_manifests_schema_and_audit(self):
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
                    self.make_image(
                        root,
                        "train",
                        diagnosis,
                        "2_Parabasal",
                        f"train_{index}.jpg",
                    )
                for index in range(3):
                    self.make_image(
                        root,
                        "val",
                        diagnosis,
                        "2_Parabasal",
                        f"test_{index}.jpg",
                    )

            summary = build_manifests(root, out_dir, 0.8, 0.1, 0.1, 20260728)
            self.assertEqual(
                summary["counts"],
                {"train": 40, "dev": 5, "calibration": 5, "test": 15},
            )
            expected_files = {
                "train_xudata_tbs5.csv",
                "dev_xudata_tbs5.csv",
                "calibration_xudata_tbs5.csv",
                "test_xudata_tbs5.csv",
                "label_schema_tbs5.json",
                "xudata_tbs5_audit.csv",
            }
            self.assertTrue(expected_files.issubset({p.name for p in out_dir.iterdir()}))

            with (out_dir / "test_xudata_tbs5.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                test_rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in test_rows}, {"test"})
            self.assertEqual({row["source_split"] for row in test_rows}, {"val"})

            with (out_dir / "label_schema_tbs5.json").open(
                encoding="utf-8"
            ) as handle:
                schema = json.load(handle)
            self.assertEqual(schema["schema_version"], "xudata-tbs5-v1")
            self.assertEqual(schema["split_seed"], 20260728)


if __name__ == "__main__":
    unittest.main()

