import csv
import tempfile
import unittest
from pathlib import Path

from experiments.tbs.multisource import (
    build_source_schedule,
    manifest_has_verified_case_identity,
    validate_tbs5_manifest,
)


def write_manifest(path, rows):
    fields = [
        "image_path",
        "diagnosis_label",
        "diagnosis_name",
        "screen_label",
        "morph_label",
        "evidence_label",
        "semantic_mask",
        "maturity_label",
        "maturity_name",
    ]
    extra = sorted({key for row in rows for key in row if key not in fields})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)


class MultiSourceManifestTests(unittest.TestCase):
    def test_schedule_respects_target_fraction(self):
        schedule = build_source_schedule(20, target_fraction=0.7)
        self.assertEqual(len(schedule), 20)
        self.assertEqual(schedule.count("target"), 14)
        self.assertEqual(schedule.count("supplement"), 6)
        self.assertLessEqual(abs(schedule.count("target") / 20 - 0.7), 0.05)

    def test_both_sources_must_use_the_same_tbs5_label_order(self):
        rows = [
            {
                "image_path": "x.jpg",
                "diagnosis_label": label,
                "diagnosis_name": name,
                "screen_label": int(label > 0),
                "morph_label": -1 if label == 0 else 0,
                "evidence_label": -1 if label == 0 else 0,
                "semantic_mask": int(label > 0),
                "maturity_label": -1,
                "maturity_name": "Unknown",
            }
            for label, name in enumerate(("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(path, rows)
            self.assertEqual(validate_tbs5_manifest(path), 5)

    def test_supplement_without_case_identity_cannot_be_case_evaluation_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xudata.csv"
            write_manifest(
                path,
                [
                    {
                        "image_path": "x.jpg",
                        "diagnosis_label": 0,
                        "diagnosis_name": "Normal",
                        "screen_label": 0,
                        "morph_label": -1,
                        "evidence_label": -1,
                        "semantic_mask": 0,
                        "maturity_label": -1,
                        "maturity_name": "Unknown",
                        "patient_id": "",
                        "slide_id": "",
                    }
                ],
            )
            self.assertFalse(manifest_has_verified_case_identity(path))

    def test_target_dev_is_the_only_allowed_evaluation_source(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target_dev.csv"
            supplement = Path(directory) / "supplement_dev.csv"
            row = {
                "image_path": "x.jpg",
                "diagnosis_label": 0,
                "diagnosis_name": "Normal",
                "screen_label": 0,
                "morph_label": -1,
                "evidence_label": -1,
                "semantic_mask": 0,
                "maturity_label": -1,
                "maturity_name": "Unknown",
            }
            write_manifest(target, [{**row, "case_key": "case-1"}])
            write_manifest(supplement, [row])
            self.assertTrue(manifest_has_verified_case_identity(target, source="target"))
            self.assertFalse(manifest_has_verified_case_identity(supplement, source="supplement"))


if __name__ == "__main__":
    unittest.main()
