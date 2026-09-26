import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.build_xudata_tbs5_clean_v2 import build_clean_manifests
from experiments.tbs.labels import CSV_FIELDS, labels_for_folders


MANIFEST_NAMES = {
    "train": "train_xudata_tbs5.csv",
    "dev": "dev_xudata_tbs5.csv",
    "calibration": "calibration_xudata_tbs5.csv",
    "test": "test_xudata_tbs5.csv",
}


class CleanManifestBuilderTests(unittest.TestCase):
    @staticmethod
    def make_row(path, split, source_split, diagnosis, maturity="2_Parabasal"):
        return {
            "image_path": path,
            "split": split,
            "source_split": source_split,
            **labels_for_folders(diagnosis, maturity),
            "patient_id": "",
            "slide_id": "",
        }

    @staticmethod
    def write_inputs(csv_dir, rows, hashes):
        csv_dir.mkdir(parents=True, exist_ok=True)
        for split, filename in MANIFEST_NAMES.items():
            split_rows = [row for row in rows if row["split"] == split]
            with (csv_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(split_rows)

        inventory_rows = []
        for row in rows:
            inventory_rows.append(
                {
                    "image_path": row["image_path"],
                    "content_sha256": hashes[row["image_path"]],
                    "read_error": "",
                }
            )
        pd.DataFrame(inventory_rows).to_csv(
            csv_dir / "image_inventory.csv", index=False
        )

    def test_deduplicates_prefers_test_and_quarantines_label_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "csv"
            out_dir = root / "clean"
            rows = [
                self.make_row("/train/lsil.jpg", "train", "train", "2_LSIL"),
                self.make_row(
                    "/train/lsil - 副本.jpg", "dev", "train", "2_LSIL"
                ),
                self.make_row("/train/asch.jpg", "train", "train", "4_ASC-H"),
                self.make_row("/train/hsil.jpg", "train", "train", "5_HSIL"),
                self.make_row("/train/test-copy.jpg", "train", "train", "2_LSIL"),
                self.make_row("/val/test.jpg", "test", "val", "2_LSIL"),
                self.make_row("/val/normal.jpg", "test", "val", "0_NIML"),
                self.make_row("/train/unique.jpg", "calibration", "train", "1_ASC-US"),
            ]
            hashes = {
                "/train/lsil.jpg": "same-train",
                "/train/lsil - 副本.jpg": "same-train",
                "/train/asch.jpg": "label-conflict",
                "/train/hsil.jpg": "label-conflict",
                "/train/test-copy.jpg": "train-test",
                "/val/test.jpg": "train-test",
                "/val/normal.jpg": "test-unique",
                "/train/unique.jpg": "train-unique",
            }
            self.write_inputs(csv_dir, rows, hashes)

            summary = build_clean_manifests(
                csv_dir=csv_dir,
                inventory_csv=csv_dir / "image_inventory.csv",
                out_dir=out_dir,
                train_ratio=0.8,
                dev_ratio=0.1,
                calibration_ratio=0.1,
                seed=20260730,
            )

            output = pd.concat(
                [
                    pd.read_csv(out_dir / f"{split}_xudata_tbs5_clean_v2.csv")
                    for split in MANIFEST_NAMES
                ],
                ignore_index=True,
            )
            self.assertEqual(output["content_sha256"].nunique(), len(output))
            self.assertEqual(
                set(output.loc[output["content_sha256"] == "train-test", "source_split"]),
                {"val"},
            )
            self.assertNotIn("label-conflict", set(output["content_sha256"]))
            self.assertIn("/train/lsil.jpg", set(output["image_path"]))
            self.assertNotIn("/train/lsil - 副本.jpg", set(output["image_path"]))

            quarantine = pd.read_csv(out_dir / "quarantine_label_conflicts.csv")
            self.assertEqual(set(quarantine["image_path"]), {"/train/asch.jpg", "/train/hsil.jpg"})
            self.assertEqual(summary["quarantined_label_conflict_images"], 2)
            self.assertEqual(summary["cross_source_duplicate_groups"], 1)

    def test_output_is_deterministic_and_preserves_original_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "csv"
            first_out = root / "first"
            second_out = root / "second"
            rows = []
            hashes = {}
            diagnoses = ("0_NIML", "1_ASC-US", "2_LSIL", "4_ASC-H", "5_HSIL")
            for diagnosis_index, diagnosis in enumerate(diagnoses):
                for index in range(20):
                    path = f"/train/{diagnosis_index}_{index}.jpg"
                    rows.append(self.make_row(path, "train", "train", diagnosis))
                    hashes[path] = f"hash-{diagnosis_index}-{index}"
                test_path = f"/val/{diagnosis_index}.jpg"
                rows.append(self.make_row(test_path, "test", "val", diagnosis))
                hashes[test_path] = f"test-hash-{diagnosis_index}"
            self.write_inputs(csv_dir, rows, hashes)
            original_train = (csv_dir / MANIFEST_NAMES["train"]).read_bytes()

            first = build_clean_manifests(
                csv_dir, csv_dir / "image_inventory.csv", first_out, seed=42
            )
            second = build_clean_manifests(
                csv_dir, csv_dir / "image_inventory.csv", second_out, seed=42
            )

            self.assertEqual(first["counts"], {"train": 80, "dev": 10, "calibration": 10, "test": 5})
            self.assertEqual(first["counts"], second["counts"])
            for split in MANIFEST_NAMES:
                first_bytes = (first_out / f"{split}_xudata_tbs5_clean_v2.csv").read_bytes()
                second_bytes = (second_out / f"{split}_xudata_tbs5_clean_v2.csv").read_bytes()
                self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                original_train,
                (csv_dir / MANIFEST_NAMES["train"]).read_bytes(),
            )

            schema = json.loads(
                (first_out / "label_schema_tbs5_clean_v2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(schema["schema_version"], "xudata-tbs5-clean-v2")
            self.assertEqual(schema["split_seed"], 42)


if __name__ == "__main__":
    unittest.main()
