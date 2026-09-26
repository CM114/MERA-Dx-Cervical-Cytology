import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from experiments.prepare_normalclass_manifests import prepare_manifests


class PrepareNormalclassManifestsTests(unittest.TestCase):
    def test_prepares_case_isolated_five_class_outer_fold_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            source_zip = root / "normalClassDataSet.zip"

            fold_rows = []
            labels = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
            with zipfile.ZipFile(source_zip, "w") as archive:
                for fold in range(5):
                    for label_index, label in enumerate(labels):
                        case_id = f"case_{fold}_{label_index}"
                        member = (
                            f"normalClassDataSet/normalClassDataset_{label}/{case_id}/patch.jpg"
                        )
                        image_path = image_dir / f"{case_id}.jpg"
                        Image.new("RGB", (8, 8), (fold * 30, 20, 10)).save(image_path)
                        archive.write(image_path, member)
                        fold_rows.append(
                            {
                                "case_key": f"normalClassDataset_{label}/{case_id}",
                                "case_id": case_id,
                                "label": label,
                                "fold": str(fold),
                                "image_count": "1",
                                "representative_member": member,
                            }
                        )

            folds_csv = root / "case_folds.csv"
            with folds_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fold_rows[0].keys())
                writer.writeheader()
                writer.writerows(fold_rows)

            summary = prepare_manifests(
                source_zip=source_zip,
                case_folds_csv=folds_csv,
                staging_dir=root / "staging",
                out_dir=root / "manifests",
                folds=5,
            )

            self.assertEqual(summary["image_count"], 25)
            self.assertEqual(summary["case_count"], 25)
            for outer_fold in range(5):
                fold_dir = root / "manifests" / f"fold_{outer_fold}"
                self.assertTrue((fold_dir / "train.csv").is_file())
                self.assertTrue((fold_dir / "dev.csv").is_file())
                self.assertTrue((fold_dir / "heldout.csv").is_file())

                frames = {}
                for split in ("train", "dev", "heldout"):
                    with (fold_dir / f"{split}.csv").open(
                        newline="", encoding="utf-8"
                    ) as handle:
                        frames[split] = list(csv.DictReader(handle))

                case_sets = {
                    split: {row["case_id"] for row in rows}
                    for split, rows in frames.items()
                }
                self.assertEqual(
                    set().union(*case_sets.values()),
                    {f"case_{fold}_{label}" for fold in range(5) for label in range(5)},
                )
                self.assertEqual(len(case_sets["train"] & case_sets["dev"]), 0)
                self.assertEqual(len(case_sets["train"] & case_sets["heldout"]), 0)
                self.assertEqual(len(case_sets["dev"] & case_sets["heldout"]), 0)
                self.assertEqual(
                    {int(row["diagnosis_label"]) for rows in frames.values() for row in rows},
                    set(range(5)),
                )
                for row in [row for rows in frames.values() for row in rows]:
                    self.assertTrue(Path(row["image_path"]).is_file())
                    self.assertEqual(row["maturity_label"], "-1")
                    self.assertEqual(row["maturity_name"], "Unknown")

            output_summary = json.loads(
                (root / "manifests" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output_summary["fold_count"], 5)
            self.assertTrue(output_summary["case_isolation_passed"])

    def test_excludes_members_from_integrity_quarantine_with_audit_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "normalClassDataSet.zip"
            fold_rows = []
            labels = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
            excluded_member = (
                "normalClassDataSet/normalClassDataset_NIML/"
                "case_0_0/empty.jpg"
            )

            with zipfile.ZipFile(source_zip, "w") as archive:
                for fold in range(5):
                    for label_index, label in enumerate(labels):
                        case_id = f"case_{fold}_{label_index}"
                        member = (
                            f"normalClassDataSet/normalClassDataset_{label}/"
                            f"{case_id}/patch.jpg"
                        )
                        image_path = root / f"{case_id}.jpg"
                        Image.new("RGB", (8, 8), (fold * 30, 20, 10)).save(image_path)
                        archive.write(image_path, member)
                        image_count = 2 if member.startswith(
                            "normalClassDataSet/normalClassDataset_NIML/case_0_0/"
                        ) else 1
                        if image_count == 2:
                            archive.writestr(excluded_member, b"")
                        fold_rows.append(
                            {
                                "case_key": f"normalClassDataset_{label}/{case_id}",
                                "case_id": case_id,
                                "label": label,
                                "fold": str(fold),
                                "image_count": str(image_count),
                                "representative_member": member,
                            }
                        )

            folds_csv = root / "case_folds.csv"
            with folds_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fold_rows[0].keys())
                writer.writeheader()
                writer.writerows(fold_rows)

            exclusions_csv = root / "bad_images.csv"
            with exclusions_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["member_path", "reason"])
                writer.writeheader()
                writer.writerow({"member_path": excluded_member, "reason": "zero_byte"})

            summary = prepare_manifests(
                source_zip=source_zip,
                case_folds_csv=folds_csv,
                staging_dir=root / "staging",
                out_dir=root / "manifests",
                folds=5,
                exclude_members_csv=exclusions_csv,
            )

            self.assertEqual(summary["source_image_count"], 26)
            self.assertEqual(summary["image_count"], 25)
            self.assertEqual(summary["excluded_image_count"], 1)
            self.assertEqual(summary["excluded_members"], [excluded_member])
            self.assertFalse((root / "staging" / excluded_member).exists())
            self.assertTrue((root / "manifests" / "excluded_members.csv").is_file())
            for outer_fold in range(5):
                for split in ("train", "dev", "heldout"):
                    manifest = root / "manifests" / f"fold_{outer_fold}" / f"{split}.csv"
                    with manifest.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertNotIn(excluded_member, {row["source_archive_member"] for row in rows})


if __name__ == "__main__":
    unittest.main()
