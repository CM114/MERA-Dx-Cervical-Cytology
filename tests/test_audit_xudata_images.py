import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from experiments.audit_xudata_images import (
    analyze_image,
    current_eval_view,
    current_train_view,
    run_audit,
    whole_cell_letterbox,
)


class ImageAuditPrimitiveTests(unittest.TestCase):
    def test_analyze_image_records_dimensions_and_edge_contact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "edge_cell.png"
            image = Image.new("RGB", (32, 24), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 6, 14, 18), fill=(50, 20, 80))
            image.save(image_path)

            record = analyze_image(image_path)

            self.assertEqual(record["width"], 32)
            self.assertEqual(record["height"], 24)
            self.assertGreaterEqual(record["touched_edge_count"], 1)
            self.assertGreater(record["border_foreground_fraction"], 0.0)
            self.assertEqual(len(record["content_sha256"]), 64)
            self.assertEqual(len(record["dhash"]), 16)

    def test_whole_cell_letterbox_preserves_both_horizontal_edges(self):
        image = Image.new("RGB", (80, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 5, 39), fill="red")
        draw.rectangle((74, 0, 79, 39), fill="blue")

        transformed = whole_cell_letterbox(image, 100, fill=(240, 240, 240))

        self.assertEqual(transformed.size, (100, 100))
        self.assertGreater(transformed.getpixel((1, 50))[0], 200)
        self.assertGreater(transformed.getpixel((98, 50))[2], 200)
        self.assertEqual(transformed.getpixel((50, 5)), (240, 240, 240))

    def test_current_views_are_fixed_size_and_training_view_is_deterministic(self):
        image = Image.new("RGB", (90, 60), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 5, 80, 55), fill=(120, 40, 160))

        evaluation = current_eval_view(image, 64)
        first_training = current_train_view(image, 64, seed=17)
        second_training = current_train_view(image, 64, seed=17)

        self.assertEqual(evaluation.size, (64, 64))
        self.assertEqual(first_training.size, (64, 64))
        self.assertEqual(first_training.tobytes(), second_training.tobytes())

    def test_content_hash_matches_across_lossless_file_formats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = Image.new("RGB", (20, 18), "white")
            ImageDraw.Draw(image).ellipse((3, 2, 16, 15), fill=(90, 30, 120))
            png_path = root / "cell.png"
            bmp_path = root / "cell.bmp"
            image.save(png_path)
            image.save(bmp_path)

            png_record = analyze_image(png_path)
            bmp_record = analyze_image(bmp_path)

            self.assertNotEqual(png_record["file_sha256"], bmp_record["file_sha256"])
            self.assertEqual(
                png_record["content_sha256"], bmp_record["content_sha256"]
            )


class FullImageAuditTests(unittest.TestCase):
    def _write_manifest(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_run_audit_writes_reports_montages_and_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            csv_directory = root / "csv_files"
            output_directory = root / "audit"
            image_directory.mkdir()
            csv_directory.mkdir()

            shared_pixels = Image.new("RGB", (48, 36), (240, 240, 240))
            ImageDraw.Draw(shared_pixels).ellipse(
                (8, 4, 40, 32), fill=(110, 40, 130)
            )
            train_image = image_directory / "0_111_80.png"
            dev_duplicate = image_directory / "200_300_400_500.png"
            shared_pixels.save(train_image)
            shared_pixels.save(dev_duplicate)

            class_images = {
                "LSIL": image_directory / "1_2_3_4_5.png",
                "ASC-H": image_directory / "caseposition_1_2index_0.png",
                "HSIL": image_directory / "source_HSILposition_3_4index_0.png",
            }
            for index, path in enumerate(class_images.values(), start=1):
                image = Image.new("RGB", (40 + index, 30 + index), "white")
                ImageDraw.Draw(image).rectangle(
                    (5, 5, 25, 25), fill=(80 + index, 30, 120)
                )
                image.save(path)

            base_columns = {
                "maturity_label": 2,
                "maturity_name": "Parabasal",
            }
            self._write_manifest(
                csv_directory / "train_xudata_tbs5.csv",
                [
                    {
                        "image_path": train_image,
                        "split": "train",
                        "diagnosis_label": 0,
                        "diagnosis_name": "Normal",
                        **base_columns,
                    }
                ],
            )
            self._write_manifest(
                csv_directory / "dev_xudata_tbs5.csv",
                [
                    {
                        "image_path": dev_duplicate,
                        "split": "dev",
                        "diagnosis_label": 1,
                        "diagnosis_name": "ASC-US",
                        **base_columns,
                    }
                ],
            )
            self._write_manifest(
                csv_directory / "calibration_xudata_tbs5.csv",
                [
                    {
                        "image_path": class_images["LSIL"],
                        "split": "calibration",
                        "diagnosis_label": 2,
                        "diagnosis_name": "LSIL",
                        **base_columns,
                    }
                ],
            )
            self._write_manifest(
                csv_directory / "test_xudata_tbs5.csv",
                [
                    {
                        "image_path": class_images["ASC-H"],
                        "split": "test",
                        "diagnosis_label": 3,
                        "diagnosis_name": "ASC-H",
                        **base_columns,
                    },
                    {
                        "image_path": class_images["HSIL"],
                        "split": "test",
                        "diagnosis_label": 4,
                        "diagnosis_name": "HSIL",
                        **base_columns,
                    },
                ],
            )

            report = run_audit(
                csv_directory,
                output_directory,
                img_size=64,
                samples_per_class=2,
                seed=7,
            )

            self.assertEqual(report["total_images"], 5)
            self.assertEqual(report["development_images"], 2)
            self.assertEqual(report["unreadable_images"], 0)
            self.assertGreaterEqual(report["cross_split_exact_duplicate_groups"], 1)
            expected_files = (
                "image_inventory.csv",
                "class_image_summary.csv",
                "maturity_image_summary.csv",
                "filename_pattern_summary.csv",
                "exact_duplicate_groups.csv",
                "dhash_duplicate_candidates.csv",
                "embedded_label_mismatches.csv",
                "audit_report.json",
                "raw_montage_Normal.jpg",
                "transform_comparison_Normal.jpg",
            )
            for filename in expected_files:
                self.assertTrue((output_directory / filename).is_file(), filename)

            saved_report = json.loads(
                (output_directory / "audit_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report["total_images"], 5)
            summary = pd.read_csv(output_directory / "class_image_summary.csv")
            self.assertEqual(set(summary["diagnosis_name"]), {"Normal", "ASC-US"})


if __name__ == "__main__":
    unittest.main()
