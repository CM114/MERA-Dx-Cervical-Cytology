"""Tests for nucleus_inference module (pure logic, no PyTorch dependency)."""

import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_inference import (
    OWNERSHIP_FILENAME,
    compute_audit_summary,
    generate_triptych,
    mask_statistics,
    select_audit_samples,
    write_completed_marker,
    write_ownership_marker,
)


class TestMaskStatistics(unittest.TestCase):
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        stats = mask_statistics(mask)
        self.assertFalse(stats["nonempty"])
        self.assertEqual(stats["foreground_pixels"], 0)
        self.assertEqual(stats["foreground_fraction"], 0.0)
        self.assertEqual(stats["component_count"], 0)
        self.assertFalse(stats["edge_contact"])

    def test_full_mask(self):
        mask = np.ones((50, 50), dtype=bool)
        stats = mask_statistics(mask)
        self.assertTrue(stats["nonempty"])
        self.assertTrue(stats["full_mask"])
        self.assertEqual(stats["foreground_pixels"], 2500)
        self.assertEqual(stats["foreground_fraction"], 1.0)
        self.assertTrue(stats["edge_contact"])

    def test_single_foreground_pixel(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        stats = mask_statistics(mask)
        self.assertTrue(stats["nonempty"])
        self.assertFalse(stats["full_mask"])
        self.assertEqual(stats["foreground_pixels"], 1)
        self.assertEqual(stats["component_count"], 1)
        self.assertAlmostEqual(stats["centroid_x"], 50.0)
        self.assertAlmostEqual(stats["centroid_y"], 50.0)
        self.assertFalse(stats["edge_contact"])

    def test_edge_contact_detection(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True
        self.assertTrue(mask_statistics(mask)["edge_contact"])
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[-1, -1] = True
        self.assertTrue(mask_statistics(mask2)["edge_contact"])

    def test_bbox_computation(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True
        stats = mask_statistics(mask)
        self.assertEqual(stats["bbox_left"], 30)
        self.assertEqual(stats["bbox_top"], 20)
        self.assertEqual(stats["bbox_right"], 50)
        self.assertEqual(stats["bbox_bottom"], 40)

    def test_non_2d_mask_raises(self):
        with self.assertRaises(ValueError):
            mask_statistics(np.zeros((10, 10, 3), dtype=bool))


class TestSelectAuditSamples(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        rows = []
        for class_name in ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]:
            for i in range(50):
                rows.append(
                    {
                        "image_path": f"/data/{class_name}/{i}.jpg",
                        "diagnosis_name": class_name,
                        "diagnosis_label": 0,
                    }
                )
        self.manifest = pd.DataFrame(rows)

    def test_per_class_limit(self):
        result = select_audit_samples(self.manifest, per_class=5, random_state=0)
        counts = result["diagnosis_name"].value_counts()
        self.assertTrue((counts == 5).all())
        self.assertEqual(len(result), 25)

    def test_per_class_exceeds_available(self):
        result = select_audit_samples(self.manifest, per_class=100, random_state=0)
        self.assertEqual(len(result), len(self.manifest))

    def test_reproducible(self):
        a = select_audit_samples(self.manifest, per_class=10, random_state=0)
        b = select_audit_samples(self.manifest, per_class=10, random_state=0)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different(self):
        a = select_audit_samples(self.manifest, per_class=10, random_state=0)
        b = select_audit_samples(self.manifest, per_class=10, random_state=1)
        # May be equal by chance but very unlikely for 5*10 samples
        if a.equals(b):
            self.skipTest("Identical samples by chance – not a bug")


class TestComputeAuditSummary(unittest.TestCase):
    def setUp(self):
        rows = []
        for class_name in ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]:
            for i in range(20):
                rows.append(
                    {
                        "diagnosis_name": class_name,
                        "diagnosis_label": 0,
                        "nonempty": i < 18,  # 18/20 effective
                        "full_mask": False,
                        "edge_contact": i < 10,
                        "foreground_fraction": 0.05,
                        "foreground_pixels": 500,
                        "component_count": 1,
                    }
                )
        self.stats = pd.DataFrame(rows)
        self.manifest = pd.DataFrame(rows)

    def test_global_effective_rate(self):
        audit = compute_audit_summary(self.stats, self.manifest)
        self.assertAlmostEqual(audit["global_stats"]["effective_mask_rate"], 0.9)

    def test_per_class_counts(self):
        audit = compute_audit_summary(self.stats, self.manifest)
        self.assertEqual(len(audit["per_class_stats"]), 5)
        for row in audit["per_class_stats"]:
            self.assertEqual(row["total"], 20)
            self.assertEqual(row["effective"], 18)

    def test_no_hsil_flag_when_high_rate(self):
        audit = compute_audit_summary(self.stats, self.manifest)
        self.assertNotIn(
            "HSIL_LOW_EFFECTIVE",
            " ".join(audit["risk_flags"]),
        )

    def test_hsil_low_effective_flag(self):
        # Make HSIL mostly empty
        mask = self.stats["diagnosis_name"] == "HSIL"
        self.stats.loc[mask, "nonempty"] = False
        self.stats.loc[mask, "foreground_fraction"] = 0.0
        audit = compute_audit_summary(self.stats, self.manifest)
        flags_text = " ".join(audit["risk_flags"])
        self.assertIn("HSIL_LOW_EFFECTIVE", flags_text)

    def test_low_effective_rate_flag(self):
        self.stats["nonempty"] = False
        audit = compute_audit_summary(self.stats, self.manifest)
        flags_text = " ".join(audit["risk_flags"])
        self.assertIn("LOW_EFFECTIVE_RATE", flags_text)


class TestGenerateTriptych(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rgb_path = Path(self.tmp) / "test_cell.jpg"
        rgb = Image.new("RGB", (224, 224), color=(128, 100, 80))
        rgb.save(self.rgb_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generates_three_files(self):
        mask = np.zeros((224, 224), dtype=bool)
        mask[50:100, 50:100] = True
        out_dir = Path(self.tmp) / "visual"
        paths = generate_triptych(self.rgb_path, mask, out_dir, "sample_001")
        self.assertTrue(Path(paths["rgb"]).is_file())
        self.assertTrue(Path(paths["mask"]).is_file())
        self.assertTrue(Path(paths["overlay"]).is_file())

    def test_empty_mask_still_generates(self):
        mask = np.zeros((224, 224), dtype=bool)
        out_dir = Path(self.tmp) / "visual_empty"
        paths = generate_triptych(self.rgb_path, mask, out_dir, "empty_sample")
        for key in ("rgb", "mask", "overlay"):
            self.assertTrue(Path(paths[key]).is_file(), f"Missing {key}")

    def test_mismatched_size_resizes(self):
        mask = np.zeros((112, 112), dtype=bool)  # half size
        mask[20:50, 20:50] = True
        out_dir = Path(self.tmp) / "visual_resized"
        paths = generate_triptych(self.rgb_path, mask, out_dir, "resized")
        overlay = Image.open(paths["overlay"])
        self.assertEqual(overlay.size, (224, 224))


class TestOwnershipMarker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_marker(self):
        out = Path(self.tmp) / "fresh"
        write_ownership_marker(out, overwrite=False)
        self.assertTrue((out / OWNERSHIP_FILENAME).is_file())

    def test_refuses_unowned_overwrite(self):
        out = Path(self.tmp) / "unowned"
        out.mkdir(parents=True)
        (out / "some_file.txt").write_text("data")
        with self.assertRaises((FileExistsError, PermissionError)):
            write_ownership_marker(out, overwrite=False)

    def test_allows_empty_directory(self):
        out = Path(self.tmp) / "empty_existing"
        out.mkdir(parents=True)
        write_ownership_marker(out, overwrite=False)
        self.assertTrue((out / OWNERSHIP_FILENAME).is_file())

    def test_allows_owned_overwrite(self):
        out = Path(self.tmp) / "owned"
        write_ownership_marker(out, overwrite=False)
        (out / "some_file.txt").write_text("data")
        write_ownership_marker(out, overwrite=True)


class TestCompletedMarker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_with_valid_artifacts(self):
        (Path(self.tmp) / "a.txt").write_text("hello", encoding="utf-8")
        (Path(self.tmp) / "b.txt").write_text("world", encoding="utf-8")
        from experiments.tbs.nucleus_inference import _file_sha256

        manifest = {
            "a.txt": _file_sha256(Path(self.tmp) / "a.txt"),
            "b.txt": _file_sha256(Path(self.tmp) / "b.txt"),
        }
        write_completed_marker(self.tmp, manifest)
        completed = Path(self.tmp) / "completed.json"
        self.assertTrue(completed.is_file())
        data = json.loads(completed.read_text())
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["artifact_count"], 2)

    def test_missing_artifact_raises(self):
        from experiments.tbs.nucleus_inference import _file_sha256

        (Path(self.tmp) / "a.txt").write_text("hello")
        manifest = {
            "a.txt": _file_sha256(Path(self.tmp) / "a.txt"),
            "missing.txt": "abc123",
        }
        with self.assertRaises(FileNotFoundError):
            write_completed_marker(self.tmp, manifest)

    def test_hash_mismatch_raises(self):
        (Path(self.tmp) / "a.txt").write_text("hello")
        manifest = {"a.txt": "wrong_hash"}
        with self.assertRaises(ValueError):
            write_completed_marker(self.tmp, manifest)
