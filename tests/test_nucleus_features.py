"""Tests for nucleus feature extraction (pure logic, no PyTorch)."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_features import (
    compute_boundary_gradient,
    extract_from_manifest,
    feature_summary,
    nucleus_features,
    rgb_features,
)


class TestNucleusFeatures(unittest.TestCase):
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        feats = nucleus_features(mask)
        self.assertFalse(feats["nonempty"])
        self.assertEqual(feats["nucleus_area_px"], 0)
        self.assertTrue(math.isnan(feats["equivalent_diameter"]))
        self.assertEqual(feats["component_count"], 0)

    def test_solid_circle(self):
        """Synthetic circle: radius=20, centre=(50,50)."""
        yy, xx = np.ogrid[:100, :100]
        mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 20**2
        feats = nucleus_features(mask)
        self.assertTrue(feats["nonempty"])
        self.assertAlmostEqual(feats["nucleus_area_px"], math.pi * 400, delta=30)
        self.assertAlmostEqual(feats["equivalent_diameter"], 40.0, delta=2)
        # Perimeter on a discrete grid overestimates ideal 2*pi*r,
        # so circularity for a discrete circle is ~0.6-0.85 rather than 1.0.
        self.assertGreater(feats["circularity"], 0.5)
        self.assertAlmostEqual(feats["eccentricity"], 0.0, delta=0.1)
        self.assertAlmostEqual(feats["centroid_x"], 50.0, delta=2)
        self.assertAlmostEqual(feats["centroid_y"], 50.0, delta=2)
        self.assertFalse(feats["edge_contact"])

    def test_rectangle_has_edge_contact(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:30, 0:30] = True  # touches top-left edge
        feats = nucleus_features(mask)
        self.assertTrue(feats["edge_contact"])

    def test_solidity_bounded(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True
        feats = nucleus_features(mask)
        self.assertGreater(feats["solidity"], 0.9)  # solid square

    def test_irregular_shape_low_circularity(self):
        """A thin horizontal line has low circularity."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 20:80] = True
        feats = nucleus_features(mask)
        self.assertLess(feats["circularity"], 0.5)

    def test_multi_component(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:30, 20:30] = True
        mask[60:70, 60:70] = True
        feats = nucleus_features(mask)
        self.assertEqual(feats["component_count"], 2)
        self.assertGreater(feats["largest_component_area_px"], 0)

    def test_non_2d_raises(self):
        with self.assertRaises(ValueError):
            nucleus_features(np.zeros((10, 10, 3), dtype=bool))


class TestBoundaryGradient(unittest.TestCase):
    def test_uniform_rgb(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        grad = compute_boundary_gradient(rgb, mask)
        self.assertAlmostEqual(grad, 0.0, delta=2.0)

    def test_empty_mask(self):
        rgb = np.random.RandomState(0).randint(0, 255, (50, 50, 3)).astype(np.uint8)
        self.assertTrue(math.isnan(compute_boundary_gradient(rgb, np.zeros((50, 50), dtype=bool))))


class TestRGBFeatures(unittest.TestCase):
    def test_returns_six_fields(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True
        rgb = np.full((50, 50, 3), 100, dtype=np.uint8)
        feats = rgb_features(rgb, mask)
        self.assertIn("nucleus_r_mean", feats)
        self.assertIn("nucleus_r_std", feats)
        self.assertIn("nucleus_g_mean", feats)
        self.assertIn("nucleus_g_std", feats)
        self.assertIn("nucleus_b_mean", feats)
        self.assertIn("nucleus_b_std", feats)
        self.assertAlmostEqual(feats["nucleus_r_mean"], 100.0)
        self.assertAlmostEqual(feats["nucleus_r_std"], 0.0)

    def test_empty_mask_returns_nan(self):
        rgb = np.full((50, 50, 3), 128, dtype=np.uint8)
        feats = rgb_features(rgb, np.zeros((50, 50), dtype=bool))
        self.assertTrue(math.isnan(feats["nucleus_r_mean"]))


class TestExtractFromManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_basic_extraction(self):
        # Create a manifest with synthetic images
        csv_path = Path(self.tmp) / "manifest.csv"
        mask_dir = Path(self.tmp) / "masks"
        mask_dir.mkdir()
        img_dir = Path(self.tmp) / "images"
        img_dir.mkdir()

        rows = []
        for i in range(5):
            sid = f"cell_{i}"
            # Create synthetic RGB image
            rgb = np.random.RandomState(i).randint(0, 255, (50, 50, 3)).astype(np.uint8)
            img_path = img_dir / f"{sid}.jpg"
            Image.fromarray(rgb).save(img_path)
            # Create synthetic mask
            mask = np.zeros((50, 50), dtype=bool)
            mask[10:40, 10:40] = True
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_dir / f"{sid}.png")
            rows.append(
                {
                    "image_path": str(img_path),
                    "diagnosis_name": "Normal",
                    "diagnosis_label": 0,
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        df = extract_from_manifest(csv_path, mask_dir)
        self.assertEqual(len(df), 5)
        self.assertEqual(df["nonempty"].sum(), 5)
        self.assertIn("nucleus_area_px", df.columns)
        self.assertIn("boundary_gradient_mean", df.columns)

    def test_missing_mask_sets_nonempty_false(self):
        csv_path = Path(self.tmp) / "manifest_missing.csv"
        mask_dir = Path(self.tmp) / "masks_missing"
        mask_dir.mkdir()
        img_dir = Path(self.tmp) / "images_missing"
        img_dir.mkdir()

        rgb = np.zeros((30, 30, 3), dtype=np.uint8)
        img_path = img_dir / "orphan.jpg"
        Image.fromarray(rgb).save(img_path)
        pd.DataFrame(
            [{"image_path": str(img_path), "diagnosis_name": "ASC-US", "diagnosis_label": 1}]
        ).to_csv(csv_path, index=False)

        df = extract_from_manifest(csv_path, mask_dir)
        self.assertEqual(len(df), 1)
        self.assertFalse(df.iloc[0]["nonempty"])


class TestFeatureSummary(unittest.TestCase):
    def test_summary_structure(self):
        rows = []
        for class_name in ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]:
            for i in range(10):
                rows.append(
                    {
                        "diagnosis_name": class_name,
                        "diagnosis_label": 0,
                        "nonempty": True,
                        "nucleus_area_px": 500.0 + 10 * i,
                        "nucleus_area_frac": 0.05,
                        "nucleus_perimeter": 80.0,
                        "equivalent_diameter": 25.0,
                        "solidity": 0.9,
                        "eccentricity": 0.3,
                        "circularity": 0.8,
                        "component_count": 1,
                        "largest_component_area_px": 500,
                        "boundary_gradient_mean": 12.0,
                        "nucleus_r_mean": 100.0,
                        "nucleus_g_mean": 80.0,
                        "nucleus_b_mean": 60.0,
                    }
                )
        df = pd.DataFrame(rows)
        summary = feature_summary(df)
        self.assertIn("global_summary", summary)
        self.assertIn("per_class_summary", summary)
        self.assertEqual(len(summary["per_class_summary"]), 5)
        self.assertIn("nucleus_area_px", summary["global_summary"])
        self.assertAlmostEqual(summary["global_summary"]["nucleus_area_px"]["mean"], 545.0)
