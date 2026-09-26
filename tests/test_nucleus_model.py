"""Tests for dual-view model dataset (pure logic; no PyTorch/torchvision)."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_dataset import DualViewDataset, fit_scaler


FEATURE_COLUMNS = [
    "nucleus_area_px", "nucleus_area_frac", "nucleus_perimeter",
    "equivalent_diameter", "solidity", "eccentricity", "circularity",
    "component_count", "largest_component_area_px", "largest_component_area_frac",
    "edge_contact", "boundary_gradient_mean",
    "nucleus_r_mean", "nucleus_r_std", "nucleus_g_mean", "nucleus_g_std",
    "nucleus_b_mean", "nucleus_b_std",
]


def _mock_transform(img):
    """Return a (3,224,224) float32 numpy array (no torchvision needed)."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr


class TestFitScaler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fit_returns_standard_scaler(self):
        rows = []
        for i in range(100):
            rows.append(
                {
                    "sample_id": f"cell_{i}",
                    "nucleus_area_px": float(500 + i * 10),
                    "nucleus_area_frac": 0.05 + 0.001 * i,
                    "nucleus_perimeter": 80.0 + i * 0.5,
                    "equivalent_diameter": 25.0 + i * 0.2,
                    "solidity": 0.85 + 0.001 * i,
                    "eccentricity": 0.5,
                    "circularity": 0.7,
                    "component_count": 1.0,
                    "largest_component_area_px": 500.0,
                    "largest_component_area_frac": 0.05,
                    "edge_contact": 0.0,
                    "boundary_gradient_mean": 12.0,
                    "nucleus_r_mean": 100.0, "nucleus_r_std": 20.0,
                    "nucleus_g_mean": 80.0, "nucleus_g_std": 15.0,
                    "nucleus_b_mean": 60.0, "nucleus_b_std": 10.0,
                }
            )
        df = pd.DataFrame(rows)
        csv_path = Path(self.tmp) / "features.csv"
        df.to_csv(csv_path, index=False)
        scaler = fit_scaler(csv_path, FEATURE_COLUMNS)
        self.assertIsNotNone(scaler)

    def test_nan_handled(self):
        rows = [
            {
                "sample_id": "cell_0",
                "nucleus_area_px": 500.0,
                "nucleus_area_frac": math.nan,
                "nucleus_perimeter": 80.0,
                "equivalent_diameter": 25.0,
                "solidity": math.nan,
                "eccentricity": 0.5,
                "circularity": 0.7,
                "component_count": 1.0,
                "largest_component_area_px": 500.0,
                "largest_component_area_frac": 0.05,
                "edge_contact": 0.0,
                "boundary_gradient_mean": 12.0,
                "nucleus_r_mean": 100.0, "nucleus_r_std": 20.0,
                "nucleus_g_mean": 80.0, "nucleus_g_std": 15.0,
                "nucleus_b_mean": 60.0, "nucleus_b_std": 10.0,
            }
        ]
        csv_path = Path(self.tmp) / "features_nan.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        scaler = fit_scaler(csv_path, FEATURE_COLUMNS)
        self.assertIsNotNone(scaler)
        data = pd.read_csv(csv_path)
        X = data[FEATURE_COLUMNS].values.astype(np.float64)
        X[np.isnan(X)] = 0.0
        transformed = scaler.transform(X)
        self.assertFalse(np.isnan(transformed).any())


class TestDualViewDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_data(self, n=5):
        """Create synthetic manifest + features CSV + images (no torchvision)."""
        manifest_rows = []
        feature_rows = []
        img_dir = Path(self.tmp) / "images"
        img_dir.mkdir()
        for i in range(n):
            sid = f"cell_{i}"
            img_path = img_dir / f"{sid}.jpg"
            rgb = np.random.RandomState(i).randint(0, 255, (224, 224, 3)).astype(np.uint8)
            Image.fromarray(rgb).save(img_path)
            manifest_rows.append(
                {
                    "image_path": str(img_path),
                    "diagnosis_label": i % 5,
                    "diagnosis_name": "Test",
                }
            )
            feature_rows.append(
                {
                    "sample_id": sid,
                    "nucleus_area_px": 500.0 + i * 10,
                    "nucleus_area_frac": 0.05,
                    "nucleus_perimeter": 80.0,
                    "equivalent_diameter": 25.0,
                    "solidity": 0.9,
                    "eccentricity": 0.5,
                    "circularity": 0.7,
                    "component_count": 1.0,
                    "largest_component_area_px": 500.0,
                    "largest_component_area_frac": 0.05,
                    "edge_contact": 0.0,
                    "boundary_gradient_mean": 12.0,
                    "nucleus_r_mean": 100.0, "nucleus_r_std": 20.0,
                    "nucleus_g_mean": 80.0, "nucleus_g_std": 15.0,
                    "nucleus_b_mean": 60.0, "nucleus_b_std": 10.0,
                }
            )
        manifest_csv = Path(self.tmp) / "manifest.csv"
        features_csv = Path(self.tmp) / "features.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
        pd.DataFrame(feature_rows).to_csv(features_csv, index=False)
        return manifest_csv, features_csv

    def test_dataset_creation(self):
        manifest_csv, features_csv = self._make_data(5)
        ds = DualViewDataset(manifest_csv, features_csv, _mock_transform)
        self.assertEqual(len(ds), 5)

    def test_getitem_shapes(self):
        manifest_csv, features_csv = self._make_data(3)
        ds = DualViewDataset(manifest_csv, features_csv, _mock_transform)
        img, nucleus, label, sid = ds[0]
        # _mock_transform returns numpy, shape (3,224,224)
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertEqual(nucleus.shape, (18,))
        self.assertEqual(nucleus.dtype, np.float32)
        self.assertIsInstance(label, int)
        self.assertIsInstance(sid, str)

    def test_with_scaler(self):
        manifest_csv, features_csv = self._make_data(10)
        scaler = fit_scaler(features_csv, FEATURE_COLUMNS)
        ds = DualViewDataset(manifest_csv, features_csv, _mock_transform, scaler, FEATURE_COLUMNS)
        _, nucleus, _, _ = ds[0]
        self.assertEqual(nucleus.shape, (18,))
        self.assertTrue(abs(nucleus.mean()) < 5.0, f"mean={nucleus.mean():.2f}")

    def test_missing_features_raises(self):
        manifest_csv, features_csv = self._make_data(3)
        feats = pd.read_csv(features_csv)
        feats = feats[feats["sample_id"] != "cell_1"]
        feats.to_csv(features_csv, index=False)
        with self.assertRaises(ValueError):
            DualViewDataset(manifest_csv, features_csv, _mock_transform)

    def test_missing_feature_column_raises(self):
        manifest_csv, features_csv = self._make_data(3)
        with self.assertRaises(ValueError):
            DualViewDataset(manifest_csv, features_csv, _mock_transform,
                            feature_columns=["nonexistent_col"])
