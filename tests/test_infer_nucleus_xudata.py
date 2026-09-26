"""Tests for infer_nucleus_xudata CLI (argument parsing and path validation).

Full main() integration requires PyTorch and is tested on the server.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.infer_nucleus_xudata import parse_args


def _make_dummy_csv(path, columns=("image_path", "diagnosis_label", "diagnosis_name")):
    df = pd.DataFrame(
        {
            "image_path": [f"/data/img_{i}.jpg" for i in range(5)],
            "diagnosis_label": np.random.RandomState(0).randint(0, 5, 5),
            "diagnosis_name": np.random.RandomState(0).choice(
                ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"], 5
            ),
        }
    )
    df.to_csv(path, index=False)
    return path


class TestArgParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_minimal_args(self):
        train = _make_dummy_csv(Path(self.tmp) / "train.csv")
        dev = _make_dummy_csv(Path(self.tmp) / "dev.csv")
        args = parse_args(
            [
                "--checkpoint", "/fake/best_model.pth",
                "--train_csv", str(train),
                "--dev_csv", str(dev),
                "--out_dir", str(Path(self.tmp) / "out"),
            ]
        )
        self.assertEqual(args.architecture, "unet_resnet34")
        self.assertEqual(args.input_size, 384)
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.audit_samples, 20)
        self.assertEqual(args.device, "cuda:1")

    def test_custom_args(self):
        train = _make_dummy_csv(Path(self.tmp) / "train.csv")
        dev = _make_dummy_csv(Path(self.tmp) / "dev.csv")
        args = parse_args(
            [
                "--checkpoint", "/ckpt.pth",
                "--train_csv", str(train),
                "--dev_csv", str(dev),
                "--out_dir", "/out",
                "--architecture", "unet_resnet50",
                "--input_size", "512",
                "--threshold", "0.3",
                "--batch_size", "8",
                "--audit_samples", "10",
                "--device", "cuda:0",
            ]
        )
        self.assertEqual(args.architecture, "unet_resnet50")
        self.assertEqual(args.input_size, 512)
        self.assertEqual(args.threshold, 0.3)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.audit_samples, 10)
        self.assertEqual(args.device, "cuda:0")


class TestPathRejection(unittest.TestCase):
    """Verify that test/calibration paths are rejected during argument parsing
    even before PyTorch is loaded."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_train_path_with_test_rejected(self):
        _make_dummy_csv(Path(self.tmp) / "dev.csv")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--checkpoint", "/fake/best_model.pth",
                    "--train_csv", str(Path(self.tmp) / "test_data" / "train.csv"),
                    "--dev_csv", str(Path(self.tmp) / "dev.csv"),
                    "--out_dir", str(Path(self.tmp) / "out"),
                ]
            )

    def test_dev_path_with_calibration_rejected(self):
        _make_dummy_csv(Path(self.tmp) / "train.csv")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--checkpoint", "/fake/best_model.pth",
                    "--train_csv", str(Path(self.tmp) / "train.csv"),
                    "--dev_csv", str(Path(self.tmp) / "calibration_dev.csv"),
                    "--out_dir", str(Path(self.tmp) / "out"),
                ]
            )
