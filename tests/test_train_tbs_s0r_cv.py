import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from experiments.train_tbs_s0r_cv import (
    CV_OUTPUT_SCHEMA,
    add_locked_pair_metric_aliases,
    parse_args,
    validate_complete_cv_artifacts,
)


class S0RCVContractTests(unittest.TestCase):
    def test_locked_pair_metrics_are_computed_from_validation_predictions(self):
        labels = np.asarray([0, 1, 2, 3, 4] * 2)
        probabilities = np.eye(5)[labels] * 0.95 + 0.01
        metrics = add_locked_pair_metric_aliases(
            {"macro_f1": 1.0, "val_loss": 0.1},
            labels,
            probabilities,
        )
        self.assertEqual(metrics["low_grade_pair_macro_f1"], 1.0)
        self.assertEqual(metrics["high_grade_pair_macro_f1"], 1.0)
        self.assertEqual(metrics["val_loss"], 0.1)

    def test_cli_exposes_no_dev_or_hyperparameter_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = parse_args(
                [
                    "--train_fit_csv", str(root / "train_fit.csv"),
                    "--select_s0_csv", str(root / "select_s0.csv"),
                    "--select_s1_csv", str(root / "select_s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--out_dir", str(root / "out"),
                    "--device", "cpu",
                    "--num_workers", "0",
                ]
            )
            self.assertEqual(args.device, "cpu")
            self.assertFalse(hasattr(args, "dev_csv"))
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--train_fit_csv", str(root / "train_fit.csv"),
                        "--select_s0_csv", str(root / "select_s0.csv"),
                        "--select_s1_csv", str(root / "select_s1.csv"),
                        "--fold_dir", str(root / "folds"),
                        "--out_dir", str(root / "out2"),
                        "--epochs", "2",
                    ]
                )

    def test_existing_output_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            (root / "out").mkdir()
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--train_fit_csv", str(root / "train_fit.csv"),
                        "--select_s0_csv", str(root / "select_s0.csv"),
                        "--select_s1_csv", str(root / "select_s1.csv"),
                        "--fold_dir", str(root / "folds"),
                        "--out_dir", str(root / "out"),
                    ]
                )

    def test_complete_cv_requires_five_full_histories_and_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in range(5):
                fold_dir = root / f"fold_{fold}"
                fold_dir.mkdir(parents=True)
                pd.DataFrame({"epoch": range(1, 31), "macro_f1": [0.8] * 30}).to_csv(
                    fold_dir / "metrics.csv", index=False
                )
            summary = validate_complete_cv_artifacts(root)
            self.assertEqual(summary["schema_version"], CV_OUTPUT_SCHEMA)
            self.assertEqual(summary["fold_count"], 5)
            (root / "fold_2" / "best_model.pth").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_complete_cv_artifacts(root)


if __name__ == "__main__":
    unittest.main()
