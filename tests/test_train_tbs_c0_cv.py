import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.c0_protocol import LOCKED_C0_CONFIG, validate_complete_c0_cv_artifacts
from experiments.train_tbs_c0_cv import parse_args


def _history(fold):
    rows = []
    for epoch in range(1, LOCKED_C0_CONFIG["epochs"] + 1):
        rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "macro_f1": 0.5,
                "abnormal_macro_f1": 0.5,
                "low_grade_pair_macro_f1": 0.5,
                "high_grade_pair_macro_f1": 0.5,
                "screen_sensitivity": 1.0,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.0,
                "full_macro_f1": 0.5,
                "local_macro_f1": 0.2,
                "view_gate_mean": 0.5,
                "local_area_mean": 0.36,
                "local_translation_abs_mean": 0.0,
                "local_cosine_similarity_mean": 0.8,
            }
        )
    return rows


class C0TrainingCliTests(unittest.TestCase):
    def test_cli_has_no_dev_or_unlocked_training_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--train_fit_csv", str(root / "fit.csv"),
                    "--select_s0_csv", str(root / "s0.csv"),
                    "--select_s1_csv", str(root / "s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "c0"),
                    "--device", "cpu",
                    "--num_workers", "0",
                ],
                validate_paths=False,
            )
            self.assertEqual(args.device, "cpu")
            for forbidden in ("dev_csv", "epochs", "checkpoint", "calibration", "test_csv"):
                self.assertFalse(hasattr(args, forbidden))
            for forbidden_arg in (
                "--dev_csv", "--epochs", "--checkpoint", "--lambda_geometry",
                "--calibration", "--test_csv",
            ):
                with self.assertRaises(SystemExit):
                    parse_args(
                        [
                            "--train_fit_csv", str(root / "fit.csv"),
                            "--select_s0_csv", str(root / "s0.csv"),
                            "--select_s1_csv", str(root / "s1.csv"),
                            "--fold_dir", str(root / "folds"),
                            "--s0r_cv_dir", str(root / "s0r"),
                            "--out_dir", str(root / "c0"),
                            forbidden_arg, "value",
                        ],
                        validate_paths=False,
                    )

    def test_cli_rejects_dev_like_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--train_fit_csv", str(root / "development" / "fit.csv"),
                        "--select_s0_csv", str(root / "s0.csv"),
                        "--select_s1_csv", str(root / "s1.csv"),
                        "--fold_dir", str(root / "folds"),
                        "--s0r_cv_dir", str(root / "s0r"),
                        "--out_dir", str(root / "c0"),
                    ]
                )

    def test_complete_artifacts_require_all_folds_and_reject_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in range(LOCKED_C0_CONFIG["fold_count"]):
                fold_dir = root / f"fold_{fold}"
                fold_dir.mkdir(parents=True)
                pd.DataFrame(_history(fold)).to_csv(fold_dir / "metrics.csv", index=False)
            result = validate_complete_c0_cv_artifacts(root)
            self.assertEqual(len(result["history_sha256"]), 5)
            (root / "model.pth").write_bytes(b"forbidden")
            with self.assertRaises(ValueError):
                validate_complete_c0_cv_artifacts(root)


if __name__ == "__main__":
    unittest.main()
