import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.train_tbs_s1r2_cv import parse_args
from experiments.tbs.s1r2_protocol import validate_complete_s1r2_cv_artifacts


class S1R2CVContractTests(unittest.TestCase):
    def test_cli_exposes_no_dev_or_training_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = parse_args([
                "--train_fit_csv", str(root / "train_fit.csv"),
                "--select_s0_csv", str(root / "select_s0.csv"),
                "--select_s1_csv", str(root / "select_s1.csv"),
                "--fold_dir", str(root / "folds"),
                "--s0r_cv_dir", str(root / "s0r_cv"),
                "--out_dir", str(root / "out"),
                "--device", "cpu",
            ])
            self.assertFalse(hasattr(args, "dev_csv"))
            with self.assertRaises(SystemExit):
                parse_args([
                    "--train_fit_csv", str(root / "train_fit.csv"),
                    "--select_s0_csv", str(root / "select_s0.csv"),
                    "--select_s1_csv", str(root / "select_s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r_cv"),
                    "--out_dir", str(root / "out2"),
                    "--epochs", "2",
                ])

    def test_cli_rejects_dev_data_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("dev.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args([
                    "--train_fit_csv", str(root / "dev.csv"),
                    "--select_s0_csv", str(root / "select_s0.csv"),
                    "--select_s1_csv", str(root / "select_s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r_cv"),
                    "--out_dir", str(root / "out"),
                ])

    def test_complete_artifacts_require_five_histories_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "macro_f1": 0.8,
                "low_grade_pair_macro_f1": 0.8,
                "high_grade_pair_macro_f1": 0.8,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.8,
                "evidence_auroc": 0.7,
                "residual_adapter_weight_norm": 0.1,
                "residual_logit_abs_mean": 0.02,
            }
            for fold in range(5):
                folder = root / f"fold_{fold}"
                folder.mkdir()
                pd.DataFrame([{"fold": fold, "epoch": epoch, **row} for epoch in range(1, 31)]).to_csv(
                    folder / "metrics.csv", index=False
                )
            result = validate_complete_s1r2_cv_artifacts(root)
            self.assertEqual(result["schema_version"], "xudata-tbs-s1r2-cv-v1")
            (root / "fold_0" / "model.pth").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_complete_s1r2_cv_artifacts(root)

    def test_complete_artifacts_reject_wrong_fold_identity_and_ckpt_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "macro_f1": 0.8,
                "low_grade_pair_macro_f1": 0.8,
                "high_grade_pair_macro_f1": 0.8,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.8,
                "evidence_auroc": 0.7,
                "residual_adapter_weight_norm": 0.1,
                "residual_logit_abs_mean": 0.02,
            }
            for fold in range(5):
                folder = root / f"fold_{fold}"
                folder.mkdir()
                reported_fold = 0 if fold == 4 else fold
                pd.DataFrame(
                    [
                        {"fold": reported_fold, "epoch": epoch, **row}
                        for epoch in range(1, 31)
                    ]
                ).to_csv(folder / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "fold identity"):
                validate_complete_s1r2_cv_artifacts(root)

            path = root / "fold_4" / "metrics.csv"
            frame = pd.read_csv(path)
            frame["fold"] = 4
            frame.to_csv(path, index=False)
            (root / "fold_0" / "model.ckpt").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_complete_s1r2_cv_artifacts(root)

    def test_complete_artifacts_require_residual_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "macro_f1": 0.8,
                "low_grade_pair_macro_f1": 0.8,
                "high_grade_pair_macro_f1": 0.8,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.8,
                "evidence_auroc": 0.7,
            }
            for fold in range(5):
                folder = root / f"fold_{fold}"
                folder.mkdir()
                pd.DataFrame(
                    [{"fold": fold, "epoch": epoch, **row} for epoch in range(1, 31)]
                ).to_csv(folder / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "residual"):
                validate_complete_s1r2_cv_artifacts(root)


if __name__ == "__main__":
    unittest.main()
