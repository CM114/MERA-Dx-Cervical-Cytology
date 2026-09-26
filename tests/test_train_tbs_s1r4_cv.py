import tempfile
import unittest
from pathlib import Path

from experiments.tbs.s1r4_training import finalize_loss_metrics
from experiments.train_tbs_s1r4_cv import parse_args


class S1R4CVContractTests(unittest.TestCase):
    def _base_args(self, root, output="out"):
        return [
            "--train_fit_csv", str(root / "train_fit.csv"),
            "--select_s0_csv", str(root / "select_s0.csv"),
            "--select_s1_csv", str(root / "select_s1.csv"),
            "--fold_dir", str(root / "folds"),
            "--s0r_cv_dir", str(root / "s0r_cv"),
            "--out_dir", str(root / output),
            "--device", "cpu",
        ]

    def test_cli_has_no_data_or_training_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = parse_args(self._base_args(root))
            for name in (
                "dev_csv", "epochs", "lambda_high_grade_risk",
                "lambda_high_grade_pair_boundary", "checkpoint_dir",
                "calibration_csv", "test_csv",
            ):
                self.assertFalse(hasattr(args, name))
            for option in (
                ["--epochs", "2"],
                ["--lambda_high_grade_risk", "1.0"],
                ["--lambda_high_grade_pair_boundary", "1.0"],
                ["--checkpoint_dir", str(root / "ckpt")],
                ["--dev_csv", str(root / "dev.csv")],
            ):
                with self.subTest(option=option):
                    with self.assertRaises(SystemExit):
                        parse_args(self._base_args(root, "out2") + option)

    def test_cli_rejects_dev_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("dev.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = self._base_args(root)
            args[1] = str(root / "dev.csv")
            with self.assertRaises(SystemExit):
                parse_args(args)

    def test_high_grade_metrics_use_high_grade_count(self):
        result = finalize_loss_metrics(
            component_totals={
                "diagnosis_loss": 6.0,
                "screen_loss": 3.0,
                "morph_loss": 3.0,
                "evidence_loss": 3.0,
                "decorr_loss": 0.6,
            },
            base_loss_total=9.0,
            risk_loss_total=4.0,
            boundary_loss_total=6.0,
            high_grade_count=2,
            sample_count=6,
        )
        self.assertAlmostEqual(result["high_grade_risk_loss"], 2.0)
        self.assertAlmostEqual(result["high_grade_pair_boundary_loss"], 3.0)
        self.assertAlmostEqual(result["loss"], 1.5 + 0.2 * 2.0 + 0.2 * 3.0)


if __name__ == "__main__":
    unittest.main()
