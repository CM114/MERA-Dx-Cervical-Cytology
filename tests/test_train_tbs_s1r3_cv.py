import tempfile
import unittest
from pathlib import Path

from experiments.tbs.s1r3_training import finalize_loss_metrics
from experiments.train_tbs_s1r3_cv import parse_args


class S1R3CVContractTests(unittest.TestCase):
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

    def test_cli_has_no_dev_epoch_or_loss_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = parse_args(self._base_args(root))
            self.assertFalse(hasattr(args, "dev_csv"))
            self.assertFalse(hasattr(args, "epochs"))
            self.assertFalse(hasattr(args, "lambda_high_grade_risk"))
            with self.assertRaises(SystemExit):
                parse_args(self._base_args(root, "out2") + ["--epochs", "2"])
            with self.assertRaises(SystemExit):
                parse_args(
                    self._base_args(root, "out3")
                    + ["--lambda_high_grade_risk", "1.0"]
                )

    def test_cli_rejects_dev_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("dev.csv", "select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = self._base_args(root)
            args[1] = str(root / "dev.csv")
            with self.assertRaises(SystemExit):
                parse_args(args)

    def test_cli_rejects_devset_and_development_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bad_name in (
                "devset.csv",
                "development.csv",
                "s1r3-devset.csv",
                "s1r3-development.csv",
            ):
                for name in (bad_name, "select_s0.csv", "select_s1.csv"):
                    (root / name).write_text("sealed\n", encoding="utf-8")
                args = self._base_args(root, output=f"out_{bad_name}")
                args[1] = str(root / bad_name)
                with self.assertRaises(SystemExit):
                    parse_args(args)

    def test_cli_rejects_source_symlink_to_devset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "hidden_devset"
            target_dir.mkdir()
            target = target_dir / "train.csv"
            target.write_text("sealed\n", encoding="utf-8")
            alias = root / "train_alias.csv"
            try:
                alias.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            for name in ("select_s0.csv", "select_s1.csv"):
                (root / name).write_text("sealed\n", encoding="utf-8")
            args = self._base_args(root)
            args[1] = str(alias)
            with self.assertRaises(SystemExit):
                parse_args(args)

    def test_risk_metric_uses_high_grade_count_not_batch_size(self):
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
            high_grade_count=2,
            sample_count=6,
        )
        self.assertAlmostEqual(result["high_grade_risk_loss"], 2.0)
        self.assertAlmostEqual(result["loss"], 1.5 + 0.2 * 2.0)


if __name__ == "__main__":
    unittest.main()
