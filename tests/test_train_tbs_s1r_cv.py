import tempfile
import unittest
from pathlib import Path

from experiments.train_tbs_s1r_cv import parse_args


class S1RCVContractTests(unittest.TestCase):
    def test_cli_has_no_dev_or_epoch_override(self):
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
                    "--s0r_cv_dir", str(root / "s0r_cv"),
                    "--device", "cpu",
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
                        "--s0r_cv_dir", str(root / "s0r_cv"),
                        "--epochs", "2",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
