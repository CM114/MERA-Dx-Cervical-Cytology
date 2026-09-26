import tempfile
import unittest
from pathlib import Path

from experiments.train_tbs_c0r2_cv import parse_args


class C0R2TrainingCliTests(unittest.TestCase):
    def test_cli_exposes_no_training_or_sealed_split_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args([
                "--train_fit_csv", str(root / "fit.csv"),
                "--select_s0_csv", str(root / "s0.csv"),
                "--select_s1_csv", str(root / "s1.csv"),
                "--fold_dir", str(root / "folds"),
                "--s0r_cv_dir", str(root / "s0r"),
                "--out_dir", str(root / "c0r2"),
                "--device", "cpu",
            ], validate_paths=False)
            self.assertEqual(args.device, "cpu")
            for name in ("dev_csv", "epochs", "checkpoint", "calibration", "test_csv"):
                self.assertFalse(hasattr(args, name))


if __name__ == "__main__":
    unittest.main()
