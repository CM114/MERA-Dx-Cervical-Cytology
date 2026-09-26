import tempfile
import unittest
from pathlib import Path

from experiments.train_tbs_c2r1_conditioned_cv import parse_args


class C2R1TrainingCliTests(unittest.TestCase):
    def test_cli_has_no_checkpoint_or_unlocked_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--train_fit_csv", str(root / "fit.csv"),
                    "--select_s0_csv", str(root / "s0.csv"),
                    "--select_s1_csv", str(root / "s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "c2r1"),
                    "--device", "cpu",
                ],
                validate_paths=False,
            )
            self.assertEqual(args.device, "cpu")
            self.assertFalse(hasattr(args, "checkpoint"))
            self.assertFalse(hasattr(args, "c1_cv_dir"))
            self.assertFalse(hasattr(args, "c2_cv_dir"))


if __name__ == "__main__":
    unittest.main()
