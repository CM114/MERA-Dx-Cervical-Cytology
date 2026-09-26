import tempfile
import unittest
from pathlib import Path

from experiments.train_tbs_c2r2_boundary_risk_cv import parse_args


class C2R2RunnerCliTests(unittest.TestCase):
    def test_cli_accepts_locked_source_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--train_fit_csv", str(root / "train_fit.csv"),
                    "--select_s0_csv", str(root / "select_s0.csv"),
                    "--select_s1_csv", str(root / "select_s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "candidate"),
                ],
                validate_paths=False,
            )
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.num_workers, 8)


if __name__ == "__main__":
    unittest.main()
