import tempfile
import unittest
from pathlib import Path

from experiments.audit_tbs_c2r3_hard_boundary_smoke import (
    _boundary_loss_scale,
    parse_args,
)


class C2R3HardBoundarySmokeCliTests(unittest.TestCase):
    def test_cli_defaults_to_one_fold_six_epochs_and_boundary_only_focus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--train_fit_csv", str(root / "train_fit.csv"),
                    "--select_s0_csv", str(root / "select_s0.csv"),
                    "--select_s1_csv", str(root / "select_s1.csv"),
                    "--fold_dir", str(root / "folds"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "smoke"),
                ],
                validate_paths=False,
            )
            self.assertEqual(args.fold, 0)
            self.assertEqual(args.epochs, 6)
            self.assertEqual(args.hard_example_fraction, 1.0)
            self.assertEqual(args.boundary_hard_example_fraction, 0.8)
            self.assertEqual(args.boundary_warmup_epochs, 0)
            self.assertEqual(args.device, "cuda:0")

    def test_boundary_warmup_is_linear_and_caps_at_one(self):
        self.assertEqual(_boundary_loss_scale(1, 4), 0.25)
        self.assertEqual(_boundary_loss_scale(3, 4), 0.75)
        self.assertEqual(_boundary_loss_scale(4, 4), 1.0)
        self.assertEqual(_boundary_loss_scale(6, 4), 1.0)
        self.assertEqual(_boundary_loss_scale(1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
