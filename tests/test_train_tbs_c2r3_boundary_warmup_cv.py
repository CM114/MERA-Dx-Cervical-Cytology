import unittest

from experiments.train_tbs_c2r3_boundary_warmup_cv import parse_args


class C2R3BoundaryWarmupTrainingCliTests(unittest.TestCase):
    def test_cli_uses_sealed_source_arguments_only(self):
        args = parse_args(
            [
                "--train_fit_csv", "train.csv",
                "--select_s0_csv", "s0.csv",
                "--select_s1_csv", "s1.csv",
                "--fold_dir", "folds",
                "--s0r_cv_dir", "s0r",
                "--out_dir", "out",
            ],
            validate_paths=False,
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.num_workers, 8)


if __name__ == "__main__":
    unittest.main()
