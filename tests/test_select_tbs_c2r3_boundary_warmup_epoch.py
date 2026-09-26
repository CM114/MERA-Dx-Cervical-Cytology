import unittest

from experiments.select_tbs_c2r3_boundary_warmup_epoch import parse_args


class C2R3BoundaryWarmupSelectorCliTests(unittest.TestCase):
    def test_cli_uses_warmup_candidate_name(self):
        args = parse_args(
            [
                "--c2r3w_cv_dir", "candidate",
                "--s0r_cv_dir", "baseline",
                "--out_dir", "selection",
            ],
            validate_paths=False,
        )
        self.assertEqual(str(args.c2r3w_cv_dir), "candidate")


if __name__ == "__main__":
    unittest.main()
