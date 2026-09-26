import unittest

from experiments.tbs.c0r2f_protocol import LOCKED_C0R2F_CONFIG


class C0R2FProtocolTests(unittest.TestCase):
    def test_only_objective_name_changes_from_c0r2(self):
        self.assertEqual(LOCKED_C0R2F_CONFIG["lambda_full_anchor"], 0.25)
        self.assertEqual(LOCKED_C0R2F_CONFIG["lambda_low_grade_pair"], 0.05)
        self.assertEqual(LOCKED_C0R2F_CONFIG["lambda_high_grade_mass"], 0.05)
        self.assertEqual(LOCKED_C0R2F_CONFIG["residual_logit_bound"], 0.10)
        self.assertEqual(LOCKED_C0R2F_CONFIG["objective"], "c0r2_full_branch_boundary_stability")


if __name__ == "__main__":
    unittest.main()
