import unittest

from experiments.tbs.c2r3_conditional_boundary_protocol import (
    C2R3_CV_COMPLETE_ROUTE,
    C2R3_CV_SCHEMA,
    LOCKED_C2R3_CONFIG,
)


class C2R3ConditionalBoundaryProtocolTests(unittest.TestCase):
    def test_config_is_independent_and_uses_group_boundary_terms(self):
        self.assertEqual(C2R3_CV_SCHEMA, "xudata-tbs-c2r3-conditional-boundary-cv-v1")
        self.assertEqual(
            C2R3_CV_COMPLETE_ROUTE,
            "C2R3_CONDITIONAL_BOUNDARY_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
        )
        self.assertEqual(
            LOCKED_C2R3_CONFIG["objective"],
            "c2r3_tbs_dualprototype_conditional_boundary",
        )
        self.assertEqual(LOCKED_C2R3_CONFIG["lambda_low_grade_pair_ce"], 0.02)
        self.assertEqual(LOCKED_C2R3_CONFIG["lambda_high_grade_pair_ce"], 0.02)
        self.assertEqual(LOCKED_C2R3_CONFIG["lambda_high_grade_boundary"], 0.02)
        self.assertEqual(LOCKED_C2R3_CONFIG["high_grade_boundary_margin"], 0.10)
        self.assertEqual(LOCKED_C2R3_CONFIG["prototype_context_scale"], 0.25)
        self.assertFalse("c1_checkpoint" in LOCKED_C2R3_CONFIG)
        self.assertFalse("c2_checkpoint" in LOCKED_C2R3_CONFIG)
        self.assertFalse("c2r2_checkpoint" in LOCKED_C2R3_CONFIG)


if __name__ == "__main__":
    unittest.main()
