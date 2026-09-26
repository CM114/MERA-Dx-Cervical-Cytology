import unittest

from experiments.tbs.c2r1_conditioned_protocol import (
    C2R1_CV_COMPLETE_ROUTE,
    C2R1_CV_SCHEMA,
    LOCKED_C2R1_CONFIG,
)


class C2R1ConditionedProtocolTests(unittest.TestCase):
    def test_config_is_independent_and_more_conservative(self):
        self.assertEqual(C2R1_CV_SCHEMA, "xudata-tbs-c2r1-conditioned-cv-v1")
        self.assertEqual(
            C2R1_CV_COMPLETE_ROUTE,
            "C2R1_CONDITIONED_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
        )
        self.assertEqual(
            LOCKED_C2R1_CONFIG["objective"],
            "c2r1_tbs_dualprototype_conditioned_residual_full_view",
        )
        self.assertEqual(LOCKED_C2R1_CONFIG["prototype_context_scale"], 0.25)
        self.assertEqual(LOCKED_C2R1_CONFIG["lambda_base_anchor"], 0.50)
        self.assertEqual(LOCKED_C2R1_CONFIG["lambda_prototype"], 0.05)
        self.assertEqual(LOCKED_C2R1_CONFIG["lambda_residual"], 0.02)
        self.assertFalse("c1_checkpoint" in LOCKED_C2R1_CONFIG)
        self.assertFalse("c2_checkpoint" in LOCKED_C2R1_CONFIG)


if __name__ == "__main__":
    unittest.main()
