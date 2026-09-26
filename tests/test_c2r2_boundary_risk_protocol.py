import unittest

from experiments.tbs.c2r2_boundary_risk_protocol import (
    C2R2_CV_COMPLETE_ROUTE,
    C2R2_CV_SCHEMA,
    LOCKED_C2R2_CONFIG,
)


class C2R2BoundaryRiskProtocolTests(unittest.TestCase):
    def test_schema_and_locked_protection_weights(self):
        self.assertEqual(C2R2_CV_SCHEMA, "xudata-tbs-c2r2-boundary-risk-cv-v1")
        self.assertEqual(
            C2R2_CV_COMPLETE_ROUTE,
            "C2R2_BOUNDARY_RISK_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
        )
        self.assertEqual(LOCKED_C2R2_CONFIG["objective"], "c2r2_tbs_dualprototype_boundary_risk_protection")
        self.assertEqual(LOCKED_C2R2_CONFIG["lambda_low_grade_protection"], 0.05)
        self.assertEqual(LOCKED_C2R2_CONFIG["lambda_high_grade_mass_protection"], 0.05)
        self.assertEqual(LOCKED_C2R2_CONFIG["prototype_context_scale"], 0.25)
        self.assertFalse("c1_checkpoint" in LOCKED_C2R2_CONFIG)
        self.assertFalse("c2_checkpoint" in LOCKED_C2R2_CONFIG)


if __name__ == "__main__":
    unittest.main()
