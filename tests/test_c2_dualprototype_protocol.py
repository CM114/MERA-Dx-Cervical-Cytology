import unittest

from experiments.tbs.c2_dualprototype_protocol import (
    C2_CV_COMPLETE_ROUTE,
    C2_CV_SCHEMA,
    LOCKED_C2_DUALPROTOTYPE_CONFIG,
)


class C2DualPrototypeProtocolTests(unittest.TestCase):
    def test_config_is_independent_dual_space_prototype_candidate(self):
        self.assertEqual(C2_CV_SCHEMA, "xudata-tbs-c2-dualprototype-cv-v1")
        self.assertEqual(
            C2_CV_COMPLETE_ROUTE,
            "C2_DUALPROTOTYPE_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
        )
        self.assertEqual(
            LOCKED_C2_DUALPROTOTYPE_CONFIG["objective"],
            "c2_tbs_dualprototype_full_view",
        )
        self.assertEqual(LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_prototype"], 0.10)
        self.assertEqual(LOCKED_C2_DUALPROTOTYPE_CONFIG["prototype_margin"], 0.05)
        self.assertEqual(LOCKED_C2_DUALPROTOTYPE_CONFIG["prototype_temperature"], 10.0)


if __name__ == "__main__":
    unittest.main()
