import unittest

from experiments.tbs.c1_factorized_protocol import (
    C1_CV_COMPLETE_ROUTE,
    C1_CV_SCHEMA,
    LOCKED_C1_FACTORIZED_CONFIG,
)


class C1FactorizedProtocolTests(unittest.TestCase):
    def test_config_is_full_view_m0_primary_factorized_candidate(self):
        self.assertEqual(C1_CV_SCHEMA, "xudata-tbs-c1-factorized-cv-v1")
        self.assertEqual(
            C1_CV_COMPLETE_ROUTE,
            "C1_FACTORIZED_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
        )
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["objective"], "c1_tbs_factorized_full_view")
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["semantic_dim"], 128)
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["residual_logit_bound"], 0.10)
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["lambda_screen"], 0.20)
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["lambda_morph"], 0.30)
        self.assertEqual(LOCKED_C1_FACTORIZED_CONFIG["lambda_evidence"], 0.30)


if __name__ == "__main__":
    unittest.main()
