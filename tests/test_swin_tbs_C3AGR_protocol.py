import unittest

from experiments.C3AGR.protocol import C3AGR_CONFIG, C3AGR_FEATURE_NAMES


class C3AGRProtocolTests(unittest.TestCase):
    def test_protocol_isolated_and_locks_residual_fallback(self):
        self.assertEqual(C3AGR_CONFIG["stage"], "C3-AGR")
        self.assertEqual(C3AGR_CONFIG["feature_count"], 8)
        self.assertTrue(C3AGR_CONFIG["freeze_c1_c2"])
        self.assertTrue(C3AGR_CONFIG["fallback_to_r1"])
        self.assertEqual(len(C3AGR_FEATURE_NAMES), 8)
        self.assertNotIn("subprototype_entropy", C3AGR_FEATURE_NAMES)
        self.assertNotIn("nearest_distance", C3AGR_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
