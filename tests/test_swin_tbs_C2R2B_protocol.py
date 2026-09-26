import unittest

from experiments.C2R2B.protocol import LOCKED_C2R2B_CONFIG, SELECTION_RULE


class C2R2BProtocolTests(unittest.TestCase):
    def test_pair_directed_constants_are_locked(self):
        self.assertEqual(LOCKED_C2R2B_CONFIG["gate_logit_offset"], -2.64)
        self.assertEqual(LOCKED_C2R2B_CONFIG["pair_boundary_threshold"], .20)
        self.assertEqual(LOCKED_C2R2B_CONFIG["pair_easy_threshold"], .60)
        self.assertEqual(LOCKED_C2R2B_CONFIG["g_max"], .15)
        self.assertEqual(SELECTION_RULE["minimum_pair_mean_delta"], .002)
        self.assertEqual(SELECTION_RULE["boundary_easy_gate_ratio"], 2.0)


if __name__ == "__main__": unittest.main()
