import unittest

from experiments.C2R2.protocol import LOCKED_C2R2_CONFIG, SELECTION_RULE


class C2R2ProtocolTests(unittest.TestCase):
    def test_locks_protected_expert(self):
        self.assertEqual(LOCKED_C2R2_CONFIG["epochs"], 10)
        self.assertTrue(LOCKED_C2R2_CONFIG["freeze_c1"])
        self.assertEqual(LOCKED_C2R2_CONFIG["g_max"], .15)
        self.assertEqual(LOCKED_C2R2_CONFIG["gate_logit_offset"], -4.0)
        self.assertEqual(LOCKED_C2R2_CONFIG["rho"], .15)
        self.assertEqual(SELECTION_RULE["minimum_pair_mean_delta"], .002)
        self.assertEqual(SELECTION_RULE["maximum_screen_mass_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
