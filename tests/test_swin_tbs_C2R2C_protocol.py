import unittest

from experiments.C2R2C.protocol import LOCKED_C2R2C_CONFIG, SELECTION_RULE


class C2R2CProtocolTests(unittest.TestCase):
    def test_locks_family_preserving_core_boundary_route(self):
        self.assertTrue(LOCKED_C2R2C_CONFIG["family_mass_preserving"])
        self.assertTrue(LOCKED_C2R2C_CONFIG["core_weighted_prototype_anchor"])
        self.assertTrue(LOCKED_C2R2C_CONFIG["boundary_weighted_pair_loss"])
        self.assertEqual(LOCKED_C2R2C_CONFIG["rho"], .15)
        self.assertEqual(SELECTION_RULE["maximum_family_mass_error"], 1e-6)

    def test_gate_starts_active_but_remains_bounded(self):
        self.assertEqual(LOCKED_C2R2C_CONFIG["gate_logit_offset"], -2.64)
        self.assertLess(LOCKED_C2R2C_CONFIG["g_max"], 0.16)

    def test_uses_only_locked_core_boundary_losses(self):
        self.assertNotIn("lambda_direction", LOCKED_C2R2C_CONFIG)


if __name__=="__main__": unittest.main()
