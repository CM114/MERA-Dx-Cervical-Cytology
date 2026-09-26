import unittest


class SwinTbsC0ProtocolTests(unittest.TestCase):
    def test_c0_protocol_locks_dual_view_control(self):
        from experiments.C0.protocol import LOCKED_C0_CONFIG

        self.assertEqual(LOCKED_C0_CONFIG["stage"], "C0")
        self.assertEqual(LOCKED_C0_CONFIG["objective"], "dual_view_five_class_control")
        self.assertEqual(LOCKED_C0_CONFIG["prototype_heads"], 0)
        self.assertEqual(LOCKED_C0_CONFIG["risk_heads"], 0)

    def test_c0_r1_uses_zero_initialized_local_residual_and_global_anchor(self):
        from experiments.C0.protocol import LOCKED_C0_CONFIG

        self.assertEqual(LOCKED_C0_CONFIG["revision"], "R1")
        self.assertEqual(LOCKED_C0_CONFIG["local_adapter_init"], "zero")
        self.assertGreater(LOCKED_C0_CONFIG["lambda_global_anchor"], 0.0)


if __name__ == "__main__":
    unittest.main()
