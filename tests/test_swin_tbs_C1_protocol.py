import unittest


class SwinTbsC1ProtocolTests(unittest.TestCase):
    def test_c1_protocol_requires_factor_metrics_and_no_checkpoints(self):
        from experiments.C1.protocol import LOCKED_C1_CONFIG, REQUIRED_C1_METRICS

        self.assertEqual(LOCKED_C1_CONFIG["stage"], "C1")
        self.assertIn("morph_accuracy", REQUIRED_C1_METRICS)
        self.assertFalse(LOCKED_C1_CONFIG["checkpoints_written"])


if __name__ == "__main__":
    unittest.main()
