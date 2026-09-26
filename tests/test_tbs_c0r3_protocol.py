import unittest

from experiments.tbs.c0r3_protocol import LOCKED_C0R3_CONFIG


class C0R3ProtocolTests(unittest.TestCase):
    def test_only_local_translation_bound_changes(self):
        self.assertEqual(LOCKED_C0R3_CONFIG["local_translation_bound"], 0.12)
        self.assertEqual(LOCKED_C0R3_CONFIG["lambda_full_anchor"], 0.25)
        self.assertEqual(LOCKED_C0R3_CONFIG["lambda_low_grade_pair"], 0.05)
        self.assertEqual(LOCKED_C0R3_CONFIG["lambda_high_grade_mass"], 0.05)


if __name__ == "__main__":
    unittest.main()
