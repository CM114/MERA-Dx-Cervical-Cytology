import unittest

from experiments.C2V1.protocol import C2V1_CONFIG, require_locked_epoch


class C2V1ProtocolTests(unittest.TestCase):
    def test_locked_route(self):
        self.assertTrue(C2V1_CONFIG["freeze_c1"])
        self.assertTrue(C2V1_CONFIG["family_mass_preserving"])
        self.assertTrue(C2V1_CONFIG["screen_mass_preserving"])
        self.assertEqual(C2V1_CONFIG["retired_inner_fold"], 0)
        self.assertEqual(C2V1_CONFIG["smoke_inner_fold"], 1)
        self.assertEqual(C2V1_CONFIG["confirm_inner_fold"], 2)

    def test_epochs_are_locked(self):
        require_locked_epoch(6, smoke=True)
        require_locked_epoch(12, smoke=False)
        with self.assertRaises(ValueError):
            require_locked_epoch(8, smoke=True)
        with self.assertRaises(ValueError):
            require_locked_epoch(10, smoke=False)


if __name__ == "__main__":
    unittest.main()
