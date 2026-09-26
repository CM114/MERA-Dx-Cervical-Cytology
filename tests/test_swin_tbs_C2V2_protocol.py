import unittest

from experiments.C2V2.protocol import C2V2_CONFIG, require_locked_epoch


class C2V2ProtocolTests(unittest.TestCase):
    def test_locked_route(self):
        self.assertTrue(C2V2_CONFIG["freeze_c1"])
        self.assertTrue(C2V2_CONFIG["family_mass_preserving"])
        self.assertTrue(C2V2_CONFIG["screen_mass_preserving"])
        self.assertTrue(C2V2_CONFIG["native_image_size_is_audit_only"])
        self.assertTrue(C2V2_CONFIG["directory_subtype_is_audit_only"])
        self.assertFalse(C2V2_CONFIG["subtype_used_as_model_input"])
        self.assertEqual(C2V2_CONFIG["retired_inner_fold"], 0)
        self.assertEqual(C2V2_CONFIG["smoke_inner_fold"], 1)
        self.assertEqual(C2V2_CONFIG["confirm_inner_fold"], 2)
        self.assertEqual(C2V2_CONFIG["analysis_grid_size"], 128)
        self.assertEqual(C2V2_CONFIG["morphology_full_audit_descriptor_dim"], 32)
        self.assertEqual(C2V2_CONFIG["morphology_descriptor_dim"], 21)
        self.assertEqual(C2V2_CONFIG["size_shortcut_risk_threshold"], 0.30)
        self.assertEqual(len(C2V2_CONFIG["size_robust_feature_names"]), 21)
        self.assertEqual(len(C2V2_CONFIG["size_excluded_feature_names"]), 11)
        self.assertTrue(C2V2_CONFIG["size_feature_mask_locked_before_inner_dev"])

    def test_epochs_are_locked(self):
        require_locked_epoch(6, smoke=True)
        require_locked_epoch(12, smoke=False)
        with self.assertRaises(ValueError):
            require_locked_epoch(8, smoke=True)
        with self.assertRaises(ValueError):
            require_locked_epoch(10, smoke=False)


if __name__ == "__main__":
    unittest.main()
