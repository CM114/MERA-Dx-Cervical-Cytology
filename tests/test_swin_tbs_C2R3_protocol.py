import unittest

from experiments.C2R3.protocol import (
    C2R3_CONFIG,
    INNER_FOLD_COUNT,
    INNER_SELECTION_RULE,
    OUTER_PROMOTION_RULE,
    require_locked_epoch,
)


class C2R3ProtocolTests(unittest.TestCase):
    def test_locks_frozen_pair_adapter_route(self):
        self.assertEqual(INNER_FOLD_COUNT, 3)
        self.assertTrue(C2R3_CONFIG["freeze_c1"])
        self.assertTrue(C2R3_CONFIG["family_mass_preserving"])
        self.assertTrue(C2R3_CONFIG["prototype_guided_hard_mining"])
        self.assertTrue(C2R3_CONFIG["prototype_does_not_set_residual_sign"])
        self.assertTrue(C2R3_CONFIG["official_pair_metric_locked"])
        self.assertTrue(C2R3_CONFIG["inner_c1_teacher_required"])
        self.assertEqual(INNER_SELECTION_RULE["maximum_family_mass_error"], 1e-6)
        self.assertGreater(OUTER_PROMOTION_RULE["minimum_pair_mean_delta"], 0.0)

    def test_epoch_is_not_posthoc_selectable(self):
        require_locked_epoch(C2R3_CONFIG["epochs"], smoke=False)
        require_locked_epoch(C2R3_CONFIG["smoke_epochs"], smoke=True)
        with self.assertRaises(ValueError):
            require_locked_epoch(C2R3_CONFIG["epochs"] - 1, smoke=False)


if __name__ == "__main__":
    unittest.main()
