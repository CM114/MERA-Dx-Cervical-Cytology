import unittest

from experiments.tbs.c2r3_boundary_warmup_protocol import (
    C2R3W_CV_SCHEMA,
    LOCKED_C2R3W_CONFIG,
    boundary_loss_scale,
)


class C2R3BoundaryWarmupProtocolTests(unittest.TestCase):
    def test_config_locks_warmup_and_preserves_pair_full_sample(self):
        self.assertEqual(C2R3W_CV_SCHEMA, "xudata-tbs-c2r3w-boundary-warmup-cv-v1")
        self.assertEqual(LOCKED_C2R3W_CONFIG["hard_example_fraction"], 1.0)
        self.assertEqual(LOCKED_C2R3W_CONFIG["boundary_hard_example_fraction"], 0.8)
        self.assertEqual(LOCKED_C2R3W_CONFIG["boundary_warmup_epochs"], 4)

    def test_boundary_scale_is_locked_linear_schedule(self):
        self.assertEqual([boundary_loss_scale(i, 4) for i in range(1, 7)], [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
