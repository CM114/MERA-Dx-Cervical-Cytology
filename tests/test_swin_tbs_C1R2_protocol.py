import unittest
from pathlib import Path


class SwinTbsC1R2ProtocolTests(unittest.TestCase):
    def test_c1r2_paths_are_isolated(self):
        from experiments.swin_tbs_common.paths import ProjectPaths

        paths = ProjectPaths(Path("/project"), Path("/source"))
        self.assertEqual(paths.c1r2_root.name, "C1_tbs_direct_diagnosis_interaction_r2_metricfix_v2")
        self.assertNotEqual(paths.c1r2_root, paths.c1_root)

    def test_protocol_locks_joint_supervision_and_mass_preservation(self):
        from experiments.C1R2.protocol import LOCKED_C1R2_CONFIG, REQUIRED_C1R2_METRICS

        self.assertEqual(LOCKED_C1R2_CONFIG["stage"], "C1-R2")
        self.assertAlmostEqual(LOCKED_C1R2_CONFIG["lambda_joint"], 0.10)
        self.assertEqual(LOCKED_C1R2_CONFIG["interaction_dim"], 32)
        self.assertAlmostEqual(LOCKED_C1R2_CONFIG["alpha_max"], 0.30)
        self.assertEqual(
            LOCKED_C1R2_CONFIG["pair_metric_protocol"],
            "locked_two_class_renormalized_binary_macro_f1",
        )
        self.assertFalse(LOCKED_C1R2_CONFIG["checkpoints_written"])
        self.assertIn("screen_mass_preservation_error", REQUIRED_C1R2_METRICS)
        self.assertIn("r2_direct_high_grade_pair_macro_f1", REQUIRED_C1R2_METRICS)


if __name__ == "__main__":
    unittest.main()
