import unittest
from pathlib import Path

from experiments.C2P.protocol import C2P_CONFIG, require_locked_epoch
from experiments.C2P.runner import build_c2p_history_row
from experiments.audit_swin_tbs_C2P_smoke import resolve_smoke_output_dir


class C2PProtocolTests(unittest.TestCase):
    def test_locked_factor_controlled_route(self):
        self.assertEqual(C2P_CONFIG["revision"], "frozen_c1_prior_trainable_dual_space_prototype_adapters_lg_guard_v1")
        self.assertEqual(C2P_CONFIG["inner_fold_count"], 6)
        self.assertEqual(C2P_CONFIG["consumed_prior_smoke_inner_fold"], 3)
        self.assertEqual(C2P_CONFIG["smoke_inner_fold"], 4)
        self.assertEqual(C2P_CONFIG["confirm_inner_fold"], 5)
        self.assertAlmostEqual(C2P_CONFIG["lambda_lg_guard"], 0.20, places=8)
        self.assertEqual(C2P_CONFIG["prototypes_per_state"], 2)
        self.assertEqual(C2P_CONFIG["prototype_count"], 8)
        self.assertEqual(C2P_CONFIG["gamma"], 0.5)
        self.assertEqual(C2P_CONFIG["warmup_epochs"], 0)
        self.assertEqual(C2P_CONFIG["joint_epochs"], 12)
        self.assertEqual(C2P_CONFIG["epochs"], 12)
        self.assertEqual(C2P_CONFIG["inner_fold_count"], 6)
        self.assertEqual(C2P_CONFIG["smoke_inner_fold"], 4)
        self.assertEqual(C2P_CONFIG["confirm_inner_fold"], 5)
        self.assertTrue(C2P_CONFIG["freeze_all_c1"])
        self.assertFalse(C2P_CONFIG["train_stage4"])
        self.assertTrue(C2P_CONFIG["screen_mass_preserving"])
        self.assertTrue(C2P_CONFIG["prototype_similarity_does_not_directly_set_residual_sign"])

    def test_low_pair_gain_damage_balance_is_locked_as_a_gate(self):
        from experiments.C2P.protocol import SMOKE_RULE

        self.assertEqual(SMOKE_RULE["minimum_low_gain_damage_balance"], 0.0)

    def test_epoch_lock(self):
        require_locked_epoch(12)
        with self.assertRaises(ValueError):
            require_locked_epoch(10)

    def test_history_row_keeps_canonical_loss_columns(self):
        train_metrics = {
            "loss": 1.0,
            "five_class_loss": 0.5,
            "pair_loss": 0.2,
            "lg_guard_loss": 0.06,
            "m_proto_loss": 0.1,
            "e_proto_loss": 0.2,
            "fccon_loss": 0.3,
            "balance_loss": 0.4,
            "diversity_loss": 0.5,
        }
        evaluation = {"metrics": {"macro_f1": 0.7}}
        row = build_c2p_history_row(
            fold=2,
            outer_fold=0,
            epoch=12,
            phase="joint",
            train_metrics=train_metrics,
            evaluation=evaluation,
        )
        for key in ("m_proto_loss", "e_proto_loss", "fccon_loss", "balance_loss", "diversity_loss", "lg_guard_loss"):
            self.assertIn(key, row)
            self.assertEqual(row[key], train_metrics[key])
            self.assertEqual(row[f"train_{key}"], train_metrics[key])

    def test_smoke_output_dir_can_be_overridden_after_failed_attempt(self):
        project_root = "/tmp/c2p-project"
        explicit = "/tmp/c2p-project/results/c2p-retry"
        self.assertEqual(resolve_smoke_output_dir(project_root, explicit), Path(explicit).resolve())


if __name__ == "__main__":
    unittest.main()
