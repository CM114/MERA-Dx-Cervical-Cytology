import unittest

from experiments.C2Semantic.protocol import C2SEMANTIC_CONFIG


class C2SemanticProtocolTests(unittest.TestCase):
    def test_semantic_geometry_route_removes_correction_mechanisms(self):
        self.assertEqual(C2SEMANTIC_CONFIG["stage"], "C2-SemanticGeometry")
        self.assertEqual(C2SEMANTIC_CONFIG["embedding_dim"], 32)
        self.assertEqual(C2SEMANTIC_CONFIG["prototypes_per_state"], 2)
        self.assertEqual(C2SEMANTIC_CONFIG["prototype_count"], 8)
        self.assertTrue(C2SEMANTIC_CONFIG["freeze_all_c1"])
        self.assertTrue(C2SEMANTIC_CONFIG["q_proto_not_fused"])
        self.assertTrue(C2SEMANTIC_CONFIG["normal_excluded"])
        self.assertFalse(C2SEMANTIC_CONFIG["poe_enabled"])
        self.assertFalse(C2SEMANTIC_CONFIG["lg_guard_enabled"])
        self.assertEqual(C2SEMANTIC_CONFIG["epochs"], 12)
        self.assertEqual(C2SEMANTIC_CONFIG["batch_size"], 32)

    def test_fccon_ablation_changes_only_fccon_weight(self):
        self.assertEqual(C2SEMANTIC_CONFIG["lambda_fccon"], 0.10)
        self.assertEqual(C2SEMANTIC_CONFIG["ablation_lambda_fccon"], 0.0)
        for key in ("lambda_proto_cls", "lambda_pair", "lambda_m", "lambda_e", "lambda_balance", "lambda_diversity"):
            self.assertEqual(C2SEMANTIC_CONFIG[key], C2SEMANTIC_CONFIG[f"ablation_{key}"])


if __name__ == "__main__":
    unittest.main()
