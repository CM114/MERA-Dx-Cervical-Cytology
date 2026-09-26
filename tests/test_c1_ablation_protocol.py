import unittest

from experiments.C1Ablation.protocol import VARIANTS, validate_variant


class C1AblationProtocolTests(unittest.TestCase):
    def test_variants_have_isolated_semantic_switches(self):
        self.assertEqual(validate_variant("direct_only")["lambda_morph"], 0.0)
        self.assertEqual(validate_variant("morph_only")["lambda_morph"], 0.15)
        self.assertEqual(validate_variant("morph_only")["lambda_evidence"], 0.0)
        self.assertEqual(validate_variant("evidence_only")["lambda_evidence"], 0.15)
        self.assertFalse(validate_variant("both_no_interaction")["enable_interaction"])
        self.assertTrue(validate_variant("full_c1r2_reference")["enable_interaction"])

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_variant("not_a_variant")

    def test_disabled_interaction_and_fusion_are_direct_identity(self):
        try:
            import torch
            from experiments.C1R2.model import C1R2Model
        except ImportError:
            self.skipTest("torch is available in the transmil server environment")

        class MeanBackbone(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=(2, 3))

        model = C1R2Model(
            MeanBackbone(), feature_dim=3, semantic_dim=4, interaction_dim=2,
            enable_interaction=False, enable_fusion=False,
        )
        output = model(torch.randn(3, 3, 4, 4))
        self.assertTrue(torch.allclose(output["p_final"], output["p_b0"], atol=1e-6))
        self.assertTrue(torch.allclose(output["eta"], torch.zeros_like(output["eta"])))
        self.assertEqual(float(output["alpha"]), 0.0)


if __name__ == "__main__":
    unittest.main()
