import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.tbs_fv_factorized_model import (
        TBSFVFactorizedResidualModel,
        compose_tbs_factor_probabilities,
    )


@unittest.skipIf(torch is None, "PyTorch is required for full-view factorized model tests")
class TBSFVFactorizedModelTests(unittest.TestCase):
    def test_factor_probabilities_have_five_classes_and_sum_to_one(self):
        screen = torch.tensor([0.8])
        morph = torch.tensor([[1.0, 0.0]])
        evidence = torch.tensor([[1.0, 0.0]])
        probabilities = compose_tbs_factor_probabilities(screen, morph, evidence)
        self.assertEqual(tuple(probabilities.shape), (1, 5))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1)))
        self.assertGreater(float(probabilities[0, 1]), 0.79)

    def test_zero_residual_preserves_base_logits(self):
        class TinyBackbone(torch.nn.Module):
            num_features = 8

            def forward(self, images):
                scalar = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return scalar.repeat(1, 8)

        model = TBSFVFactorizedResidualModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, residual_logit_bound=0.10
        )
        output = model(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.allclose(output["diagnosis_logits"], output["base_diagnosis_logits"]))
        self.assertEqual(float(output["residual_logit_max_abs"]), 0.0)

    def test_residual_logits_have_hard_bound(self):
        class TinyBackbone(torch.nn.Module):
            num_features = 8

            def forward(self, images):
                scalar = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return scalar.repeat(1, 8)

        model = TBSFVFactorizedResidualModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, residual_logit_bound=0.10
        )
        with torch.no_grad():
            model.residual_head[-1].bias.fill_(100.0)
        output = model(torch.randn(2, 3, 8, 8))
        self.assertLessEqual(float(output["residual_logit_max_abs"]), 0.10 + 1e-6)


if __name__ == "__main__":
    unittest.main()
