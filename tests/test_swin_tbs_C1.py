import unittest

from experiments.C1.protocol import LOCKED_C1_CONFIG

try:
    import torch
    import torch.nn as nn

    from experiments.C1.loss import c1_loss
    from experiments.C1.model import C1FactorizedModel, factorized_probabilities
except ModuleNotFoundError:
    torch = None
    nn = None
    c1_loss = None
    C1FactorizedModel = None
    factorized_probabilities = None


if nn is None:
    class TinyBackbone:
        pass
else:
    class TinyBackbone(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.num_features = feature_dim
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(3, feature_dim)

        def forward(self, images):
            return self.projection(self.pool(images).flatten(1))


class SwinTbsC1Tests(unittest.TestCase):
    def test_c1_locks_true_factorized_composition(self):
        self.assertEqual(LOCKED_C1_CONFIG["stage"], "C1")
        self.assertEqual(LOCKED_C1_CONFIG["objective"], "tbs_factorized_diagnosis")
        self.assertEqual(LOCKED_C1_CONFIG["factor_count"], 4)

    def test_c1_r1_conditions_evidence_on_morphology_and_tightens_residual(self):
        self.assertEqual(LOCKED_C1_CONFIG["revision"], "R1")
        self.assertEqual(LOCKED_C1_CONFIG["evidence_conditioning"], "morph_semantic")
        self.assertLessEqual(LOCKED_C1_CONFIG["max_residual"], 0.03)

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_factorized_probabilities_sum_to_one_and_use_factor_order(self):
        screen = torch.tensor([0.0])
        morph = torch.tensor([[8.0, -8.0]])
        evidence = torch.tensor([[8.0, -8.0]])
        probabilities = factorized_probabilities(screen, morph, evidence)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(1), atol=1e-6))
        self.assertGreater(float(probabilities[0, 1]), 0.99)

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_residual_is_zero_at_initialization_and_screen_loss_is_connected(self):
        model = C1FactorizedModel(TinyBackbone(), feature_dim=8, semantic_dim=4)
        output = model(torch.randn(3, 3, 32, 32))
        self.assertTrue(torch.allclose(output["residual"], torch.zeros_like(output["residual"])))
        losses = c1_loss(
            output,
            torch.tensor([0, 1, 3]),
            torch.tensor([0, 1, 1]),
            torch.tensor([-1, 0, 1]),
            torch.tensor([-1, 0, 1]),
            torch.tensor([0, 1, 1]),
        )
        self.assertGreater(float(losses["screen_loss"]), 0.0)
        self.assertGreater(float(losses["loss"]), float(losses["diagnosis_loss"]))


if __name__ == "__main__":
    unittest.main()
