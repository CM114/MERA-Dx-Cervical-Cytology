import unittest

try:
    import torch
    import torch.nn as nn

    from experiments.C1R2.loss import c1r2_loss
    from experiments.C1R2.model import (
        C1R2Model,
        checkerboard_joint_logits,
        mass_preserving_fusion,
    )
except ModuleNotFoundError:
    torch = None
    nn = None
    c1r2_loss = None
    C1R2Model = None
    checkerboard_joint_logits = None
    mass_preserving_fusion = None


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


class SwinTbsC1R2Tests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_checkerboard_eta_gradient_is_nonzero(self):
        # A common shift [eta, eta, eta, eta] is softmax-invariant and is not
        # a valid interaction-gradient test.  The checkerboard signs make eta
        # change the relative odds of the four TBS cells.
        eta = torch.zeros(2, requires_grad=True)
        joint_logits = torch.stack(
            [eta, -eta, -eta, eta],
            dim=1,
        )
        labels = torch.tensor([0, 3])
        torch.nn.functional.cross_entropy(joint_logits, labels).backward()
        self.assertIsNotNone(eta.grad)
        self.assertGreater(float(eta.grad.abs().sum().item()), 1e-8)

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_checkerboard_interaction_signs(self):
        morph = torch.tensor([[1.0, 2.0]])
        evidence = torch.tensor([[3.0, 4.0]])
        eta = torch.tensor([0.5])
        got = checkerboard_joint_logits(morph, evidence, eta)
        expected = torch.tensor([[4.5, 4.5, 4.5, 6.5]])
        self.assertTrue(torch.allclose(got, expected))

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_mass_preserving_fusion(self):
        output = mass_preserving_fusion(
            torch.randn(4, 5), torch.randn(4, 4), torch.tensor(0.05)
        )
        self.assertTrue(torch.allclose(output["p_final"][:, 0], output["p_b0"][:, 0], atol=1e-6))
        self.assertTrue(torch.allclose(output["p_final"][:, 1:].sum(1), output["p_b0"][:, 1:].sum(1), atol=1e-6))
        self.assertTrue(torch.allclose(output["p_final"].sum(1), torch.ones(4), atol=1e-6))
        self.assertLess(float(output["screen_mass_error"]), 1e-6)

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_interaction_is_zero_initialized_and_joint_loss_has_nonzero_gradient(self):
        model = C1R2Model(TinyBackbone(), feature_dim=8, semantic_dim=4, interaction_dim=3)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertTrue(torch.allclose(output["eta"], torch.zeros_like(output["eta"])))
        losses = c1r2_loss(
            output,
            torch.tensor([1, 4]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
        )
        losses["loss"].backward()
        eta_grad = output["eta"].grad
        self.assertIsNotNone(eta_grad)
        self.assertGreater(float(eta_grad.abs().sum().item()), 1e-8)


if __name__ == "__main__":
    unittest.main()
