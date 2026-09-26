import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from experiments.tbs.c0r1_model import TBSC0R1ResidualModel


if torch is not None:

    class TinyBackbone(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.projection = nn.Linear(3, feature_dim)

        def forward(self, images):
            return self.projection(images.mean(dim=(2, 3)))


@unittest.skipIf(torch is None, "PyTorch is required for C0-R1 model tests")
class C0R1ModelTests(unittest.TestCase):
    def test_zero_residual_preserves_full_view_logits(self):
        model = TBSC0R1ResidualModel(TinyBackbone(), feature_dim=8)
        output = model(torch.randn(4, 3, 16, 16))
        self.assertTrue(torch.allclose(output["diagnosis_logits"], output["full_diagnosis_logits"], atol=1e-6))
        self.assertEqual(float(output["residual_logits"].abs().sum()), 0.0)

    def test_residual_logits_have_hard_bound(self):
        model = TBSC0R1ResidualModel(TinyBackbone(), feature_dim=8)
        with torch.no_grad():
            model.residual_head.bias.fill_(100.0)
        output = model(torch.randn(3, 3, 16, 16))
        self.assertLessEqual(float(output["residual_logits"].abs().max()), 0.10 + 1e-6)
        self.assertLessEqual(float(output["residual_logit_max_abs"]), 0.10 + 1e-6)

    def test_geometry_and_residual_paths_backpropagate(self):
        from experiments.tbs.c0r1_loss import c0r1_loss

        model = TBSC0R1ResidualModel(TinyBackbone(), feature_dim=8)
        with torch.no_grad():
            model.residual_head.weight.fill_(0.1)
        output = model(torch.randn(5, 3, 16, 16))
        losses = c0r1_loss(output, torch.tensor([0, 1, 2, 3, 4]))
        losses["loss"].backward()
        self.assertGreater(float(model.backbone.projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.local_view.affine[-1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
