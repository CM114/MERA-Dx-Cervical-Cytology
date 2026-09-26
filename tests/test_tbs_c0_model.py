import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from experiments.tbs.c0_model import LocalCellViewGenerator, TBSDualViewC0Model


if torch is not None:

    class TinyBackbone(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.projection = nn.Linear(3, feature_dim)

        def forward(self, images):
            return self.projection(images.mean(dim=(2, 3)))


@unittest.skipIf(torch is None, "PyTorch is required for C0 model tests")
class C0ModelTests(unittest.TestCase):
    def test_local_view_generator_returns_bounded_local_cell_view(self):
        generator = LocalCellViewGenerator()
        images = torch.randn(4, 3, 16, 16)
        local_images, theta = generator(images)

        self.assertEqual(tuple(local_images.shape), tuple(images.shape))
        self.assertEqual(tuple(theta.shape), (4, 2, 3))
        determinant = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        self.assertTrue(torch.all(determinant > 0.0))
        self.assertTrue(torch.all(torch.abs(theta[:, :, 2]) <= 0.35))

    def test_localizer_hard_bounds_survive_extreme_affine_logits(self):
        generator = LocalCellViewGenerator()
        with torch.no_grad():
            generator.affine[-1].bias.fill_(100.0)
        _, theta = generator(torch.randn(2, 3, 16, 16))
        determinant = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        self.assertTrue(torch.all(determinant > 0.20))
        self.assertTrue(torch.all(determinant < 0.64))
        self.assertTrue(torch.all(torch.abs(theta[:, :, 2]) <= 0.35))

    def test_zero_local_head_preserves_full_view_logits_at_initialization(self):
        model = TBSDualViewC0Model(TinyBackbone(), feature_dim=8)
        self.assertTrue(model.activation_checkpointing)
        images = torch.randn(5, 3, 16, 16)
        output = model(images)

        full_only = output["full_diagnosis_logits"]
        fused = output["diagnosis_logits"]
        self.assertTrue(torch.allclose(fused, full_only, atol=1e-6))
        self.assertEqual(float(model.diagnosis_head.weight[:, 8:].abs().sum()), 0.0)

    def test_forward_exposes_view_and_geometry_audit_fields(self):
        model = TBSDualViewC0Model(TinyBackbone(), feature_dim=8)
        output = model(torch.randn(3, 3, 16, 16))

        for key in (
            "diagnosis_logits",
            "diagnosis_log_probs",
            "diagnosis_probs",
            "full_features",
            "local_features",
            "view_gate",
            "theta",
            "local_area",
            "local_translation_abs",
            "local_cosine_similarity",
        ):
            self.assertIn(key, output)
        self.assertEqual(tuple(output["diagnosis_probs"].shape), (3, 5))
        self.assertTrue(torch.isfinite(output["local_area"]).all())
        self.assertTrue(torch.isfinite(output["local_cosine_similarity"]).all())

    def test_geometry_and_diagnosis_loss_backpropagate_to_view_path(self):
        from experiments.tbs.c0_loss import c0_loss

        model = TBSDualViewC0Model(TinyBackbone(), feature_dim=8)
        with torch.no_grad():
            model.diagnosis_head.weight[:, 8:].fill_(0.1)
        output = model(torch.randn(5, 3, 16, 16))
        labels = torch.tensor([0, 1, 2, 3, 4])
        losses = c0_loss(output, labels)
        losses["loss"].backward()

        self.assertGreater(float(model.backbone.projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.local_view.affine[-1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
