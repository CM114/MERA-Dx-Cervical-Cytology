import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from experiments.tbs.s1r2_model import TBSResidualSingleViewModel

    class TinyBackbone(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.projection = nn.Linear(3, feature_dim)

        def forward(self, images):
            return self.projection(images.mean(dim=(2, 3)))


@unittest.skipIf(torch is None, "PyTorch is required for S1-R2 model tests")
class S1R2ModelTests(unittest.TestCase):
    def test_zero_initialized_residual_preserves_direct_logits(self):
        model = TBSResidualSingleViewModel(TinyBackbone(), 8, semantic_dim=4)
        output = model(torch.randn(6, 3, 16, 16))
        self.assertTrue(torch.equal(output["diagnosis_logits"], output["direct_diagnosis_logits"]))
        self.assertEqual(float(output["factorized_residual_logits"].abs().sum()), 0.0)
        self.assertEqual(float(output["residual_adapter_weight_norm"]), 0.0)

    def test_residual_adapter_and_semantic_heads_receive_gradients(self):
        from experiments.tbs.singleview_model import singleview_tbs_loss

        model = TBSResidualSingleViewModel(TinyBackbone(), 8, semantic_dim=4)
        output = model(torch.randn(5, 3, 16, 16))
        targets = {
            "diagnosis_labels": torch.tensor([0, 1, 2, 3, 4]),
            "screen_labels": torch.tensor([0, 1, 1, 1, 1]),
            "morph_labels": torch.tensor([0, 0, 0, 1, 1]),
            "evidence_labels": torch.tensor([0, 0, 1, 0, 1]),
            "semantic_mask": torch.tensor([False, True, True, True, True]),
        }
        singleview_tbs_loss(output, targets, stage="s1")["loss"].backward()
        self.assertGreater(float(model.residual_adapter.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.diagnosis_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.screen_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.morph_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.evidence_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.morph_projector[0].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.evidence_projector[0].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.backbone.projection.weight.grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
