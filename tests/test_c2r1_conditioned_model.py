import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.tbs_fv_c2r1_model import (
        TBSFVDualPrototypeConditionedResidualModel,
    )


@unittest.skipIf(torch is None, "PyTorch is required for C2-R1 model tests")
class C2R1ConditionedModelTests(unittest.TestCase):
    def _model(self):
        class TinyBackbone(torch.nn.Module):
            num_features = 8

            def forward(self, images):
                scalar = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return scalar.repeat(1, 8)

        return TBSFVDualPrototypeConditionedResidualModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, residual_logit_bound=0.10
        )

    def test_prototype_context_is_in_residual_path(self):
        model = self._model()
        self.assertEqual(model.residual_head[0].in_features, 8 + 2 * 4 + 4)
        output = model(torch.randn(3, 3, 8, 8))
        self.assertEqual(tuple(output["prototype_context"].shape), (3, 4))
        self.assertTrue(torch.isfinite(output["prototype_context"]).all())

    def test_zero_initialized_conditioned_residual_preserves_base_logits(self):
        model = self._model()
        output = model(torch.randn(3, 3, 8, 8))
        self.assertTrue(
            torch.allclose(
                output["diagnosis_logits"], output["base_diagnosis_logits"], atol=1e-6
            )
        )

    def test_conditioned_residual_has_hard_bound(self):
        model = self._model()
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(100.0)
            model.residual_head[-1].bias.fill_(100.0)
        output = model(torch.randn(3, 3, 8, 8))
        self.assertLessEqual(float(output["residual_logits"].abs().max()), 0.10 + 1e-6)


if __name__ == "__main__":
    unittest.main()
