import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.tbs_fv_factorized_model import TBSFVDualPrototypeResidualModel


@unittest.skipIf(torch is None, "PyTorch is required for dual-prototype model tests")
class C2DualPrototypeModelTests(unittest.TestCase):
    def _model(self):
        class TinyBackbone(torch.nn.Module):
            num_features = 8

            def forward(self, images):
                scalar = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return scalar.repeat(1, 8)

        return TBSFVDualPrototypeResidualModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, residual_logit_bound=0.10
        )

    def test_dual_prototypes_are_two_by_semantic_dim_and_normalized(self):
        model = self._model()
        self.assertEqual(tuple(model.morph_prototypes.shape), (2, 4))
        self.assertEqual(tuple(model.evidence_prototypes.shape), (2, 4))
        norms = torch.linalg.vector_norm(model.morph_prototypes.detach(), dim=1)
        self.assertTrue(torch.isfinite(norms).all())

    def test_forward_exposes_prototype_factor_logits(self):
        model = self._model()
        output = model(torch.randn(3, 3, 8, 8))
        self.assertEqual(tuple(output["morph_prototype_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["evidence_prototype_logits"].shape), (3, 2))

    def test_empty_factor_group_is_rejected_during_initialization(self):
        model = self._model()
        features = torch.randn(3, 4)
        labels = torch.tensor([0, 0, 0])
        with self.assertRaises(ValueError):
            model.initialize_prototypes_from_train_features(features, labels, features, labels)


if __name__ == "__main__":
    unittest.main()
