import unittest

import torch
import torch.nn as nn

from experiments.C2Semantic.loss import c2semantic_loss, factor_controlled_contrastive_loss, factor_targets
from experiments.C2Semantic.model import C2SemanticModel
from experiments.C2Semantic.sampler import FourClassBalancedBatchSampler


class DummyC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_layer = nn.Linear(8, 8)
        self.direct_head = nn.Linear(8, 5)

    def forward(self, images):
        h = torch.tanh(self.features_layer(images.flatten(1)))
        logits = self.direct_head(h)
        p = torch.softmax(logits, dim=1)
        return {"features": h, "p_final": p, "logits_b0": logits}


class C2SemanticTests(unittest.TestCase):
    def _model(self):
        return C2SemanticModel(
            DummyC1(),
            feature_dim=8,
            embedding_dim=4,
            prototypes_per_state=2,
            prototype_temperature=0.2,
        )

    def test_c1_is_frozen_and_c2_does_not_fuse_into_c1(self):
        model = self._model()
        out = model(torch.randn(8, 1, 2, 4))
        self.assertFalse(any(parameter.requires_grad for parameter in model.c1.parameters()))
        self.assertFalse(out["features"].requires_grad)
        self.assertNotIn("p_final", out)
        self.assertEqual(tuple(out["q_proto"].shape), (8, 4))

    def test_factor_targets_are_deterministic_and_normal_is_masked(self):
        morph, evidence, semantic = factor_targets(torch.tensor([0, 1, 2, 3, 4]))
        self.assertEqual(morph.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(evidence.tolist(), [0, 0, 1, 0, 1])
        self.assertEqual(semantic.tolist(), [False, True, True, True, True])

    def test_factor_controlled_contrastive_has_gradient(self):
        z_m = torch.randn(8, 4, requires_grad=True)
        z_e = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        loss = factor_controlled_contrastive_loss(z_m, z_e, labels, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(z_m.grad.abs().sum() + z_e.grad.abs().sum()), 1e-8)

    def test_semantic_loss_has_all_locked_terms_and_gradient(self):
        model = self._model()
        out = model(torch.randn(8, 1, 2, 4))
        labels = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        losses = c2semantic_loss(out, labels)
        for key in ("loss", "proto_cls_loss", "pair_loss", "m_state_loss", "e_state_loss", "fccon_loss", "balance_loss", "diversity_loss"):
            self.assertIn(key, losses)
        losses["loss"].backward()
        self.assertGreater(float(model.morph_prototypes.grad.abs().sum()), 1e-8)
        self.assertGreater(float(model.evidence_prototypes.grad.abs().sum()), 1e-8)

    def test_balanced_sampler_emits_all_four_abnormal_classes(self):
        labels = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4]
        sampler = FourClassBalancedBatchSampler(labels, batch_size=8, seed=42)
        batches = list(iter(sampler))
        self.assertTrue(batches)
        for batch in batches:
            values = [labels[index] for index in batch]
            self.assertEqual(sorted(values), [1, 1, 2, 2, 3, 3, 4, 4])


if __name__ == "__main__":
    unittest.main()
