import unittest

import torch
import torch.nn as nn

from experiments.C2P.loss import (
    c2p_loss,
    factor_controlled_contrastive_loss,
    low_pair_correct_preservation_guard,
    pair_difficulty_weights,
)
from experiments.C2P.model import C2PModel, compose_geometric_product


class DummyC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict({
            "stage1": nn.Linear(8, 8),
            "stage2": nn.Linear(8, 8),
            "stage3": nn.Linear(8, 8),
            "stage4": nn.Linear(8, 8),
        })
        self.direct_head = nn.Linear(8, 5)

    def forward(self, images):
        x = images.flatten(1)
        for stage in self.backbone.values():
            x = torch.tanh(stage(x))
        logits = self.direct_head(x)
        p = torch.softmax(logits, dim=1)
        q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {
            "features": x,
            "p_final": p,
            "q_final": q,
            "eta": torch.zeros(x.shape[0], device=x.device),
        }


class C2PTests(unittest.TestCase):
    def _model(self):
        return C2PModel(
            DummyC1(),
            feature_dim=8,
            semantic_dim=16,
            prototype_dim=8,
            prototypes_per_state=2,
            gamma=0.5,
            prototype_temperature=0.2,
        )

    def test_eight_trainable_prototypes_and_frozen_c1_policy(self):
        model = self._model()
        self.assertEqual(tuple(model.morph_prototypes.shape), (2, 2, 8))
        self.assertEqual(tuple(model.evidence_prototypes.shape), (2, 2, 8))
        model.configure_trainable(joint=False)
        self.assertFalse(any(p.requires_grad for p in model.c1.parameters()))
        model.configure_trainable(joint=True)
        self.assertFalse(any(p.requires_grad for p in model.c1.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.morph_projection.parameters()))
        direct = [p for n, p in model.c1.named_parameters() if "direct_head" in n]
        self.assertTrue(direct and not any(p.requires_grad for p in direct))

    def test_adapters_are_nonlinear_and_c1_features_are_detached(self):
        model = self._model()
        self.assertTrue(any(isinstance(module, nn.Dropout) for module in model.morph_projection.modules()))
        self.assertTrue(any(isinstance(module, nn.Dropout) for module in model.evidence_projection.modules()))
        out = model(torch.randn(5, 1, 2, 4))
        self.assertFalse(out["features"].requires_grad)
        self.assertFalse(out["p_c1"].requires_grad)

    def test_composition_and_mass_preservation(self):
        model = self._model()
        out = model(torch.randn(5, 1, 2, 4))
        self.assertEqual(tuple(out["q_proto"].shape), (5, 4))
        self.assertLessEqual(float(out["screen_mass_error"]), 1e-6)
        self.assertLessEqual(float(out["abnormal_conditional_row_error"]), 1e-6)
        self.assertTrue(torch.isfinite(out["p_final"]).all())

    def test_geometric_product_uses_fixed_gamma(self):
        parent = torch.tensor([[0.1, 0.2, 0.3, 0.1]])
        expert = torch.tensor([[0.4, 0.1, 0.2, 0.3]])
        result = compose_geometric_product(parent, expert, gamma=0.5)
        expected = torch.sqrt(parent * expert)
        expected = expected / expected.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_factor_controlled_contrastive_has_gradient(self):
        z_m = torch.randn(8, 8, requires_grad=True)
        z_e = torch.randn(8, 8, requires_grad=True)
        labels = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        loss = factor_controlled_contrastive_loss(z_m, z_e, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(z_m.grad.abs().sum() + z_e.grad.abs().sum()), 1e-8)

    def test_frozen_c1_difficulty_weight_is_one_to_two(self):
        q_c1 = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.495, 0.005, 0.25, 0.25]])
        labels = torch.tensor([1, 1])
        weights = pair_difficulty_weights(q_c1, labels, (1, 2))
        self.assertTrue(torch.all(weights >= 1.0))
        self.assertTrue(torch.all(weights <= 2.0))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[0]), 1.5, places=6)
        self.assertAlmostEqual(float(weights[1]), 1.01, places=6)

    def test_c2p_loss_has_all_terms_and_gradient(self):
        model = self._model()
        images = torch.randn(8, 1, 2, 4)
        out = model(images)
        labels = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        losses = c2p_loss(out, labels)
        for key in ("loss", "five_class_loss", "pair_loss", "lg_guard_loss", "m_proto_loss", "e_proto_loss", "fccon_loss", "balance_loss", "diversity_loss"):
            self.assertIn(key, losses)
        losses["loss"].backward()
        self.assertGreater(float(model.morph_prototypes.grad.abs().sum()), 1e-8)
        self.assertGreater(float(model.evidence_prototypes.grad.abs().sum()), 1e-8)

    def test_low_pair_guard_only_penalizes_c1_correct_margin_regression(self):
        q_c1 = torch.tensor([
            [0.75, 0.25, 0.0, 0.0],  # label 1, C1 correct
            [0.25, 0.75, 0.0, 0.0],  # label 2, C1 correct
            [0.25, 0.75, 0.0, 0.0],  # label 1, C1 wrong: must stay plastic
        ])
        q_final = torch.tensor([
            [0.35, 0.65, 0.0, 0.0],
            [0.45, 0.55, 0.0, 0.0],
            [0.35, 0.65, 0.0, 0.0],
        ], requires_grad=True)
        labels = torch.tensor([1, 2, 1])
        guard = low_pair_correct_preservation_guard(q_c1, q_final, labels)
        self.assertGreater(float(guard), 0.0)
        guard.backward()
        self.assertGreater(float(q_final.grad[:2].abs().sum()), 1e-8)
        self.assertAlmostEqual(float(q_final.grad[2].abs().sum()), 0.0, places=8)

    def test_low_pair_guard_is_zero_when_c1_has_no_correct_low_samples(self):
        q_c1 = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
        q_final = torch.tensor([[0.35, 0.65, 0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1])
        guard = low_pair_correct_preservation_guard(q_c1, q_final, labels)
        self.assertAlmostEqual(float(guard), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
