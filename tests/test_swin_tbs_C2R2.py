import unittest

import torch
import torch.nn as nn

from experiments.C2R2.loss import c2r2_loss
from experiments.C2R2.model import C2R2ExpertModel


class DummyC1(nn.Module):
    semantic_dim = 8

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(3))

    def forward(self, images):
        n = images.shape[0]
        z = torch.ones(n, 8, device=images.device)
        p = torch.softmax(torch.tensor([[0.1, 1.0, .5, .2, -.1]], device=images.device).repeat(n, 1), 1)
        q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {"features": z, "morph_features": z, "evidence_features": z, "morph_probs": torch.full((n, 2), .5, device=images.device), "evidence_probs": torch.full((n, 2), .5, device=images.device), "p_final": p, "q_final": q, "p_b0": p, "logits_b0": p.log()}


class C2R2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = C2R2ExpertModel(DummyC1(), adapter_dim=4)
        self.images = torch.randn(6, 3, 8, 8)

    def test_c1_is_frozen_and_gate_zero_is_identical(self):
        before = self.model.c1_parameter_hash()
        out1 = self.model(self.images, gate_override=torch.zeros(6))
        self.assertTrue(torch.allclose(out1["p_c1"], out1["p_c2"], atol=1e-6))
        delta = torch.tensor([[1., -1., .5, -.5]]).repeat(6, 1)
        fused, _ = self.model.fuse_with_gate(out1["p_c1"], out1["q_c1"], delta, torch.full((6, 1), .12))
        self.assertFalse(torch.allclose(out1["p_c1"], fused))
        self.assertEqual(before, self.model.c1_parameter_hash())
        self.assertTrue(all(not p.requires_grad for p in self.model.c1.parameters()))

    def test_mass_rows_and_gate_residual_bounds(self):
        out = self.model(self.images)
        self.assertTrue(torch.allclose(out["p_c2"].sum(1), torch.ones(6), atol=1e-6))
        self.assertTrue(torch.allclose(out["p_c2"][:, 0], out["p_c1"][:, 0], atol=1e-6))
        self.assertLessEqual(float(out["gate"].max().detach()), .15 + 1e-6)
        self.assertLessEqual(float(out["family_residuals"].norm(dim=-1).max().detach()), .15 + 1e-6)
        self.assertTrue(torch.allclose(self.model.family_residual_raw, torch.zeros_like(self.model.family_residual_raw)))

    def test_loss_is_finite_and_normal_excluded_from_ema(self):
        out = self.model(self.images, gate_override=torch.full((6,), .1))
        y = torch.tensor([0, 1, 2, 3, 4, 0]); morph = torch.tensor([-1, 0, 0, 1, 1, -1]); evidence = torch.tensor([-1, 0, 1, 0, 1, -1]); mask = morph >= 0
        losses = c2r2_loss(out, y, morph, evidence, mask)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.model.accumulate_epoch_statistics(out, morph, evidence, mask)
        self.assertEqual(int(self.model.normal_proto_update_count), 0)


if __name__ == "__main__":
    unittest.main()
