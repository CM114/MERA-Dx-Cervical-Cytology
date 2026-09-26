import unittest

import torch
import torch.nn as nn

from experiments.C2R2.loss import c2r2_loss
from experiments.C2R2B.model import C2R2BExpertModel


class DummyC1(nn.Module):
    semantic_dim = 8

    def __init__(self):
        super().__init__(); self.anchor = nn.Parameter(torch.randn(3))

    def forward(self, images):
        n = images.shape[0]; z = torch.randn(n, 8, device=images.device)
        logits = torch.tensor([[.1, 1.0, .5, .2, -.1]], device=images.device).repeat(n, 1)
        p = torch.softmax(logits, 1); q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {"features": z, "morph_features": z, "evidence_features": z, "morph_probs": torch.full((n, 2), .5, device=images.device), "evidence_probs": torch.full((n, 2), .5, device=images.device), "p_final": p, "q_final": q, "p_b0": p, "logits_b0": logits}


class C2R2BTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11); self.model = C2R2BExpertModel(DummyC1(), adapter_dim=4); self.images = torch.randn(6, 3, 8, 8)

    def test_pair_gate_and_mass_closure(self):
        out = self.model(self.images)
        self.assertEqual(tuple(out["pair_gate"].shape), (6, 2))
        self.assertTrue(torch.all(out["pair_gate"] <= .15 + 1e-6))
        self.assertTrue(torch.allclose(out["p_c2"].sum(1), torch.ones(6), atol=1e-6))
        self.assertTrue(torch.allclose(out["p_c2"][:, 0], out["p_c1"][:, 0], atol=1e-6))

    def test_pair_delta_signs_are_directed(self):
        out = self.model(self.images)
        d = out["evidence_pair_deltas"]; r = out["morph_routing"]; g = out["pair_gate"]
        expected = torch.stack((-r[:, 0] * g[:, 0] * d[:, 0], r[:, 0] * g[:, 0] * d[:, 0], -r[:, 1] * g[:, 1] * d[:, 1], r[:, 1] * g[:, 1] * d[:, 1]), 1)
        self.assertTrue(torch.allclose(out["prototype_delta"], expected, atol=1e-6))

    def test_zero_gate_is_c1_and_c1_is_frozen(self):
        before = self.model.c1_parameter_hash(); out = self.model(self.images, gate_override=torch.zeros(6, 2))
        self.assertTrue(torch.allclose(out["p_c1"], out["p_c2"], atol=1e-6)); self.assertEqual(before, self.model.c1_parameter_hash())
        self.assertTrue(all(not p.requires_grad for p in self.model.c1.parameters()))

    def test_loss_finite_and_normal_excluded(self):
        out = self.model(self.images); y = torch.tensor([0, 1, 2, 3, 4, 0]); morph = torch.tensor([-1, 0, 0, 1, 1, -1]); evidence = torch.tensor([-1, 0, 1, 0, 1, -1]); mask = morph >= 0
        self.assertTrue(torch.isfinite(c2r2_loss(out, y, morph, evidence, mask)["loss"]))
        self.model.accumulate_epoch_statistics(out, morph, evidence, mask); self.assertEqual(int(self.model.normal_proto_update_count), 0)


if __name__ == "__main__": unittest.main()
