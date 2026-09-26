import unittest

import torch
import torch.nn as nn

from experiments.C2R2C.loss import c2r2c_loss
from experiments.C2R2C.model import C2R2CExpertModel


class DummyC1(nn.Module):
    semantic_dim = 8

    def __init__(self):
        super().__init__(); self.anchor = nn.Parameter(torch.randn(3))

    def forward(self, images):
        n=images.shape[0]; z=torch.randn(n,8,device=images.device); logits=torch.tensor([[.1,1.,.5,.2,-.1]],device=images.device).repeat(n,1); p=torch.softmax(logits,1); q=p[:,1:]/p[:,1:].sum(1,keepdim=True)
        return {"features":z,"morph_features":z,"evidence_features":z,"morph_probs":torch.full((n,2),.5,device=images.device),"evidence_probs":torch.full((n,2),.5,device=images.device),"p_final":p,"q_final":q,"p_b0":p,"logits_b0":logits}


class C2R2CTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13); self.model=C2R2CExpertModel(DummyC1(),adapter_dim=4); self.images=torch.randn(6,3,8,8)

    def test_exact_family_and_screen_mass_closure(self):
        out=self.model(self.images)
        self.assertTrue(torch.allclose(out["q_c2"][:,:2].sum(1),out["q_c1"][:,:2].sum(1),atol=1e-6))
        self.assertTrue(torch.allclose(out["q_c2"][:,2:].sum(1),out["q_c1"][:,2:].sum(1),atol=1e-6))
        self.assertTrue(torch.allclose(out["p_c2"][:,0],out["p_c1"][:,0],atol=1e-6))
        self.assertTrue(torch.allclose(out["p_c2"].sum(1),torch.ones(6),atol=1e-6))

    def test_smooth_residual_does_not_jump_to_rho(self):
        zero = self.model._smooth_residuals().sum(); zero.backward()
        self.assertGreater(float(self.model.family_residual_raw.grad.abs().sum()), 0.0)
        self.model.family_residual_raw.grad = None
        with torch.no_grad(): self.model.family_residual_raw[0,0,0]=.01
        residual=self.model._smooth_residuals()
        self.assertGreater(float(residual[0,0].norm()),0.0); self.assertLess(float(residual[0,0].norm()),.02); self.assertLessEqual(float(residual.norm(dim=-1).max()),.15+1e-6)

    def test_zero_gate_and_loss(self):
        out=self.model(self.images,gate_override=torch.zeros(6,2)); self.assertTrue(torch.allclose(out["p_c1"],out["p_c2"],atol=1e-6))
        y=torch.tensor([0,1,2,3,4,0]); morph=torch.tensor([-1,0,0,1,1,-1]); evidence=torch.tensor([-1,0,1,0,1,-1]); mask=morph>=0
        self.assertTrue(torch.isfinite(c2r2c_loss(out,y,morph,evidence,mask)["loss"]))
        self.assertTrue(all(not p.requires_grad for p in self.model.c1.parameters()))

    def test_pair_correction_is_reliability_routed_by_morphology(self):
        with torch.no_grad():
            self.model.morph_prototypes[0, 0] = 1.0
            self.model.morph_prototypes[1, 1] = 1.0
            self.model.evidence_prototypes[0, 0] = 1.0
            self.model.evidence_prototypes[1, 1] = 1.0
        out = self.model(self.images, gate_override=torch.full((6, 2), 0.1))
        expected_low = out["pair_gate"][:, 0] * out["morph_routing"][:, 0] * out["evidence_pair_deltas"][:, 0]
        expected_high = out["pair_gate"][:, 1] * out["morph_routing"][:, 1] * out["evidence_pair_deltas"][:, 1]
        self.assertTrue(torch.allclose(out["pair_correction"][:, 0], expected_low, atol=1e-6))
        self.assertTrue(torch.allclose(out["pair_correction"][:, 1], expected_high, atol=1e-6))


if __name__=="__main__": unittest.main()
