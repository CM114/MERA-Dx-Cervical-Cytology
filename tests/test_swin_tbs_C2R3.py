import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.C2R3.inner_split import build_inner_splits
from experiments.C2R3.loss import c2r3_loss
from experiments.C2R3.model import C2R3AdapterModel
from experiments.C2R3.training import _official_pair_macro_f1


class DummyC1(nn.Module):
    semantic_dim = 8

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(3))

    def forward(self, images):
        n = images.shape[0]
        # Deterministic features make invariance tests meaningful.
        base = images.flatten(1).mean(1, keepdim=True)
        z = base.repeat(1, 8)
        logits = torch.tensor([[.1, 1., .5, .2, -.1]], device=images.device).repeat(n, 1)
        p = torch.softmax(logits, 1)
        q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {
            "features": z,
            "morph_features": z,
            "evidence_features": z,
            "morph_probs": torch.full((n, 2), .5, device=images.device),
            "evidence_probs": torch.full((n, 2), .5, device=images.device),
            "p_final": p,
            "q_final": q,
            "p_b0": p,
            "logits_b0": logits,
            "eta": torch.zeros(n, device=images.device),
        }


class C2R3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = C2R3AdapterModel(DummyC1(), semantic_dim=8, hidden_dim=16, residual_bound=.15)
        self.images = torch.randn(6, 3, 8, 8)

    def test_c1_is_frozen_and_family_mass_is_exact(self):
        out = self.model(self.images)
        self.assertTrue(all(not p.requires_grad for p in self.model.c1.parameters()))
        self.assertTrue(torch.allclose(out["p_c2"][:, 0], out["p_c1"][:, 0], atol=1e-6))
        self.assertTrue(torch.allclose(out["q_c2"][:, :2].sum(1), out["q_c1"][:, :2].sum(1), atol=1e-6))
        self.assertTrue(torch.allclose(out["q_c2"][:, 2:].sum(1), out["q_c1"][:, 2:].sum(1), atol=1e-6))

    def test_zero_initialized_adapter_is_exact_c1(self):
        out = self.model(self.images)
        self.assertTrue(torch.allclose(out["p_c2"], out["p_c1"], atol=1e-7, rtol=0.0))
        self.assertEqual(float(out["actual_max_residual"].detach().item()), 0.0)

    def test_prototype_sign_is_feature_only_and_adapter_has_gradient(self):
        out = self.model(self.images)
        y = torch.tensor([1, 2, 3, 4, 1, 4])
        mask = torch.ones(6, dtype=torch.bool)
        losses = c2r3_loss(out, y, mask)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        grad = sum(p.grad.abs().sum() for p in self.model.trainable_parameters if p.grad is not None)
        self.assertGreater(float(grad), 0.0)

    def test_disagreement_orientation(self):
        # C1 AU preference => ambiguous; prototype ambiguous preference => agreement.
        c1_amb_over_def = torch.tensor([2.0, 2.0, -2.0, -2.0])
        proto_def_minus_amb = torch.tensor([-1.0, 1.0, 1.0, -1.0])
        got = self.model._definitive_disagreement(c1_amb_over_def, proto_def_minus_amb)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.equal(got, expected))

    def test_normal_only_batch_never_counts_as_proto_update(self):
        out = self.model(self.images)
        n = self.images.shape[0]
        self.model.reset_epoch_accumulators()
        self.model.accumulate_epoch_statistics(
            out,
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.long),
        )
        self.assertEqual(int(self.model.normal_proto_update_count.item()), 0)

    def test_official_pair_metric_is_pair_local(self):
        labels = np.array([1, 2, 1, 2])
        # Global argmax is Normal for every row, but pair-local decisions are perfect.
        prob = np.array([
            [.60, .25, .15, 0., 0.],
            [.60, .15, .25, 0., 0.],
            [.55, .30, .15, 0., 0.],
            [.55, .15, .30, 0., 0.],
        ])
        self.assertAlmostEqual(_official_pair_macro_f1(labels, prob, (1, 2)), 1.0, places=7)

    def test_inner_split_is_deterministic_disjoint_and_handles_missing_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in range(5):
                for j in range(6):
                    i = label * 6 + j
                    rows.append({
                        "image_path": f"img_{i}.png",
                        "diagnosis_label": label,
                        "patient_id": None if j % 2 == 0 else f"p_{i}",
                        "slide_id": f"s_{i}",
                        "content_sha256": f"hash_{i}",
                    })
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            first = build_inner_splits(source, root / "a", n_folds=3)
            second = build_inner_splits(source, root / "b", n_folds=3)
            for fold in range(3):
                a_dev = pd.read_csv(first[fold]["dev"])
                b_dev = pd.read_csv(second[fold]["dev"])
                self.assertEqual(a_dev["content_sha256"].tolist(), b_dev["content_sha256"].tolist())
                train = pd.read_csv(first[fold]["train"])
                self.assertEqual(set(train["content_sha256"]).intersection(a_dev["content_sha256"]), set())
                self.assertTrue({1, 2, 3, 4}.issubset(set(a_dev["diagnosis_label"].astype(int))))


if __name__ == "__main__":
    unittest.main()
