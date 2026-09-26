import unittest

import torch
import torch.nn as nn

from experiments.tbs.singleview_model import TBSFactorizedSingleViewModel
from experiments.tbs.singleview_model import singleview_tbs_loss


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


class SingleViewModelTests(unittest.TestCase):
    def test_s0_is_an_ordinary_single_view_five_class_model(self):
        model = TBSFactorizedSingleViewModel(TinyBackbone(), 8, stage="s0")
        output = model(torch.randn(5, 3, 16, 16))
        self.assertEqual(tuple(output["diagnosis_probs"].shape), (5, 5))
        self.assertFalse(hasattr(model, "local_view"))
        self.assertFalse(hasattr(model, "view_gate"))

    def test_s1_diagnosis_is_composed_from_tbs_factors(self):
        model = TBSFactorizedSingleViewModel(
            TinyBackbone(), 8, semantic_dim=4, stage="s1"
        )
        output = model(torch.randn(5, 3, 16, 16))
        expected = torch.stack(
            [
                1.0 - output["screen_probs"],
                output["screen_probs"] * output["morph_probs"][:, 0] * output["evidence_probs"][:, 0],
                output["screen_probs"] * output["morph_probs"][:, 0] * output["evidence_probs"][:, 1],
                output["screen_probs"] * output["morph_probs"][:, 1] * output["evidence_probs"][:, 0],
                output["screen_probs"] * output["morph_probs"][:, 1] * output["evidence_probs"][:, 1],
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(output["diagnosis_probs"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(output["diagnosis_probs"].sum(1), torch.ones(5), atol=1e-6))

    def test_s1_loss_backpropagates_to_all_semantic_heads(self):
        model = TBSFactorizedSingleViewModel(
            TinyBackbone(), 8, semantic_dim=4, stage="s1"
        )
        diagnosis = torch.tensor([0, 1, 2, 3, 4])
        output = model(torch.randn(5, 3, 16, 16))
        losses = singleview_tbs_loss(
            output,
            {
                "diagnosis_labels": diagnosis,
                "screen_labels": (diagnosis > 0).float(),
                "morph_labels": torch.tensor([-1, 0, 0, 1, 1]),
                "evidence_labels": torch.tensor([-1, 0, 1, 0, 1]),
                "semantic_mask": diagnosis > 0,
            },
            stage="s1",
        )
        losses["loss"].backward()
        for head in (model.screen_head, model.morph_head, model.evidence_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
