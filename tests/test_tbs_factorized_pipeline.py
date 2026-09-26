import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from experiments.tbs.factorized_losses import (
    factorized_tbs_loss,
    localizer_geometry_loss,
)
from experiments.xudata_gain_common import compute_locked_candidate_metrics
from experiments.train_tbs_factorized import _build_optimizer
from experiments.tbs.factorized_model import TBSFactorizedDualSpaceModel


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


def semantic_targets():
    diagnosis = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    return {
        "diagnosis_labels": diagnosis,
        "screen_labels": (diagnosis > 0).float(),
        "morph_labels": torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long),
        "evidence_labels": torch.tensor([-1, 0, 1, 0, 1], dtype=torch.long),
        "semantic_mask": diagnosis > 0,
    }


class FactorizedModelTests(unittest.TestCase):
    def test_optimizer_uses_conservative_backbone_learning_rate(self):
        model = TBSFactorizedDualSpaceModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, stage="c1"
        )
        optimizer = _build_optimizer(
            model,
            SimpleNamespace(
                lr=1e-4,
                backbone_lr_multiplier=0.1,
                weight_decay=1e-4,
            ),
        )
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(rates["backbone"], 1e-5)
        self.assertAlmostEqual(rates["task"], 1e-4)

    def test_locked_evaluation_reports_both_tbs_boundary_pairs(self):
        metrics = compute_locked_candidate_metrics(
            np.array([0, 1, 2, 3, 4]),
            np.eye(5),
        )
        self.assertEqual(metrics["low_grade_pair_macro_f1"], 1.0)
        self.assertEqual(metrics["high_grade_pair_macro_f1"], 1.0)

    def test_c1_probabilities_are_explicit_tbs_factor_combinations(self):
        model = TBSFactorizedDualSpaceModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, stage="c1"
        )
        output = model(torch.randn(5, 3, 16, 16))

        self.assertEqual(tuple(output["diagnosis_probs"].shape), (5, 5))
        self.assertTrue(
            torch.allclose(
                output["diagnosis_probs"].sum(dim=1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        expected = torch.stack(
            [
                1.0 - output["screen_probs"],
                output["screen_probs"]
                * output["morph_probs"][:, 0]
                * output["evidence_probs"][:, 0],
                output["screen_probs"]
                * output["morph_probs"][:, 0]
                * output["evidence_probs"][:, 1],
                output["screen_probs"]
                * output["morph_probs"][:, 1]
                * output["evidence_probs"][:, 0],
                output["screen_probs"]
                * output["morph_probs"][:, 1]
                * output["evidence_probs"][:, 1],
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(output["diagnosis_probs"], expected, atol=1e-6))

    def test_c2_has_separate_morphology_and_evidence_prototypes(self):
        model = TBSFactorizedDualSpaceModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, stage="c2"
        )
        output = model(torch.randn(5, 3, 16, 16))

        self.assertEqual(tuple(model.morph_prototypes.shape), (2, 4))
        self.assertEqual(tuple(model.evidence_prototypes.shape), (2, 4))
        self.assertIn("morph_prototype_logits", output)
        self.assertIn("evidence_prototype_logits", output)

    def test_c2_prototypes_can_be_initialized_from_factor_centroids(self):
        model = TBSFactorizedDualSpaceModel(
            TinyBackbone(), feature_dim=8, semantic_dim=2, stage="c2"
        )
        morph_features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]]
        )
        evidence_features = torch.tensor(
            [[0.0, 1.0], [0.1, 0.9], [0.0, -1.0], [0.1, -0.9]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        model.initialize_factor_prototypes(
            morph_features, labels, evidence_features, labels
        )
        self.assertGreater(float(model.morph_prototypes[0, 0]), 0.9)
        self.assertLess(float(model.morph_prototypes[1, 0]), -0.9)
        self.assertGreater(float(model.evidence_prototypes[0, 1]), 0.9)
        self.assertLess(float(model.evidence_prototypes[1, 1]), -0.9)

    def test_factorized_loss_trains_screen_and_both_prototype_spaces(self):
        model = TBSFactorizedDualSpaceModel(
            TinyBackbone(), feature_dim=8, semantic_dim=4, stage="c2"
        )
        output = model(torch.randn(5, 3, 16, 16))
        losses = factorized_tbs_loss(output, semantic_targets())
        losses["loss"].backward()

        self.assertIsNotNone(model.screen_head.weight.grad)
        self.assertGreater(float(model.screen_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.morph_prototypes.grad)
        self.assertGreater(float(model.morph_prototypes.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.evidence_prototypes.grad)
        self.assertGreater(float(model.evidence_prototypes.grad.abs().sum()), 0.0)

    def test_localizer_geometry_loss_penalizes_nonlocal_or_invalid_transforms(self):
        identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
        )
        local = torch.tensor(
            [[[0.55, 0.12, 0.0], [0.04, 0.55, 0.0]]], dtype=torch.float32
        )
        self.assertGreater(float(localizer_geometry_loss(identity)), 0.0)
        self.assertLess(float(localizer_geometry_loss(local)), float(localizer_geometry_loss(identity)))

    def test_localizer_geometry_loss_penalizes_reflection(self):
        valid = torch.tensor(
            [[[0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]], dtype=torch.float32
        )
        reflected = torch.tensor(
            [[[-0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]], dtype=torch.float32
        )
        self.assertGreater(
            float(localizer_geometry_loss(reflected)),
            float(localizer_geometry_loss(valid)),
        )


if __name__ == "__main__":
    unittest.main()
