import unittest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - local CPU-only authoring env
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for C2-R1 tensor tests")
class SwinTbsC2R1Tests(unittest.TestCase):
    def test_true_factor_weight_uses_thresholded_true_probability(self):
        from experiments.C2.loss import core_weight_from_true_probability

        probabilities = torch.tensor(
            [[0.40, 0.60], [0.50, 0.50], [0.75, 0.25], [1.00, 0.00]],
            dtype=torch.float32,
        )
        labels = torch.zeros(4, dtype=torch.long)
        got = core_weight_from_true_probability(probabilities, labels)
        expected = torch.tensor([0.25, 0.25, 0.625, 1.0])
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))
        self.assertFalse(got.requires_grad)

    def test_epoch_level_ema_normalizes_and_uses_single_update(self):
        from experiments.C2.model import epoch_level_ema_update

        previous = torch.tensor([1.0, 0.0])
        weighted_sum = torch.tensor([0.0, 2.0])
        updated, valid = epoch_level_ema_update(
            previous, weighted_sum, torch.tensor(2.0), momentum=0.95, valid=True
        )
        expected = torch.tensor([0.95, 0.05])
        expected = expected / expected.norm()
        self.assertTrue(valid)
        self.assertTrue(torch.allclose(updated, expected, atol=1e-6))
        self.assertAlmostEqual(float(updated.norm()), 1.0, places=6)

    def test_lambda_zero_c2_is_c1_forward_and_loss_equivalent(self):
        from experiments.C1R2.loss import c1r2_loss
        from experiments.C1R2.model import C1R2Model
        from experiments.C2.loss import c2r1_loss
        from experiments.C2.model import C2R1Model

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3 * 4 * 4, 8)

            def forward(self, images):
                return self.projection(images.flatten(1))

        torch.manual_seed(42)
        c1 = C1R2Model(TinyBackbone(), 8, semantic_dim=4, interaction_dim=2)
        torch.manual_seed(43)
        c2 = C2R1Model(TinyBackbone(), 8, semantic_dim=4, interaction_dim=2)
        c2.load_state_dict(c1.state_dict(), strict=False)
        images = torch.randn(2, 3, 4, 4)
        diagnosis = torch.tensor([1, 2])
        morph = torch.tensor([0, 0])
        evidence = torch.tensor([0, 1])
        mask = torch.tensor([True, True])
        out1 = c1(images)
        out2 = c2(images)
        self.assertTrue(torch.allclose(out1["p_final"], out2["p_final"], atol=1e-6))
        base = c1r2_loss(out1, diagnosis, morph, evidence, mask)
        candidate = c2r1_loss(
            out2, diagnosis, morph, evidence, mask, c2, epoch=1,
            lambda_proto_morph=0.0, lambda_proto_evidence=0.0,
        )
        differences = {
            key: float((out1[key] - out2[key]).abs().max().item())
            for key in (
                "features", "logits_b0", "morph_features", "evidence_features",
                "morph_logits", "evidence_logits", "eta", "joint_logits",
                "final_log_probs", "p_final",
            )
        }
        terms = {
            key: (float(base[key].detach().item()), float(candidate[key].detach().item()))
            for key in ("loss", "five_class_loss", "joint_loss", "morph_loss", "evidence_loss", "eta_reg_loss")
        }
        self.assertTrue(
            torch.allclose(base["loss"], candidate["loss"], atol=1e-6),
            msg=f"C1/C2 lambda-zero mismatch; tensor_diffs={differences}; loss_terms={terms}",
        )
        c1.zero_grad(set_to_none=True)
        c2.zero_grad(set_to_none=True)
        base["loss"].backward()
        candidate["loss"].backward()
        c2_parameters = dict(c2.named_parameters())
        for name, parameter in c1.named_parameters():
            self.assertIn(name, c2_parameters)
            self.assertIsNotNone(parameter.grad, msg=f"missing C1 gradient for {name}")
            self.assertIsNotNone(c2_parameters[name].grad, msg=f"missing C2 gradient for {name}")
            self.assertTrue(
                torch.allclose(parameter.grad, c2_parameters[name].grad, atol=1e-6, rtol=1e-6),
                msg=f"C1/C2 lambda-zero gradient mismatch for {name}",
            )

    def test_prototype_perturbation_does_not_change_final_diagnosis_output(self):
        from experiments.C2.model import C2R1Model

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3 * 4 * 4, 8)

            def forward(self, images):
                return self.projection(images.flatten(1))

        torch.manual_seed(7)
        model = C2R1Model(TinyBackbone(), 8, semantic_dim=4, interaction_dim=2)
        model.eval()
        images = torch.randn(3, 3, 4, 4)
        before = model(images)["p_final"].detach().clone()
        with torch.no_grad():
            model.morph_prototypes.normal_()
            model.evidence_prototypes.normal_()
        after = model(images)["p_final"].detach()
        self.assertTrue(
            torch.allclose(before, after, atol=1e-7, rtol=0.0),
            msg="prototype buffers unexpectedly changed the C1-R2 final diagnosis output",
        )

    def test_normal_samples_never_update_prototypes(self):
        from experiments.C2.model import C2R1Model, finalize_epoch_prototypes

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3 * 4 * 4, 8)

            def forward(self, images):
                return self.projection(images.flatten(1))

        model = C2R1Model(TinyBackbone(), 8, semantic_dim=4, interaction_dim=2)
        output = {
            "morph_features": torch.randn(3, 4),
            "evidence_features": torch.randn(3, 4),
            "morph_probs": torch.full((3, 2), 0.5),
            "evidence_probs": torch.full((3, 2), 0.5),
        }
        labels = torch.zeros(3, dtype=torch.long)
        invalid_factor_labels = torch.full((3,), -1, dtype=torch.long)
        semantic_mask = torch.zeros(3, dtype=torch.bool)
        model.accumulate_epoch_statistics(
            output, invalid_factor_labels, invalid_factor_labels, semantic_mask
        )
        finalize_epoch_prototypes(model)
        self.assertEqual(int(model.normal_proto_update_count.item()), 0)
        self.assertEqual(int(model.morph_update_count.sum().item()), 0)
        self.assertFalse(bool(model.morph_valid.any()))

    def test_prototypes_are_buffers_not_optimizer_parameters(self):
        from experiments.C2.model import C2R1Model

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3 * 4 * 4, 8)

            def forward(self, images):
                return self.projection(images.flatten(1))

        model = C2R1Model(TinyBackbone(), 8, semantic_dim=4, interaction_dim=2)
        parameter_names = {name for name, _ in model.named_parameters()}
        self.assertNotIn("morph_prototypes", parameter_names)
        self.assertNotIn("evidence_prototypes", parameter_names)


if __name__ == "__main__":
    unittest.main()
