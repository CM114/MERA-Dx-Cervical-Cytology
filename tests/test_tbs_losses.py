import unittest

import torch
import torch.nn.functional as F

from experiments.tbs.losses import pair_boundary_supcon_loss


class PairBoundarySupConTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_separated_pairs_have_lower_loss_than_mixed_pairs(self):
        separated = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0], [0.9, 0.1],
            [-1.0, 0.0], [-0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
            [0.0, -1.0], [0.1, -0.9],
        ])
        mixed = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0], [-1.0, 0.0],
            [0.9, 0.1], [-0.9, 0.1],
            [0.0, 1.0], [0.0, -1.0],
            [0.1, 0.9], [0.1, -0.9],
        ])
        clean = pair_boundary_supcon_loss(separated, self.labels, 0.1)
        confused = pair_boundary_supcon_loss(mixed, self.labels, 0.1)
        self.assertLess(float(clean.loss), float(confused.loss))

    def test_normal_features_do_not_change_loss(self):
        features = torch.randn(9, 4)
        first = pair_boundary_supcon_loss(features, self.labels, 0.1)
        changed = features.clone()
        changed[0] = torch.tensor([100.0, -100.0, 50.0, -50.0])
        second = pair_boundary_supcon_loss(changed, self.labels, 0.1)
        self.assertTrue(torch.allclose(first.loss, second.loss, atol=1e-7))

    def test_total_is_equal_mean_of_pair_losses(self):
        result = pair_boundary_supcon_loss(torch.randn(9, 8), self.labels, 0.1)
        self.assertTrue(
            torch.allclose(result.loss, (result.low_loss + result.high_loss) / 2)
        )

    def test_partial_valid_anchor_coverage_reports_exact_counts(self):
        result = pair_boundary_supcon_loss(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]]),
            torch.tensor([1, 1, 2]), 0.1,
        )
        self.assertEqual((result.low_candidates, result.low_valid), (3, 2))
        self.assertEqual(result.low_coverage, 2 / 3)
        self.assertEqual((result.high_candidates, result.high_valid), (0, 0))
        self.assertEqual(result.high_coverage, 0.0)

    def test_fixed_equal_weighting_with_imbalanced_pair_sizes(self):
        features = torch.tensor([
            [1.0, 0.0], [0.99, 0.1], [0.98, -0.1], [0.97, 0.05],
            [-1.0, 0.0], [-0.99, 0.1],
            [0.0, 1.0], [0.1, 0.99], [0.3, 0.95], [0.4, 0.9],
        ])
        labels = torch.tensor([1, 1, 1, 1, 2, 2, 3, 3, 4, 4])
        temperature = 0.1
        result = pair_boundary_supcon_loss(features, labels, temperature)
        reference_low = self._reference_pair_loss(
            features, labels, (1, 2), temperature
        )
        reference_high = self._reference_pair_loss(
            features, labels, (3, 4), temperature
        )
        expected = (reference_low + reference_high) / 2.0
        weighted_by_candidates = (reference_low * 6 + reference_high * 4) / 10.0

        self.assertFalse(torch.allclose(reference_low, reference_high))
        self.assertTrue(torch.allclose(result.loss, expected, atol=1e-7))
        self.assertFalse(torch.allclose(result.loss, weighted_by_candidates))

    @staticmethod
    def _reference_pair_loss(features, labels, pair_labels, temperature):
        normalized = F.normalize(features.float(), p=2, dim=1)
        indexes = [
            index
            for index, label in enumerate(labels.tolist())
            if label in pair_labels
        ]
        per_anchor = []
        for anchor in indexes:
            positives = [
                other
                for other in indexes
                if other != anchor and labels[other] == labels[anchor]
            ]
            negatives = [
                other for other in indexes if labels[other] != labels[anchor]
            ]
            if not positives or not negatives:
                continue
            logits = torch.stack([
                normalized[anchor].dot(normalized[other]) / temperature
                for other in indexes
                if other != anchor
            ])
            denominator = torch.logsumexp(logits, dim=0)
            positive_log_probs = torch.stack([
                normalized[anchor].dot(normalized[other]) / temperature
                - denominator
                for other in positives
            ])
            per_anchor.append(-positive_log_probs.mean())
        return torch.stack(per_anchor).mean()

    def test_invalid_anchors_return_graph_connected_zero(self):
        features = torch.randn(3, 5, requires_grad=True)
        result = pair_boundary_supcon_loss(
            features, torch.tensor([0, 1, 2]), 0.1
        )
        result.loss.backward()
        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features.grad)))

    def test_no_positive_samples_returns_graph_connected_zero(self):
        features = torch.randn(4, 5, requires_grad=True)
        result = pair_boundary_supcon_loss(
            features, torch.tensor([1, 2, 3, 4]), 0.1
        )
        self.assertEqual(result.low_candidates, 2)
        self.assertEqual(result.high_candidates, 2)
        self.assertEqual(result.low_valid, 0)
        self.assertEqual(result.high_valid, 0)
        self.assertEqual(result.low_coverage, 0.0)
        self.assertEqual(result.high_coverage, 0.0)
        self.assertTrue(torch.isfinite(result.low_loss))
        self.assertTrue(torch.isfinite(result.high_loss))
        self.assertTrue(result.low_loss.requires_grad)
        self.assertTrue(result.high_loss.requires_grad)
        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features.grad)))

    def test_no_negative_samples_returns_graph_connected_zero(self):
        features = torch.randn(4, 5, requires_grad=True)
        result = pair_boundary_supcon_loss(
            features, torch.tensor([1, 1, 3, 3]), 0.1
        )
        self.assertEqual(result.low_valid, 0)
        self.assertEqual(result.high_valid, 0)
        self.assertEqual(result.low_coverage, 0.0)
        self.assertEqual(result.high_coverage, 0.0)
        self.assertEqual(float(result.low_loss.detach()), 0.0)
        self.assertEqual(float(result.high_loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(result.low_loss))
        self.assertTrue(torch.isfinite(result.high_loss))
        self.assertTrue(result.low_loss.requires_grad)
        self.assertTrue(result.high_loss.requires_grad)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features.grad)))

    def test_missing_boundary_is_zero_but_valid_boundary_keeps_fixed_half_weight(self):
        features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1], [0.0, 1.0],
        ], requires_grad=True)
        result = pair_boundary_supcon_loss(
            features, torch.tensor([1, 1, 2, 2, 0]), 0.1
        )
        self.assertEqual(result.high_candidates, 0)
        self.assertEqual(result.high_valid, 0)
        self.assertEqual(result.high_coverage, 0.0)
        self.assertEqual(float(result.high_loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(result.high_loss))
        self.assertTrue(result.high_loss.requires_grad)
        self.assertGreater(result.low_valid, 0)
        self.assertEqual(result.low_coverage, 1.0)
        self.assertTrue(torch.allclose(result.loss, result.low_loss / 2.0))

    def test_normal_only_batch_returns_graph_connected_zero(self):
        features = torch.randn(3, 5, requires_grad=True)
        result = pair_boundary_supcon_loss(features, torch.tensor([0, 0, 0]), 0.1)
        self.assertEqual(result.low_candidates, 0)
        self.assertEqual(result.high_candidates, 0)
        self.assertEqual(result.low_valid, 0)
        self.assertEqual(result.high_valid, 0)
        self.assertEqual(result.low_coverage, 0.0)
        self.assertEqual(result.high_coverage, 0.0)
        self.assertEqual(float(result.low_loss.detach()), 0.0)
        self.assertEqual(float(result.high_loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(result.low_loss))
        self.assertTrue(torch.isfinite(result.high_loss))
        self.assertTrue(result.low_loss.requires_grad)
        self.assertTrue(result.high_loss.requires_grad)
        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features.grad)))

    def test_half_precision_inputs_are_promoted_and_finite(self):
        result = pair_boundary_supcon_loss(
            torch.randn(8, 6, dtype=torch.float16),
            torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]),
            0.1,
        )
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result.loss))

    def test_normal_nan_is_rejected_before_pair_masking(self):
        features = torch.randn(9, 4)
        features[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "features.*finite"):
            pair_boundary_supcon_loss(features, self.labels, 0.1)

    def test_valid_pair_loss_produces_finite_nonzero_gradient(self):
        features = torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, 0.2],
            [0.0, 1.0], [0.2, 0.8], [0.0, -1.0], [0.2, -0.8],
        ], requires_grad=True)
        result = pair_boundary_supcon_loss(
            features, torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]), 0.1
        )
        result.loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            pair_boundary_supcon_loss(torch.randn(4), torch.tensor([1, 1, 2, 2]))
        with self.assertRaisesRegex(ValueError, "batch size"):
            pair_boundary_supcon_loss(torch.randn(4, 3), torch.tensor([1, 2]))
        with self.assertRaisesRegex(ValueError, "labels must be in"):
            pair_boundary_supcon_loss(torch.randn(4, 3), torch.tensor([1, 1, 2, 5]))
        with self.assertRaisesRegex(ValueError, "integer"):
            pair_boundary_supcon_loss(
                torch.randn(4, 3), torch.tensor([1.0, 1.0, 2.0, 2.0])
            )
        with self.assertRaisesRegex(ValueError, "same device"):
            pair_boundary_supcon_loss(
                torch.randn(4, 3),
                torch.empty(4, dtype=torch.long, device="meta"),
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            pair_boundary_supcon_loss(
                torch.randn(4, 3), torch.tensor([1, 1, 2, 2]), 0.0
            )
