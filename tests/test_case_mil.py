import importlib.util
import unittest

import numpy as np

from experiments.tbs.case_mil import CaseFeatureBags, pool_case_probabilities


class CaseBagTests(unittest.TestCase):
    def test_groups_patches_and_rejects_case_label_conflicts(self):
        features = np.arange(20, dtype=np.float32).reshape(5, 4)
        bags = CaseFeatureBags(
            features,
            labels=np.array([0, 0, 4, 4, 4]),
            case_ids=np.array(["a", "a", "b", "b", "b"]),
            image_paths=np.array([f"{i}.jpg" for i in range(5)]),
        )
        self.assertEqual(len(bags), 2)
        self.assertEqual(bags.case_ids, ("a", "b"))
        self.assertEqual(bags.get_case(1, train=False)["features"].shape, (3, 4))
        with self.assertRaisesRegex(ValueError, "multiple labels"):
            CaseFeatureBags(
                features[:2],
                labels=np.array([0, 1]),
                case_ids=np.array(["x", "x"]),
                image_paths=np.array(["x0", "x1"]),
            )

    def test_training_sampling_is_deterministic_and_dev_uses_all_patches(self):
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        bags = CaseFeatureBags(
            features,
            labels=np.zeros(6, dtype=np.int64),
            case_ids=np.array(["a"] * 6),
            image_paths=np.array([str(i) for i in range(6)]),
            max_patches=3,
            seed=42,
        )
        first = bags.get_case(0, train=True, epoch=2)
        second = bags.get_case(0, train=True, epoch=2)
        self.assertEqual(first["image_paths"].tolist(), second["image_paths"].tolist())
        self.assertEqual(first["features"].shape[0], 3)
        self.assertEqual(bags.get_case(0, train=False)["features"].shape[0], 6)


class PoolingTests(unittest.TestCase):
    def test_topk_pooling_normalizes_and_selects_highest_evidence(self):
        probabilities = np.array(
            [
                [0.90, 0.05, 0.03, 0.01, 0.01],
                [0.10, 0.10, 0.10, 0.60, 0.10],
                [0.10, 0.10, 0.10, 0.20, 0.50],
            ],
            dtype=np.float64,
        )
        pooled = pool_case_probabilities(probabilities, method="topk", topk_fraction=2 / 3)
        self.assertTrue(np.allclose(pooled.sum(), 1.0))
        self.assertEqual(int(pooled.argmax()), 3)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model tests")
class AttentionModelTests(unittest.TestCase):
    def test_collate_pads_bags_and_marks_valid_patches(self):
        import torch
        from experiments.tbs.case_mil import case_mil_collate

        batch = case_mil_collate(
            [
                {
                    "features": np.ones((2, 4), dtype=np.float32),
                    "label": 0,
                    "case_id": "a",
                    "image_paths": np.array(["a0", "a1"]),
                    "patch_count": 2,
                },
                {
                    "features": np.ones((3, 4), dtype=np.float32),
                    "label": 4,
                    "case_id": "b",
                    "image_paths": np.array(["b0", "b1", "b2"]),
                    "patch_count": 3,
                },
            ]
        )
        self.assertEqual(tuple(batch["features"].shape), (2, 3, 4))
        self.assertTrue(torch.equal(batch["mask"].sum(dim=1), torch.tensor([2, 3])))

    def test_attention_respects_padding_mask_and_outputs_five_classes(self):
        import torch
        from experiments.tbs.case_mil import GatedAttentionMIL

        model = GatedAttentionMIL(feature_dim=4, hidden_dim=8)
        features = torch.randn(2, 5, 4)
        mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, True]])
        output = model(features, mask)
        self.assertEqual(tuple(output["logits"].shape), (2, 5))
        self.assertTrue(torch.allclose(output["probs"].sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(output["attention"][0, 2:], torch.zeros(3), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
