import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiments.pretrain_xudata_simsiam import parse_args


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch integration runs on the RTX 5090 server")
class SimSiamModelTests(unittest.TestCase):
    def test_identical_normalized_vectors_have_zero_negative_cosine_loss(self):
        import torch

        from experiments.tbs.xudata_ssl import negative_cosine_similarity

        values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(float(negative_cosine_similarity(values, values)), -1.0)

    def test_simsiam_forward_has_finite_loss_and_expected_shapes(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.xudata_ssl import SimSiamModel

        model = SimSiamModel(
            nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16)),
            16,
            projection_dim=16,
            hidden_dim=32,
        )
        loss, first, second = model(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(first.shape), (2, 16))
        self.assertEqual(tuple(second.shape), (2, 16))
        self.assertTrue(torch.isfinite(loss))


class SimSiamCliTests(unittest.TestCase):
    def test_parser_rejects_forbidden_train_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--train_csv",
                        str(Path(tmp) / "test" / "train.csv"),
                        "--out_dir",
                        str(Path(tmp) / "out"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
