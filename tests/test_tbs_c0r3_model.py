import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.c0r3_model import StableLocalCellViewGenerator


@unittest.skipIf(torch is None, "PyTorch is required for C0-R3 model tests")
class C0R3ModelTests(unittest.TestCase):
    def test_translation_hard_bound_is_locked_to_012(self):
        generator = StableLocalCellViewGenerator(translation_bound=0.12)
        with torch.no_grad():
            generator.affine[-1].bias[2] = 100.0
            generator.affine[-1].bias[5] = -100.0
        _, theta = generator(torch.randn(2, 3, 32, 32))
        self.assertLessEqual(float(theta[:, :, 2].abs().max()), 0.12 + 1e-6)


if __name__ == "__main__":
    unittest.main()
