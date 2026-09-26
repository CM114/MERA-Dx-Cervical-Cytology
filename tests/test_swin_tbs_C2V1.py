import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from experiments.C2V1.model import C2V1BoundaryEvidenceModel, fixed_morphology_descriptors
from experiments.C2V1.training import _official_pair_macro_f1
from experiments.C2V1.transforms import C2V1DualResolutionTransform, morphology_centroid


class DummyC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        b = x.shape[0]
        p = torch.tensor([[.10, .25, .15, .20, .30]], device=x.device).repeat(b, 1)
        q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {"p_final": p, "q_final": q}


class DummyLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(torch.nn.functional.avg_pool2d(x, 8))


class C2V1Tests(unittest.TestCase):
    def _model(self):
        return C2V1BoundaryEvidenceModel(
            DummyC1(), DummyLocal(), 8,
            local_projected_dim=16, descriptor_dim=4, pair_hidden_dim=8,
            pair_logodds_shift_bound=1.0,
        )

    def test_zero_init_is_exact_c1_and_c1_frozen(self):
        model = self._model()
        out = model(torch.randn(4, 3, 16, 16), torch.randn(4, 3, 32, 32))
        self.assertTrue(torch.allclose(out["p_c1"], out["p_c2"], atol=1e-7))
        self.assertFalse(any(p.requires_grad for p in model.c1.parameters()))

    def test_family_and_screen_mass_are_exact_after_shift(self):
        model = self._model()
        with torch.no_grad():
            model.low_head.net[-1].bias.fill_(0.6)
            model.high_head.net[-1].bias.fill_(-0.5)
        out = model(torch.randn(3, 3, 16, 16), torch.randn(3, 3, 32, 32))
        self.assertLessEqual(float(out["screen_mass_error_c2"]), 1e-6)
        self.assertLessEqual(float(out["low_family_mass_error"]), 1e-6)
        self.assertLessEqual(float(out["high_family_mass_error"]), 1e-6)
        self.assertGreater(float(out["actual_max_shift"]), 0.0)

    def test_official_pair_metric_is_pair_local(self):
        labels = np.array([1, 2, 0, 3, 4])
        p = np.array([
            [.60, .20, .19, .005, .005],
            [.60, .19, .20, .005, .005],
            [.99, .0025, .0025, .0025, .0025],
            [.60, .005, .005, .20, .19],
            [.60, .005, .005, .19, .20],
        ])
        self.assertAlmostEqual(_official_pair_macro_f1(labels, p, (1, 2)), 1.0)
        self.assertAlmostEqual(_official_pair_macro_f1(labels, p, (3, 4)), 1.0)

    def test_morphology_centroid_is_clamped(self):
        arr = np.full((200, 300, 3), 255, dtype=np.uint8)
        arr[:30, :30] = 0
        cx, cy, disp = morphology_centroid(Image.fromarray(arr), max_center_shift=.20)
        self.assertGreaterEqual(cx, 0.30 * 300)
        self.assertLessEqual(cx, 0.70 * 300)
        self.assertGreaterEqual(cy, 0.30 * 200)
        self.assertLessEqual(cy, 0.70 * 200)
        self.assertTrue(np.isfinite(disp))

    def test_dual_transform_retains_native_metadata(self):
        image = Image.fromarray(np.full((400, 500, 3), 180, dtype=np.uint8))
        transform = C2V1DualResolutionTransform(
            lambda x: torch.zeros(3, 224, 224), local_size=320, crop_fraction=.65, train=False
        )
        out = transform(image)
        self.assertEqual(tuple(out["global"].shape), (3, 224, 224))
        self.assertEqual(tuple(out["local"].shape), (3, 320, 320))
        self.assertEqual(out["raw_min_side"], 400.0)
        self.assertEqual(out["crop_native_side"], 260.0)

    def test_morphology_descriptors_are_finite(self):
        d = fixed_morphology_descriptors(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(d.shape), (2, 8))
        self.assertTrue(torch.isfinite(d).all())


if __name__ == "__main__":
    unittest.main()
