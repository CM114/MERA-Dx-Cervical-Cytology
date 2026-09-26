import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from experiments.C2V2.data import parse_subtype_from_path, resolve_to_data_root
from experiments.C2V2.model import C2V2ExplicitMorphologyModel
from experiments.C2V2.training import _official_pair_macro_f1
from experiments.C2V2.transforms import (
    MORPHOLOGY_DESCRIPTOR_DIM,
    MORPHOLOGY_FEATURE_NAMES,
    SIZE_ROBUST_MORPHOLOGY_DESCRIPTOR_DIM,
    SIZE_ROBUST_MORPHOLOGY_FEATURE_NAMES,
    SIZE_EXCLUDED_MORPHOLOGY_FEATURE_NAMES,
    select_size_robust_morphology_descriptors,
    C2V2MorphologyTransform,
    extract_soft_morphology_descriptors,
)


class DummyC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        b = x.shape[0]
        p = torch.tensor([[.10, .25, .15, .20, .30]], device=x.device).repeat(b, 1)
        q = p[:, 1:] / p[:, 1:].sum(1, keepdim=True)
        return {"p_final": p, "q_final": q}


class C2V2Tests(unittest.TestCase):
    def _model(self):
        return C2V2ExplicitMorphologyModel(
            DummyC1(),
            descriptor_hidden_dim=16,
            descriptor_embedding_dim=8,
            pair_hidden_dim=8,
            pair_logodds_shift_bound=1.0,
        )

    def test_zero_init_is_exact_c1_and_c1_frozen(self):
        model = self._model()
        out = model(torch.randn(4, 3, 16, 16), torch.randn(4, SIZE_ROBUST_MORPHOLOGY_DESCRIPTOR_DIM))
        self.assertTrue(torch.allclose(out["p_c1"], out["p_c2"], atol=1e-7))
        self.assertFalse(any(p.requires_grad for p in model.c1.parameters()))

    def test_family_and_screen_mass_are_exact_after_shift(self):
        model = self._model()
        with torch.no_grad():
            model.low_head.net[-1].bias.fill_(0.6)
            model.high_head.net[-1].bias.fill_(-0.5)
        out = model(torch.randn(3, 3, 16, 16), torch.randn(3, SIZE_ROBUST_MORPHOLOGY_DESCRIPTOR_DIM))
        self.assertLessEqual(float(out["screen_mass_error_c2"]), 1e-6)
        self.assertLessEqual(float(out["low_family_mass_error"]), 1e-6)
        self.assertLessEqual(float(out["high_family_mass_error"]), 1e-6)
        self.assertGreater(float(out["actual_max_shift"]), 0.0)

    def test_descriptor_shape_finite_and_named(self):
        image = Image.new("RGB", (90, 70), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 18, 62, 52), fill=(80, 30, 90))
        d, audit = extract_soft_morphology_descriptors(image, return_audit=True)
        self.assertEqual(tuple(d.shape), (32,))
        self.assertEqual(len(MORPHOLOGY_FEATURE_NAMES), 32)
        robust = select_size_robust_morphology_descriptors(d)
        self.assertEqual(tuple(robust.shape), (21,))
        self.assertEqual(len(SIZE_ROBUST_MORPHOLOGY_FEATURE_NAMES), 21)
        self.assertTrue(set(SIZE_ROBUST_MORPHOLOGY_FEATURE_NAMES).isdisjoint(SIZE_EXCLUDED_MORPHOLOGY_FEATURE_NAMES))
        self.assertTrue(torch.isfinite(d).all())
        self.assertGreater(audit["soft_nuclear_area_ratio"], 0.0)
        self.assertLess(audit["soft_nuclear_area_ratio"], 1.0)

    def test_descriptor_is_native_resolution_normalized(self):
        base = Image.new("RGB", (160, 160), "white")
        draw = ImageDraw.Draw(base)
        draw.ellipse((45, 35, 115, 125), fill=(95, 45, 120))
        small = base.resize((64, 64), Image.Resampling.BILINEAR)
        large = base.resize((256, 256), Image.Resampling.BILINEAR)
        d1 = extract_soft_morphology_descriptors(small).numpy()
        d2 = extract_soft_morphology_descriptors(large).numpy()
        self.assertLess(float(np.mean(np.abs(d1 - d2))), 0.04)

    def test_transform_keeps_native_size_audit_only(self):
        image = Image.new("RGB", (73, 91), "white")
        transform = C2V2MorphologyTransform(lambda x: torch.zeros(3, 224, 224), analysis_size=128)
        out = transform(image)
        self.assertEqual(tuple(out["global"].shape), (3, 224, 224))
        self.assertEqual(tuple(out["morphology"].shape), (21,))
        self.assertEqual(out["raw_min_side"], 73.0)

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

    def test_subtype_parser_is_audit_only_helper(self):
        self.assertEqual(parse_subtype_from_path("/x/train/5_HSIL/2_Parabasal/a.jpg"), 2)
        self.assertEqual(parse_subtype_from_path("/x/train/1_ASC-US/0_Superficial/a.jpg"), 0)
        self.assertEqual(parse_subtype_from_path("/x/unknown/a.jpg"), -1)

    def test_size_robust_mask_is_locked_and_excludes_high_risk_features(self):
        self.assertEqual(SIZE_ROBUST_MORPHOLOGY_DESCRIPTOR_DIM, 21)
        for name in (
            "soft_boundary_energy", "soft_boundary_p90", "nuclear_luminance",
            "nuclear_optical_density", "cytoplasm_weighted_gradient",
        ):
            self.assertNotIn(name, SIZE_ROBUST_MORPHOLOGY_FEATURE_NAMES)
        for name in (
            "soft_nc_ratio_proxy", "soft_nuclear_area_ratio", "soft_anisotropy",
            "nucleus_minus_cytoplasm_purple_score",
        ):
            self.assertIn(name, SIZE_ROBUST_MORPHOLOGY_FEATURE_NAMES)

    def test_data_root_resolver_preserves_xudata_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "xudata"
            target = root / "train/1_ASC-US/0_Superficial/a.jpg"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x")
            old = Path("/old/server/xudata/train/1_ASC-US/0_Superficial/a.jpg")
            resolved = resolve_to_data_root(old, root)
            self.assertEqual(resolved, target.resolve())


if __name__ == "__main__":
    unittest.main()
