import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_source_semantic_audit import (
    compute_group_metrics,
    inverse_member_transform,
    render_group_contact_sheet,
    select_stratified_samples,
)


class SemanticMetricTests(unittest.TestCase):
    def test_inverse_member_transform_restores_augmented_member(self):
        base = np.arange(20, dtype=np.uint8).reshape(4, 5)
        self.assertTrue(
            np.array_equal(
                inverse_member_transform(np.flipud(base), 1),
                base,
            )
        )
        self.assertTrue(
            np.array_equal(
                inverse_member_transform(np.fliplr(base), 2),
                base,
            )
        )
        self.assertTrue(
            np.array_equal(
                inverse_member_transform(np.flipud(np.fliplr(base)), 3),
                base,
            )
        )

    def test_compute_group_metrics_reports_geometry_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.zeros((20, 20), dtype=np.uint8)
            base[2:3, 3:5] = 1
            paths = []
            for member, array in {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }.items():
                path = root / f"cell_{member}.tif"
                Image.fromarray(array).save(path)
                paths.append(str(path))
            row = {
                "mask_key": "cell",
                "pair_status": "unique_geometry_valid",
                "source_structure": "unsuffixed",
                "member_paths": "|".join(paths),
                "source_paths": "",
                "mask_transform_min_match": 1.0,
            }
            metrics = compute_group_metrics(row)
            self.assertEqual(metrics["readable_member_count"], 4)
            self.assertAlmostEqual(metrics["member0_foreground_fraction"], 0.005)
            self.assertEqual(metrics["empty_member_count"], 0)
            self.assertIn("eligible_random", metrics["strata"].split("|"))
            self.assertIn("small_foreground", metrics["strata"].split("|"))

    def test_sampling_is_deterministic_and_reports_missing_strata(self):
        frame = pd.DataFrame(
            [
                {
                    "mask_key": key,
                    "pair_status": "unique_geometry_valid",
                    "mask_transform_min_match": value,
                    "strata": "eligible_random|lowest_transform_consistency",
                }
                for key, value in (("a", 1.0), ("b", 0.98), ("c", 0.99))
            ]
        )
        first, missing_first = select_stratified_samples(frame, per_stratum=2, seed=7)
        second, missing_second = select_stratified_samples(frame, per_stratum=2, seed=7)
        self.assertEqual(first["mask_key"].tolist(), second["mask_key"].tolist())
        self.assertEqual(first["selected_strata"].tolist(), second["selected_strata"].tolist())
        self.assertEqual(missing_first, missing_second)
        self.assertIn("empty_mask", missing_first)

    def test_contact_sheet_has_stable_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.zeros((8, 10), dtype=np.uint8)
            base[2:4, 3:6] = 1
            mask_paths = []
            for member, array in {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }.items():
                path = root / f"cell_{member}.tif"
                Image.fromarray(array).save(path)
                mask_paths.append(str(path))
            rgb_path = root / "cell.jpg"
            Image.new("RGB", (10, 8), (130, 80, 100)).save(rgb_path)
            row = {
                "mask_key": "cell",
                "pair_status": "unique_geometry_valid",
                "source_structure": "unsuffixed",
                "member_paths": "|".join(mask_paths),
                "source_paths": str(rgb_path),
            }
            output = root / "contact.png"
            render_group_contact_sheet(row, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (1024, 1052))


if __name__ == "__main__":
    unittest.main()
