import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_instance_audit import (
    analyze_label_group,
    analyze_foreground_transform_consistency,
    analyze_instance_geometry_transform_consistency,
    analyze_member_transform_consistency,
    render_instance_contact_sheet,
    select_instance_samples,
)


class InstanceMetricTests(unittest.TestCase):
    @staticmethod
    def base_labels():
        base = np.zeros((12, 14), dtype=np.uint16)
        base[2:5, 2:5] = 1
        base[7:10, 8:12] = 2
        return base

    def test_label_count_matches_foreground_components_for_instance_like_map(self):
        base = self.base_labels()
        arrays = {
            0: base,
            1: np.flipud(base),
            2: np.fliplr(base),
            3: np.flipud(np.fliplr(base)),
        }
        result = analyze_label_group(arrays)
        self.assertEqual(result["member0_nonzero_label_count"], 2)
        self.assertEqual(result["member0_foreground_component_count"], 2)
        self.assertEqual(result["member0_labels_with_multiple_components"], 0)
        self.assertTrue(result["member0_label_component_match"])

    def test_repeated_disconnected_label_is_flagged(self):
        base = np.zeros((8, 8), dtype=np.uint16)
        base[1:3, 1:3] = 1
        base[5:7, 5:7] = 1
        arrays = {member: base.copy() for member in range(4)}
        result = analyze_label_group(arrays)
        self.assertEqual(result["member0_nonzero_label_count"], 1)
        self.assertEqual(result["member0_foreground_component_count"], 2)
        self.assertEqual(result["member0_labels_with_multiple_components"], 1)
        self.assertFalse(result["member0_label_component_match"])

    def test_raw_label_transform_consistency_requires_exact_ids(self):
        base = self.base_labels()
        arrays = {
            0: base,
            1: np.flipud(base),
            2: np.fliplr(base),
            3: np.flipud(np.fliplr(base)),
        }
        result = analyze_member_transform_consistency(arrays)
        self.assertEqual(result["label_transform_min_pixel_agreement"], 1.0)
        self.assertTrue(result["label_id_set_consistent"])
        changed = dict(arrays)
        changed[2] = changed[2].copy()
        changed[2][0, 0] = 7
        changed_result = analyze_member_transform_consistency(changed)
        self.assertLess(changed_result["label_transform_min_pixel_agreement"], 1.0)

    def test_instance_geometry_ignores_member_specific_label_renumbering(self):
        base = self.base_labels()

        def remap(array, mapping):
            output = np.zeros_like(array)
            for old, new in mapping.items():
                output[array == old] = new
            return output

        arrays = {
            0: base,
            1: remap(np.flipud(base), {1: 101, 2: 102}),
            2: remap(np.fliplr(base), {1: 201, 2: 202}),
            3: remap(np.flipud(np.fliplr(base)), {1: 301, 2: 302}),
        }
        foreground = analyze_foreground_transform_consistency(arrays)
        geometry = analyze_instance_geometry_transform_consistency(arrays)
        self.assertTrue(foreground["foreground_transform_exact"])
        self.assertTrue(geometry["instance_region_transform_exact"])
        self.assertEqual(geometry["instance_region_count_member0"], 2)
        self.assertEqual(geometry["instance_region_transform_min_iou"], 1.0)

    def test_instance_geometry_flags_split_region_after_transform(self):
        base = np.zeros((8, 10), dtype=np.uint16)
        base[2:6, 2:6] = 1
        split = np.flipud(base).copy()
        split[2:4, 2:6] = 7
        split[4:6, 2:6] = 8
        arrays = {
            0: base,
            1: split,
            2: np.fliplr(base),
            3: np.flipud(np.fliplr(base)),
        }
        foreground = analyze_foreground_transform_consistency(arrays)
        geometry = analyze_instance_geometry_transform_consistency(arrays)
        self.assertTrue(foreground["foreground_transform_exact"])
        self.assertFalse(geometry["instance_region_transform_exact"])
        self.assertLess(geometry["instance_region_transform_min_iou"], 1.0)

    def test_sampling_is_deterministic_and_reports_missing_strata(self):
        frame = pd.DataFrame(
            [
                {
                    "mask_key": key,
                    "strata": strata,
                    "label_transform_min_pixel_agreement": score,
                }
                for key, strata, score in (
                    ("a", "instance_random|lowest_transform_consistency", 1.0),
                    ("b", "label_component_mismatch", 0.98),
                    ("c", "binary_control", 1.0),
                )
            ]
        )
        first, missing_first = select_instance_samples(frame, per_stratum=1, seed=42)
        second, missing_second = select_instance_samples(frame, per_stratum=1, seed=42)
        self.assertEqual(first["mask_key"].tolist(), second["mask_key"].tolist())
        self.assertEqual(first["selected_strata"].tolist(), second["selected_strata"].tolist())
        self.assertEqual(missing_first, missing_second)
        self.assertIn("edge_contact", missing_first)

    def test_color_contact_sheet_has_fixed_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.base_labels()
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
            Image.new("RGB", (14, 12), (130, 80, 100)).save(rgb_path)
            row = {
                "mask_key": "cell",
                "pair_status": "mask_ineligible",
                "source_structure": "unsuffixed",
                "member_paths": "|".join(mask_paths),
                "source_paths": str(rgb_path),
            }
            output = root / "instance_contact.png"
            render_instance_contact_sheet(row, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (1024, 1052))

    def test_unsuffixed_rgb_overlays_inverse_transform_for_each_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.base_labels()
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
            Image.new("RGB", (14, 12), (130, 80, 100)).save(rgb_path)
            row = {
                "mask_key": "cell",
                "pair_status": "mask_ineligible",
                "source_structure": "unsuffixed",
                "member_paths": "|".join(mask_paths),
                "source_paths": str(rgb_path),
            }
            output = root / "instance_contact.png"
            render_instance_contact_sheet(row, output)
            with Image.open(output) as image:
                row0 = np.asarray(image.crop((512, 49, 768, 284)))
                row1 = np.asarray(image.crop((512, 305, 768, 540)))
                self.assertTrue(np.array_equal(row0, row1))


if __name__ == "__main__":
    unittest.main()
