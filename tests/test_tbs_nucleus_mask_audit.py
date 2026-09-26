import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_mask_audit import (
    analyze_mask_group,
    binary_mask_metrics,
    build_pairing_inventory,
    determine_audit_route,
    dice_score,
    inventory_masks,
    load_binary_mask,
    parse_mask_identity,
    summarize_pairing,
)


class MaskIdentityTests(unittest.TestCase):
    def test_parse_mask_identity_removes_only_terminal_member_suffix(self):
        key, member = parse_mask_identity(Path("10000_45258_110_114_3.tif"))
        self.assertEqual(key, "10000_45258_110_114")
        self.assertEqual(member, 3)

    def test_parse_mask_identity_rejects_missing_member_suffix(self):
        with self.assertRaisesRegex(ValueError, "terminal integer member suffix"):
            parse_mask_identity(Path("10000_45258_110_114.tif"))


class BinaryMaskTests(unittest.TestCase):
    @staticmethod
    def write_mask(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(array, dtype=np.uint16)).save(path)

    def test_load_binary_mask_accepts_zero_one_uint16(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_0.tif"
            expected = np.zeros((6, 8), dtype=np.uint16)
            expected[1:5, 2:6] = 1
            self.write_mask(path, expected)
            actual = load_binary_mask(path)
            self.assertEqual(actual.dtype, np.bool_)
            np.testing.assert_array_equal(actual, expected.astype(bool))

    def test_load_binary_mask_rejects_more_than_two_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_0.tif"
            invalid = np.array([[0, 1], [2, 0]], dtype=np.uint16)
            self.write_mask(path, invalid)
            with self.assertRaisesRegex(ValueError, "binary"):
                load_binary_mask(path)

    def test_binary_metrics_reports_geometry(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 2:6] = True
        metrics = binary_mask_metrics(mask)
        self.assertEqual(metrics["width"], 8)
        self.assertEqual(metrics["height"], 6)
        self.assertEqual(metrics["foreground_pixels"], 16)
        self.assertAlmostEqual(metrics["foreground_fraction"], 16 / 48)
        self.assertEqual(metrics["component_count"], 1)
        self.assertEqual(metrics["bbox_left"], 2)
        self.assertEqual(metrics["bbox_top"], 1)
        self.assertFalse(metrics["edge_contact"])

    def test_dice_handles_two_empty_masks(self):
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(dice_score(empty, empty), 1.0)

    def test_group_agreement_detects_identical_members(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        result = analyze_mask_group({0: mask.copy(), 1: mask.copy()})
        self.assertEqual(result["member_count"], 2)
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["pairwise_dice_mean"], 1.0)
        self.assertEqual(result["pairwise_jaccard_mean"], 1.0)
        self.assertTrue(result["consensus_nonempty"])
        self.assertEqual(result["consensus_foreground_pixels"], 16)

    def test_group_agreement_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            analyze_mask_group(
                {
                    0: np.zeros((8, 8), dtype=bool),
                    1: np.zeros((7, 8), dtype=bool),
                }
            )

    def test_four_member_consensus_requires_strict_three_of_four_majority(self):
        foreground = np.zeros((4, 4), dtype=bool)
        foreground[1, 1] = True
        empty = np.zeros((4, 4), dtype=bool)
        result = analyze_mask_group(
            {0: foreground, 1: foreground.copy(), 2: empty, 3: empty.copy()}
        )
        self.assertEqual(result["vote_threshold"], 3)
        self.assertFalse(result["consensus_nonempty"])


class PairingTests(unittest.TestCase):
    @staticmethod
    def write_mask(path, foreground=True):
        array = np.zeros((6, 8), dtype=np.uint16)
        if foreground:
            array[1:5, 2:6] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)

    def test_inventory_marks_duplicate_member_indices_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mask(root / "a" / "cell_0.tif")
            self.write_mask(root / "b" / "cell_0.tif")
            masks, groups = inventory_masks(root)
            self.assertEqual(len(masks), 2)
            self.assertEqual(len(groups), 1)
            self.assertTrue(bool(groups.iloc[0]["duplicate_member_index"]))
            self.assertFalse(bool(groups.iloc[0]["group_valid"]))

    def test_inventory_requires_all_four_member_indices_for_valid_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for member in (0, 1, 2):
                self.write_mask(root / f"cell_{member}.tif")
            _, groups = inventory_masks(root)
            self.assertEqual(len(groups), 1)
            self.assertFalse(bool(groups.iloc[0]["complete_member_set"]))
            self.assertFalse(bool(groups.iloc[0]["group_valid"]))

    def test_inventory_rejects_when_no_filename_has_valid_member_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mask(root / "cell_114.tif")
            with self.assertRaisesRegex(ValueError, "No masks with member suffix"):
                inventory_masks(root)

    def test_inventory_reports_final_group_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for member in range(4):
                self.write_mask(root / f"cell_{member}.tif")
            progress = []
            inventory_masks(
                root,
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                ),
            )
            self.assertEqual(progress, [(1, 1)])

    def test_pairing_marks_duplicate_rgb_stems_ambiguous(self):
        manifest = pd.DataFrame(
            [
                {
                    "image_path": "/train/a/cell.jpg",
                    "split": "train",
                    "diagnosis_label": 1,
                    "diagnosis_name": "ASC-US",
                    "rgb_width": 8,
                    "rgb_height": 6,
                },
                {
                    "image_path": "/train/b/cell.jpg",
                    "split": "train",
                    "diagnosis_label": 2,
                    "diagnosis_name": "LSIL",
                    "rgb_width": 8,
                    "rgb_height": 6,
                },
            ]
        )
        groups = pd.DataFrame(
            [
                {
                    "mask_key": "cell",
                    "member_count": 4,
                    "group_valid": True,
                    "consensus_nonempty": True,
                    "mask_width": 8,
                    "mask_height": 6,
                }
            ]
        )
        paired = build_pairing_inventory(manifest, groups)
        self.assertEqual(set(paired["pair_status"]), {"ambiguous_rgb_key"})

    def test_pairing_requires_exact_rgb_mask_dimensions(self):
        manifest = pd.DataFrame(
            [
                {
                    "image_path": "/train/a/cell.jpg",
                    "split": "train",
                    "diagnosis_label": 1,
                    "diagnosis_name": "ASC-US",
                    "rgb_width": 9,
                    "rgb_height": 6,
                }
            ]
        )
        groups = pd.DataFrame(
            [
                {
                    "mask_key": "cell",
                    "member_count": 4,
                    "group_valid": True,
                    "consensus_nonempty": True,
                    "mask_width": 8,
                    "mask_height": 6,
                }
            ]
        )
        paired = build_pairing_inventory(manifest, groups)
        self.assertEqual(paired.iloc[0]["pair_status"], "shape_mismatch")

    def test_route_never_declares_training_ready_without_semantics(self):
        summary = {
            "cross_split_key_collisions": 0,
            "train_unique_coverage": 1.0,
            "dev_unique_coverage": 1.0,
            "valid_nonempty_group_rate": 1.0,
            "shape_mismatch_count": 0,
        }
        self.assertEqual(
            determine_audit_route(summary),
            "GEOMETRY_PASS_SEMANTICS_REQUIRED",
        )

    def test_route_stops_on_cross_split_key_collision_first(self):
        summary = {
            "cross_split_key_collisions": 1,
            "train_unique_coverage": 1.0,
            "dev_unique_coverage": 1.0,
            "valid_nonempty_group_rate": 1.0,
            "shape_mismatch_count": 0,
        }
        self.assertEqual(
            determine_audit_route(summary),
            "STOP_CROSS_SPLIT_KEY_COLLISION",
        )

    def test_summarize_pairing_uses_fixed_denominators(self):
        paired = pd.DataFrame(
            [
                {"split": "train", "mask_key": "a", "pair_status": "unique_geometry_valid"},
                {"split": "train", "mask_key": "b", "pair_status": "missing_mask_group"},
                {"split": "dev", "mask_key": "c", "pair_status": "unique_geometry_valid"},
            ]
        )
        groups = pd.DataFrame(
            [
                {"mask_key": "a", "group_valid": True, "consensus_nonempty": True},
                {"mask_key": "c", "group_valid": True, "consensus_nonempty": True},
            ]
        )
        summary, split_summary = summarize_pairing(paired, groups)
        self.assertEqual(summary["train_unique_coverage"], 0.5)
        self.assertEqual(summary["dev_unique_coverage"], 1.0)
        self.assertEqual(split_summary.set_index("split").loc["train", "total_images"], 2)

    def test_unique_pairing_coverage_is_separate_from_geometry_validity(self):
        paired = pd.DataFrame(
            [
                {"split": "train", "mask_key": "a", "pair_status": "invalid_mask_group"},
                {"split": "train", "mask_key": "b", "pair_status": "shape_mismatch"},
                {"split": "dev", "mask_key": "c", "pair_status": "unique_geometry_valid"},
            ]
        )
        groups = pd.DataFrame(
            [
                {"mask_key": "a", "group_valid": False, "consensus_nonempty": False},
                {"mask_key": "b", "group_valid": True, "consensus_nonempty": True},
                {"mask_key": "c", "group_valid": True, "consensus_nonempty": True},
            ]
        )
        summary, split_summary = summarize_pairing(paired, groups)
        by_split = split_summary.set_index("split")
        self.assertEqual(summary["train_unique_coverage"], 1.0)
        self.assertEqual(by_split.loc["train", "unique_pair_match_count"], 2)
        self.assertEqual(by_split.loc["train", "unique_geometry_valid"], 0)
        self.assertEqual(summary["shape_mismatch_count"], 1)
        self.assertLess(summary["valid_nonempty_group_rate"], 1.0)

    def test_shape_mismatch_route_is_not_masked_by_pairing_coverage(self):
        paired = pd.DataFrame(
            [
                {"split": "train", "mask_key": "a", "pair_status": "shape_mismatch"},
                {"split": "dev", "mask_key": "b", "pair_status": "unique_geometry_valid"},
            ]
        )
        groups = pd.DataFrame(
            [
                {"mask_key": "a", "group_valid": True, "consensus_nonempty": True},
                {"mask_key": "b", "group_valid": True, "consensus_nonempty": True},
            ]
        )
        summary, _ = summarize_pairing(paired, groups)
        self.assertEqual(summary["train_unique_coverage"], 1.0)
        self.assertEqual(
            determine_audit_route(summary),
            "STOP_RGB_MASK_SHAPE_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
