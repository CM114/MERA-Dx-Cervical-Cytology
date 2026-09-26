import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_source_audit import (
    analyze_mask_transforms,
    analyze_rgb_transforms,
    build_source_pairing,
    canonical_scan_path,
    determine_source_audit_route,
    inventory_mask_transforms,
    parse_candidate_identity,
    parse_member_identity,
    scan_source_candidates,
    summarize_source_pairing,
)


def create_directory_link(test_case, link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt":
            test_case.skipTest(f"Directory symlinks are unavailable: {exc}")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        test_case.skipTest(
            f"Directory links are unavailable: {completed.stderr.strip()}"
        )


def remove_directory_link(link):
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rmdir()


class IdentityTests(unittest.TestCase):
    def test_member_identity_accepts_only_terminal_zero_to_three(self):
        self.assertEqual(
            parse_member_identity(Path("10000_45258_110_114_3.tif")),
            ("10000_45258_110_114", 3),
        )
        with self.assertRaisesRegex(ValueError, "member suffix"):
            parse_member_identity(Path("10000_45258_110_114_4.tif"))

    def test_candidate_identity_uses_only_exact_known_keys(self):
        keys = {"cell", "case_cell"}
        self.assertEqual(
            parse_candidate_identity(Path("case_cell.png"), keys),
            ("case_cell", "unsuffixed", None),
        )
        self.assertEqual(
            parse_candidate_identity(Path("case_cell_2.jpg"), keys),
            ("case_cell", "member", 2),
        )
        self.assertIsNone(parse_candidate_identity(Path("other_cell.png"), keys))

    def test_candidate_identity_rejects_dual_key_interpretation(self):
        with self.assertRaisesRegex(ValueError, "ambiguous candidate identity"):
            parse_candidate_identity(Path("cell_0.png"), {"cell", "cell_0"})


class TransformTests(unittest.TestCase):
    @staticmethod
    def base_mask():
        array = np.zeros((7, 9), dtype=np.uint16)
        array[1:4, 2:5] = 1
        array[4, 5] = 1
        return array

    def test_mask_transform_analysis_confirms_expected_members(self):
        base = self.base_mask()
        result = analyze_mask_transforms(
            {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }
        )
        self.assertTrue(result["eligible_binary_group"])
        self.assertEqual(result["mask_transform_min_match"], 1.0)
        self.assertTrue(result["mask_transform_confirmed"])

    def test_multivalue_masks_are_audited_but_not_training_eligible(self):
        base = self.base_mask()
        base[5, 7] = 2
        result = analyze_mask_transforms(
            {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }
        )
        self.assertFalse(result["binary_group"])
        self.assertFalse(result["eligible_binary_group"])
        self.assertEqual(result["mask_transform_min_match"], 1.0)

    def test_rgb_transform_analysis_uses_normalized_mae(self):
        base = np.zeros((7, 9, 3), dtype=np.uint8)
        base[1:4, 2:5] = (100, 20, 200)
        result = analyze_rgb_transforms(
            {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }
        )
        self.assertEqual(result["rgb_transform_max_normalized_mae"], 0.0)
        self.assertTrue(result["rgb_transform_confirmed"])


class SourceScanTests(unittest.TestCase):
    @staticmethod
    def write_rgb(path, size=(9, 7)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (180, 80, 150)).save(path)

    @staticmethod
    def write_gray(path, size=(9, 7)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", size, 1).save(path)

    def test_scan_excludes_mask_root_and_keeps_grayscale_as_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "corpus"
            masks = search / "masks"
            self.write_gray(masks / "cell_0.tif")
            self.write_rgb(search / "images" / "cell.png")
            self.write_gray(search / "labels" / "case.png")
            candidates, summary = scan_source_candidates(
                [search], {"cell", "case"}, excluded_roots=[masks]
            )
            self.assertEqual(set(candidates["mask_key"]), {"cell", "case"})
            self.assertNotIn(str(masks / "cell_0.tif"), set(candidates["image_path"]))
            by_key = candidates.set_index("mask_key")
            self.assertTrue(bool(by_key.loc["cell", "is_rgb_source"]))
            self.assertFalse(bool(by_key.loc["case", "is_rgb_source"]))
            self.assertEqual(int(summary["matching_candidate_files"].sum()), 2)

    def test_scan_rejects_search_root_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            link = root / "linked"
            real.mkdir()
            create_directory_link(self, link, real)
            try:
                with self.assertRaisesRegex(ValueError, "regular directory"):
                    scan_source_candidates([link], {"cell"})
            finally:
                remove_directory_link(link)

    def test_scan_path_identity_uses_platform_native_case_semantics(self):
        path = Path("MixedCase") / "Cell.PNG"
        self.assertEqual(
            canonical_scan_path(path),
            os.path.normcase(str(path.resolve())),
        )

    def test_scan_marks_dual_key_candidate_identity_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_rgb(root / "cell_0.png")
            candidates, _ = scan_source_candidates(
                [root], {"cell", "cell_0"}
            )
            self.assertEqual(len(candidates), 2)
            self.assertEqual(set(candidates["mask_key"]), {"cell", "cell_0"})
            self.assertTrue(candidates["identity_ambiguous"].all())

            groups = PairingTests.mask_groups(("cell", "cell_0"))
            pairing = build_source_pairing(groups, candidates)
            self.assertEqual(
                set(pairing["pair_status"]), {"ambiguous_source"}
            )


class PairingTests(unittest.TestCase):
    @staticmethod
    def mask_groups(keys=("a", "b")):
        return pd.DataFrame(
            [
                {
                    "mask_key": key,
                    "member_count": 4,
                    "mask_width": 9,
                    "mask_height": 7,
                    "eligible_binary_group": True,
                    "mask_transform_confirmed": True,
                    "member_paths": "",
                }
                for key in keys
            ]
        )

    @staticmethod
    def candidate(key, path, kind="unsuffixed", member=None, width=9, height=7):
        return {
            "mask_key": key,
            "image_path": str(path),
            "candidate_kind": kind,
            "member_index": member,
            "width": width,
            "height": height,
            "is_rgb_source": True,
            "read_error": "",
        }

    def test_unique_unsuffixed_rgb_is_geometry_valid(self):
        groups = self.mask_groups(("a",))
        candidates = pd.DataFrame([self.candidate("a", "/images/a.png")])
        pairing = build_source_pairing(groups, candidates)
        self.assertEqual(pairing.iloc[0]["pair_status"], "unique_geometry_valid")
        self.assertEqual(pairing.iloc[0]["source_structure"], "unsuffixed")

    def test_duplicate_unsuffixed_rgb_is_ambiguous(self):
        groups = self.mask_groups(("a",))
        candidates = pd.DataFrame(
            [
                self.candidate("a", "/images/a.png"),
                self.candidate("a", "/other/a.jpg"),
            ]
        )
        pairing = build_source_pairing(groups, candidates)
        self.assertEqual(pairing.iloc[0]["pair_status"], "ambiguous_source")

    def test_shape_mismatch_is_not_geometry_valid(self):
        groups = self.mask_groups(("a",))
        candidates = pd.DataFrame(
            [self.candidate("a", "/images/a.png", width=8, height=7)]
        )
        pairing = build_source_pairing(groups, candidates)
        self.assertEqual(pairing.iloc[0]["pair_status"], "geometry_mismatch")

    def test_complete_augmented_rgb_set_is_verified_pixelwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.zeros((7, 9, 3), dtype=np.uint8)
            base[1:4, 2:5] = (100, 20, 200)
            arrays = {
                0: base,
                1: np.flipud(base),
                2: np.fliplr(base),
                3: np.flipud(np.fliplr(base)),
            }
            rows = []
            for member, array in arrays.items():
                path = root / f"a_{member}.png"
                Image.fromarray(array).save(path)
                rows.append(
                    self.candidate(
                        "a", path, kind="member", member=member
                    )
                )
            pairing = build_source_pairing(
                self.mask_groups(("a",)), pd.DataFrame(rows)
            )
            self.assertEqual(pairing.iloc[0]["pair_status"], "unique_geometry_valid")
            self.assertEqual(pairing.iloc[0]["source_structure"], "augmented")
            self.assertTrue(bool(pairing.iloc[0]["rgb_transform_confirmed"]))

    def test_summary_uses_only_eligible_binary_groups_as_denominator(self):
        groups = self.mask_groups(("a", "b"))
        groups = pd.concat(
            [
                groups,
                pd.DataFrame(
                    [
                        {
                            "mask_key": "multivalue",
                            "member_count": 4,
                            "mask_width": 9,
                            "mask_height": 7,
                            "eligible_binary_group": False,
                            "mask_transform_confirmed": True,
                            "member_paths": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        candidates = pd.DataFrame([self.candidate("a", "/images/a.png")])
        pairing = build_source_pairing(groups, candidates)
        summary = summarize_source_pairing(pairing)
        self.assertEqual(summary["eligible_binary_groups"], 2)
        self.assertEqual(summary["unique_geometry_valid_groups"], 1)
        self.assertEqual(summary["unique_geometry_coverage"], 0.5)
        self.assertEqual(summary["mask_transform_unconfirmed_groups"], 0)

    def test_incomplete_mask_group_is_ineligible_instead_of_crashing(self):
        groups = pd.DataFrame(
            [
                {
                    "mask_key": "incomplete",
                    "member_count": 3,
                    "eligible_binary_group": False,
                    "mask_transform_confirmed": False,
                    "member_paths": "",
                }
            ]
        )
        candidates = pd.DataFrame(
            [self.candidate("incomplete", "/images/incomplete.png")]
        )
        pairing = build_source_pairing(groups, candidates)
        self.assertEqual(pairing.iloc[0]["pair_status"], "mask_ineligible")
        summary = summarize_source_pairing(pairing)
        self.assertEqual(summary["eligible_binary_groups"], 0)
        self.assertEqual(
            determine_source_audit_route(summary),
            "STOP_SOURCE_RGB_COVERAGE_INSUFFICIENT",
        )

    def test_route_priority_and_success_are_fixed(self):
        base = {
            "matching_rgb_candidate_files": 10,
            "eligible_binary_groups": 10,
            "ambiguous_eligible_groups": 0,
            "unique_geometry_coverage": 1.0,
            "geometry_mismatch_groups": 0,
            "mask_transform_unconfirmed_groups": 0,
            "mask_transform_confirmed_rate": 1.0,
            "augmented_rgb_groups": 0,
            "rgb_transform_confirmed_rate": 1.0,
        }
        self.assertEqual(
            determine_source_audit_route(base),
            "SOURCE_PAIRS_FOUND_SEMANTICS_REQUIRED",
        )
        failed = dict(base, matching_rgb_candidate_files=0)
        self.assertEqual(
            determine_source_audit_route(failed), "STOP_NO_SOURCE_RGB_MATCH"
        )
        failed = dict(base, ambiguous_eligible_groups=1)
        self.assertEqual(
            determine_source_audit_route(failed), "STOP_SOURCE_RGB_AMBIGUOUS"
        )
        failed = dict(base, unique_geometry_coverage=0.94)
        self.assertEqual(
            determine_source_audit_route(failed),
            "STOP_SOURCE_RGB_COVERAGE_INSUFFICIENT",
        )
        failed = dict(base, geometry_mismatch_groups=1)
        self.assertEqual(
            determine_source_audit_route(failed),
            "STOP_RGB_MASK_GEOMETRY_MISMATCH",
        )
        failed = dict(
            base,
            mask_transform_unconfirmed_groups=1,
            mask_transform_confirmed_rate=0.99,
        )
        self.assertEqual(
            determine_source_audit_route(failed),
            "STOP_MASK_AUGMENTATION_RELATION_UNCONFIRMED",
        )
        failed = dict(base, augmented_rgb_groups=2, rgb_transform_confirmed_rate=0.5)
        self.assertEqual(
            determine_source_audit_route(failed),
            "STOP_RGB_AUGMENTATION_RELATION_UNCONFIRMED",
        )


class MaskInventoryTests(unittest.TestCase):
    @staticmethod
    def write_mask(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(array, dtype=np.uint16)).save(path)

    def test_inventory_reads_binary_and_multivalue_groups_without_thresholding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = np.zeros((7, 9), dtype=np.uint16)
            binary[1:4, 2:5] = 1
            multivalue = binary.copy()
            multivalue[5, 7] = 2
            for key, base in (("binary", binary), ("labels", multivalue)):
                members = {
                    0: base,
                    1: np.flipud(base),
                    2: np.fliplr(base),
                    3: np.flipud(np.fliplr(base)),
                }
                for member, array in members.items():
                    self.write_mask(root / f"{key}_{member}.tif", array)
            inventory, groups = inventory_mask_transforms(root)
            self.assertEqual(len(inventory), 8)
            by_key = groups.set_index("mask_key")
            self.assertTrue(bool(by_key.loc["binary", "eligible_binary_group"]))
            self.assertFalse(bool(by_key.loc["labels", "eligible_binary_group"]))
            self.assertTrue(bool(by_key.loc["labels", "mask_transform_confirmed"]))

    def test_inventory_rejects_mask_root_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            link = root / "linked"
            real.mkdir()
            create_directory_link(self, link, real)
            try:
                with self.assertRaisesRegex(ValueError, "regular directory"):
                    inventory_mask_transforms(link)
            finally:
                remove_directory_link(link)


if __name__ == "__main__":
    unittest.main()
