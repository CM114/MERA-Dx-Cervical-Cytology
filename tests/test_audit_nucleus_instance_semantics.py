import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.audit_nucleus_instance_semantics import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    build_parser,
    prepare_output_directory,
    run_audit,
)


class InstanceAuditCliTests(unittest.TestCase):
    def test_parser_has_no_training_or_sealed_split_arguments(self):
        actions = {action.dest for action in build_parser()._actions}
        self.assertEqual(
            actions,
            {
                "help",
                "pairing_csv",
                "mask_root",
                "search_root",
                "out_dir",
                "per_stratum",
                "seed",
                "overwrite",
            },
        )

    def test_output_inside_input_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "outside input roots"):
                prepare_output_directory(root / "nested", [root], overwrite=False)

    @staticmethod
    def write_group(root, key, multi_value=True):
        mask_root = root / "masks"
        image_root = root / "images"
        mask_root.mkdir(parents=True, exist_ok=True)
        image_root.mkdir(parents=True, exist_ok=True)
        base = np.zeros((12, 14), dtype=np.uint16)
        base[2:5, 2:5] = 1
        if multi_value:
            base[7:10, 8:12] = 2
        mask_paths = []
        for member, array in {
            0: base,
            1: np.flipud(base),
            2: np.fliplr(base),
            3: np.flipud(np.fliplr(base)),
        }.items():
            path = mask_root / f"{key}_{member}.tif"
            Image.fromarray(array).save(path)
            mask_paths.append(str(path))
        image = image_root / f"{key}.jpg"
        Image.new("RGB", (14, 12), (140, 90, 120)).save(image)
        return {
            "mask_key": key,
            "member_paths": "|".join(mask_paths),
            "source_paths": str(image),
            "pair_status": "mask_ineligible" if multi_value else "unique_geometry_valid",
            "source_structure": "unsuffixed",
        }

    def test_end_to_end_writes_instance_review_route_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            out_dir = root / "results"
            rows = [
                self.write_group(input_root, "multi", multi_value=True),
                self.write_group(input_root, "binary", multi_value=False),
            ]
            pairing_csv = root / "source_mask_pairing.csv"
            pd.DataFrame(rows).to_csv(pairing_csv, index=False)
            result = run_audit(
                pairing_csv=pairing_csv,
                mask_root=input_root / "masks",
                search_roots=[input_root / "images"],
                out_dir=out_dir,
                per_stratum=1,
                seed=42,
            )
            self.assertEqual(result["route"], "INSTANCE_SEMANTICS_REVIEW_REQUIRED")
            self.assertFalse(result["instance_semantics_confirmed"])
            self.assertEqual(result["multi_value_group_count"], 1)
            self.assertEqual(result["multi_value_foreground_transform_exact_group_count"], 1)
            self.assertEqual(result["multi_value_instance_geometry_exact_group_count"], 1)
            self.assertEqual(result["multi_value_raw_label_transform_exact_group_count"], 1)
            self.assertTrue(list((out_dir / "visual_audit").glob("*.png")))
            completed = json.loads((out_dir / "completed.json").read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertIn("artifact_manifest.json", completed["artifact_sha256"])
            self.assertNotIn("completed.json", completed["artifact_sha256"])
            for filename, expected in completed["artifact_sha256"].items():
                actual = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            names = {path.name for path in out_dir.iterdir()}
            self.assertTrue(set(ARTIFACT_FILENAMES) | {OWNER_FILENAME, "visual_audit"} <= names)


if __name__ == "__main__":
    unittest.main()
