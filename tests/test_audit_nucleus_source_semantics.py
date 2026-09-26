import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.audit_nucleus_source_semantics import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    build_parser,
    prepare_output_directory,
    run_audit,
)


class SemanticAuditCliTests(unittest.TestCase):
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

    def test_existing_unowned_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "results"
            out_dir.mkdir()
            (out_dir / "foreign.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "unowned"):
                prepare_output_directory(out_dir, [root / "input"], overwrite=True)

    @staticmethod
    def write_group(root, key, status="unique_geometry_valid"):
        mask_root = root / "masks"
        image_root = root / "images"
        mask_root.mkdir(parents=True, exist_ok=True)
        image_root.mkdir(parents=True, exist_ok=True)
        base = np.zeros((8, 10), dtype=np.uint8)
        base[2:4, 3:6] = 1
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
        Image.new("RGB", (10, 8), (140, 90, 120)).save(image)
        return {
            "mask_key": key,
            "member_paths": "|".join(mask_paths),
            "source_paths": str(image),
            "pair_status": status,
            "source_structure": "unsuffixed",
            "eligible_binary_group": status == "unique_geometry_valid",
            "mask_transform_min_match": 1.0,
        }

    def test_end_to_end_writes_review_route_and_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "results"
            rows = [
                self.write_group(input_root, "cell"),
                self.write_group(input_root, "bad", status="mask_ineligible"),
            ]
            pairing_csv = root / "source_mask_pairing.csv"
            pd.DataFrame(rows).to_csv(pairing_csv, index=False)
            result = run_audit(
                pairing_csv=pairing_csv,
                mask_root=input_root / "masks",
                search_roots=[input_root / "images"],
                out_dir=output_root,
                per_stratum=1,
                seed=7,
            )
            self.assertEqual(result["route"], "SEMANTIC_REVIEW_REQUIRED")
            self.assertFalse(result["mask_semantics_confirmed"])
            self.assertTrue((output_root / "visual_audit").is_dir())
            self.assertTrue(list((output_root / "visual_audit").glob("*.png")))
            completed = json.loads(
                (output_root / "completed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("completed.json", completed["artifact_sha256"])
            self.assertIn("artifact_manifest.json", completed["artifact_sha256"])
            for filename, expected in completed["artifact_sha256"].items():
                actual = hashlib.sha256((output_root / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            expected_names = set(ARTIFACT_FILENAMES) | {OWNER_FILENAME, "visual_audit"}
            self.assertTrue(expected_names.issubset({path.name for path in output_root.iterdir()}))


if __name__ == "__main__":
    unittest.main()
