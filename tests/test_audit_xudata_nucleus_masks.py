import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

from experiments.audit_xudata_nucleus_masks import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    OWNER_SCHEMA,
    _verify_artifact_hashes,
    build_parser,
    load_development_manifests,
    prepare_output_directory,
    run_audit,
)


class NucleusMaskAuditCliTests(unittest.TestCase):
    @staticmethod
    def write_rgb(path, size=(8, 6)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (190, 120, 170)).save(path)

    @staticmethod
    def write_mask(path, shift=0):
        array = np.zeros((6, 8), dtype=np.uint16)
        array[1:5, 2 + shift : 6 + shift] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)

    @staticmethod
    def write_manifest(path, split, image_path):
        pd.DataFrame(
            [
                {
                    "image_path": str(image_path),
                    "split": split,
                    "diagnosis_label": 1 if split == "train" else 4,
                    "diagnosis_name": "ASC-US" if split == "train" else "HSIL",
                }
            ]
        ).to_csv(path, index=False)

    def test_parser_exposes_no_calibration_or_test_arguments(self):
        actions = {action.dest for action in build_parser()._actions}
        self.assertNotIn("calibration_csv", actions)
        self.assertNotIn("test_csv", actions)
        self.assertEqual(
            actions,
            {"help", "train_csv", "dev_csv", "mask_root", "out_dir", "overwrite"},
        )

    def test_manifest_loader_rejects_wrong_declared_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "cell.jpg"
            self.write_rgb(image)
            train_csv = root / "train.csv"
            dev_csv = root / "dev.csv"
            self.write_manifest(train_csv, "dev", image)
            self.write_manifest(dev_csv, "dev", image)
            with self.assertRaisesRegex(ValueError, "train manifest"):
                load_development_manifests(train_csv, dev_csv)

    def test_existing_unowned_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            (out_dir / "foreign.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "unowned"):
                prepare_output_directory(out_dir, overwrite=True)

    def test_owned_output_with_nested_link_is_rejected_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            nested = out_dir / "nested"
            nested.mkdir(parents=True)
            marker = out_dir / OWNER_FILENAME
            marker.write_text(
                json.dumps({"schema": OWNER_SCHEMA}),
                encoding="utf-8",
            )
            nested_link = nested / "linked-output"
            nested_link.write_text("sentinel", encoding="utf-8")

            def is_reparse_point(path):
                return Path(path) == nested_link

            with mock.patch(
                "experiments.audit_xudata_nucleus_masks._is_reparse_point",
                side_effect=is_reparse_point,
            ):
                with self.assertRaisesRegex(FileExistsError, "containing links"):
                    prepare_output_directory(out_dir, overwrite=True)
            self.assertTrue(nested_link.is_file())

    def test_registered_artifact_hashes_are_reopened_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            artifact = out_dir / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            _verify_artifact_hashes(out_dir, {"artifact.txt": expected})
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _verify_artifact_hashes(out_dir, {"artifact.txt": expected})

    def test_end_to_end_audit_writes_verified_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_root = root / "xudata"
            mask_root = root / "masks"
            out_dir = root / "results"
            train_image = rgb_root / "train" / "train_cell.jpg"
            dev_image = rgb_root / "dev" / "dev_cell.jpg"
            self.write_rgb(train_image)
            self.write_rgb(dev_image)
            for key in ("train_cell", "dev_cell"):
                for member in range(4):
                    self.write_mask(mask_root / f"{key}_{member}.tif")

            train_csv = root / "train.csv"
            dev_csv = root / "dev.csv"
            self.write_manifest(train_csv, "train", train_image)
            self.write_manifest(dev_csv, "dev", dev_image)

            progress = []
            result = run_audit(
                train_csv,
                dev_csv,
                mask_root,
                out_dir,
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                ),
            )

            self.assertEqual(result["route"], "GEOMETRY_PASS_SEMANTICS_REQUIRED")
            self.assertEqual(progress, [(2, 2)])
            self.assertTrue(result["geometry_passed"])
            self.assertFalse(result["calibration_used"])
            self.assertFalse(result["test_used"])
            self.assertFalse(result["training_manifest_generated"])
            self.assertFalse(result["model_trained"])
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                set(ARTIFACT_FILENAMES) | {".m4_d0_audit_owner.json"},
            )

            completed = json.loads(
                (out_dir / "completed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("completed.json", completed["artifact_sha256"])
            self.assertEqual(
                set(completed["artifact_sha256"]),
                set(ARTIFACT_FILENAMES[:-1]),
            )
            for filename, expected_hash in completed["artifact_sha256"].items():
                digest = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(digest, expected_hash)


if __name__ == "__main__":
    unittest.main()
