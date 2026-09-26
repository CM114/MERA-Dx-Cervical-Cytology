import os
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.audit_nucleus_segmentation_source import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    OWNER_SCHEMA,
    build_parser,
    prepare_output_directory,
    run_audit,
)


class SourceAuditCliTests(unittest.TestCase):
    @staticmethod
    def write_mask_group(root, key):
        base = np.zeros((7, 9), dtype=np.uint16)
        base[1:4, 2:5] = 1
        members = {
            0: base,
            1: np.flipud(base),
            2: np.fliplr(base),
            3: np.flipud(np.fliplr(base)),
        }
        for member, array in members.items():
            Image.fromarray(array).save(root / f"{key}_{member}.tif")

    @staticmethod
    def write_rgb(path, size=(9, 7)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (180, 80, 150)).save(path)

    def test_parser_exposes_only_external_audit_arguments(self):
        actions = {action.dest for action in build_parser()._actions}
        self.assertEqual(
            actions,
            {"help", "mask_root", "search_root", "out_dir", "overwrite"},
        )
        for forbidden in (
            "train_csv",
            "dev_csv",
            "calibration_csv",
            "test_csv",
            "checkpoint",
            "device",
        ):
            self.assertNotIn(forbidden, actions)

    def test_output_must_be_outside_all_input_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "corpus"
            search.mkdir()
            with self.assertRaisesRegex(ValueError, "outside input roots"):
                prepare_output_directory(
                    search / "results",
                    input_roots=[search],
                    overwrite=False,
                )

    def test_existing_unowned_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            (out_dir / "foreign.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "unowned"):
                prepare_output_directory(out_dir, input_roots=[], overwrite=True)

    def test_owned_output_with_unknown_entry_is_not_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            prepare_output_directory(out_dir, input_roots=[])
            unknown = out_dir / "do_not_delete.txt"
            unknown.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "unexpected"):
                prepare_output_directory(out_dir, input_roots=[], overwrite=True)
            self.assertEqual(unknown.read_text(encoding="utf-8"), "keep")

    def test_linked_ownership_marker_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "results"
            prepare_output_directory(out_dir, input_roots=[])
            marker = out_dir / OWNER_FILENAME
            marker.unlink()
            external = root / "external-owner.json"
            external.write_text(
                json.dumps({"schema": OWNER_SCHEMA, "sentinel": "keep"}),
                encoding="utf-8",
            )
            try:
                marker.symlink_to(external)
            except OSError:
                os.link(external, marker)
            with self.assertRaisesRegex(FileExistsError, "ownership marker"):
                prepare_output_directory(out_dir, input_roots=[], overwrite=True)
            self.assertEqual(
                json.loads(external.read_text(encoding="utf-8"))["sentinel"],
                "keep",
            )

    def test_end_to_end_audit_writes_completion_last_with_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            masks = corpus / "masks"
            images = corpus / "images"
            out_dir = root / "results"
            masks.mkdir(parents=True)
            for key in ("cell_a", "cell_b"):
                self.write_mask_group(masks, key)
                self.write_rgb(images / f"{key}.png")

            decision = run_audit(
                mask_root=masks,
                search_roots=[corpus],
                out_dir=out_dir,
            )

            self.assertEqual(
                decision["route"], "SOURCE_PAIRS_FOUND_SEMANTICS_REQUIRED"
            )
            self.assertFalse(decision["mask_semantics_confirmed"])
            self.assertFalse(decision["segmentation_model_trained"])
            self.assertFalse(decision["input_scope_verified"])
            self.assertFalse(decision["sealed_split_exclusion_verified"])
            self.assertNotIn("xudata_used", decision)
            self.assertNotIn("calibration_used", decision)
            self.assertNotIn("test_used", decision)
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                set(ARTIFACT_FILENAMES) | {OWNER_FILENAME},
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
            for filename, expected in completed["artifact_sha256"].items():
                actual = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            manifest = json.loads(
                (out_dir / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertRegex(
                manifest["inputs"]["mask_inventory_sha256"], r"^[0-9a-f]{64}$"
            )


if __name__ == "__main__":
    unittest.main()
