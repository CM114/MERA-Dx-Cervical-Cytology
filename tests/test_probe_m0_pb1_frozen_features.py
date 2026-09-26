import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.probe_m0_pb1_frozen_features import (
    OUTPUT_OWNER_FILENAME,
    build_parser,
    prepare_output_directory,
    validate_source_audit,
    write_completed_marker,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ParserTests(unittest.TestCase):
    def test_parser_has_no_calibration_test_or_model_arguments(self):
        destinations = {action.dest for action in build_parser()._actions}

        self.assertEqual(
            destinations,
            {"help", "audit_dir", "out_dir", "overwrite"},
        )


class SourceAuditTests(unittest.TestCase):
    def _source(self, root):
        root = Path(root)
        files = {
            "features/train_index.csv": b"row_index,image_path,true_label\n",
            "features/dev_index.csv": b"row_index,image_path,true_label\n",
            "features/m0_train_features.npy": b"m0-train",
            "features/m0_dev_features.npy": b"m0-dev",
            "features/pb1_train_features.npy": b"pb1-train",
            "features/pb1_dev_features.npy": b"pb1-dev",
        }
        hashes = {}
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            hashes[relative] = _sha256(path)
        marker = {
            "status": "completed",
            "schema": "xudata-tbs5-m0-pb1-failure-audit-v1",
            "seed": 42,
            "train_samples": 7985,
            "dev_samples": 998,
            "data_scope": "clean_v2 train/dev only",
            "calibration_used": False,
            "test_used": False,
            "new_deep_model_trained": False,
            "pb1_promoted": False,
            "artifact_sha256": hashes,
        }
        (root / "completed.json").write_text(json.dumps(marker), encoding="utf-8")
        return root

    def test_validates_locked_source_and_required_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)

            manifest = validate_source_audit(source)

            self.assertEqual(len(manifest), 6)
            self.assertTrue((manifest["sha256_match"] == True).all())

    def test_rejects_tampered_source_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            (source / "features/m0_dev_features.npy").write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_source_audit(source)


class OutputTests(unittest.TestCase):
    def test_overwrite_requires_matching_owner_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            prepare_output_directory(output)
            self.assertTrue((output / OUTPUT_OWNER_FILENAME).is_file())
            (output / "artifact.txt").write_text("x", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)
            prepare_output_directory(output, overwrite=True)
            self.assertFalse((output / "artifact.txt").exists())

    def test_completion_marker_seals_diagnostic_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "decision.json").write_text("{}\n", encoding="utf-8")

            marker = write_completed_marker(output, ["decision.json"])
            payload = json.loads(marker.read_text(encoding="utf-8"))

            self.assertFalse(payload["calibration_used"])
            self.assertFalse(payload["test_used"])
            self.assertFalse(payload["new_deep_model_trained"])
            self.assertTrue(payload["linear_probe_fitted"])
            self.assertFalse(payload["pb1_promoted"])


if __name__ == "__main__":
    unittest.main()

