import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.audit_tbs_candidate_data_sources import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    OWNER_SCHEMA,
    build_parser,
    prepare_output_directory,
    run_audit,
    write_csv_artifact,
)
from experiments.tbs.data_source_audit import AuditLimits


class CandidateSourceAuditCliTests(unittest.TestCase):
    def test_parser_exposes_only_read_only_audit_arguments(self):
        actions = {action.dest for action in build_parser()._actions}
        self.assertEqual(
            actions,
            {
                "help",
                "root",
                "out_dir",
                "source",
                "max_entries_per_source",
                "max_metadata_preview_bytes",
                "max_image_samples_per_source",
                "duplicate_mode",
                "overwrite",
            },
        )
        for forbidden in ("checkpoint", "device", "epochs", "test_csv"):
            self.assertNotIn(forbidden, actions)

    def test_output_inside_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()

            with self.assertRaisesRegex(ValueError, "outside every input"):
                prepare_output_directory(
                    source / "results",
                    input_sources=[source],
                    overwrite=False,
                )

    def test_existing_unowned_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            out_dir.mkdir()
            sentinel = out_dir / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "unowned"):
                prepare_output_directory(out_dir, [], overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_linked_owner_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "results"
            prepare_output_directory(out_dir, [], overwrite=False)
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
                prepare_output_directory(out_dir, [], overwrite=True)

            self.assertEqual(
                json.loads(external.read_text(encoding="utf-8"))["sentinel"],
                "keep",
            )


class CandidateSourceAuditEndToEndTests(unittest.TestCase):
    @staticmethod
    def _png_bytes(size=(32, 24), color=(170, 70, 140)):
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        return buffer.getvalue()

    def _make_candidate_root(self, parent):
        root = parent / "data-root"
        root.mkdir()
        archive = root / "ClassDataset.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("train/Normal/a.png", self._png_bytes())
            handle.writestr("train/ASC-US/b.png", self._png_bytes(color=(90, 40, 120)))
            handle.writestr("train/LSIL/c.png", self._png_bytes(color=(100, 60, 130)))
            handle.writestr("train/ASC-H/d.png", self._png_bytes(color=(120, 50, 150)))
            handle.writestr("train/HSIL/e.png", self._png_bytes(color=(140, 40, 160)))
            handle.writestr(
                "metadata.csv",
                "patient_id,wsi_id,label,source\nP1,W1,HSIL,external\n",
            )
        return root

    def test_csv_artifact_round_trips_special_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata_inventory.csv"
            frame = pd.DataFrame(
                [
                    {
                        "source_name": "细胞\"七分类.zip",
                        "member_path": "train/ASC-US/sample\\01.png",
                        "size_bytes": 12,
                        "bytes_read": 12,
                        "encoding": "utf-8",
                        "field_candidates": '["核质比", "核异型"]',
                        "preview": "label,核质比\nASC-US,0.42",
                        "status": "ok",
                    }
                ],
                columns=(
                    "source_name",
                    "member_path",
                    "size_bytes",
                    "bytes_read",
                    "encoding",
                    "field_candidates",
                    "preview",
                    "status",
                ),
            )

            write_csv_artifact(path, frame)
            loaded = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                escapechar="\\",
            )

            self.assertEqual(loaded.loc[0, "source_name"], '细胞"七分类.zip')
            self.assertEqual(loaded.loc[0, "member_path"], r"train/ASC-US/sample\01.png")
            self.assertEqual(loaded.loc[0, "preview"], "label,核质比\nASC-US,0.42")

    def test_run_audit_writes_fixed_artifacts_and_completion_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = self._make_candidate_root(parent)
            out_dir = parent / "results"

            summary = run_audit(
                root=root,
                out_dir=out_dir,
                sources=["ClassDataset.zip"],
                limits=AuditLimits(max_image_samples_per_source=2),
            )

            self.assertEqual(summary["route"], "AUDIT_COMPLETE")
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                set(ARTIFACT_FILENAMES) | {OWNER_FILENAME},
            )
            completed = json.loads(
                (out_dir / "completed.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("completed.json", completed["artifact_sha256"])
            for name, digest in completed["artifact_sha256"].items():
                actual = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)
            scorecard = pd.read_csv(out_dir / "source_scorecard.csv")
            self.assertEqual(scorecard.loc[0, "source_name"], "ClassDataset.zip")
            report = (out_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("External evaluation", report)
            self.assertIn("Patient/WSI grouping", report)

    def test_corrupt_source_is_recorded_without_losing_other_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = self._make_candidate_root(parent)
            (root / "broken.zip").write_bytes(b"not-a-zip")
            out_dir = parent / "results"

            summary = run_audit(
                root=root,
                out_dir=out_dir,
                sources=["ClassDataset.zip", "broken.zip"],
                limits=AuditLimits(),
            )

            self.assertEqual(summary["source_failures"], 1)
            inventory = pd.read_csv(out_dir / "source_inventory.csv")
            broken = inventory.loc[inventory["source_name"] == "broken.zip"].iloc[0]
            self.assertEqual(broken["status"], "archive_unreadable")

    def test_entry_cap_sets_complete_with_truncation_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = self._make_candidate_root(parent)
            out_dir = parent / "results"

            summary = run_audit(
                root=root,
                out_dir=out_dir,
                sources=["ClassDataset.zip"],
                limits=AuditLimits(max_entries_per_source=2),
            )

            self.assertEqual(
                summary["route"],
                "AUDIT_COMPLETE_WITH_TRUNCATION",
            )


if __name__ == "__main__":
    unittest.main()
