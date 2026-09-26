import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.filter_nucleus_instance_candidates import (
    ARTIFACT_FILENAMES,
    OWNER_FILENAME,
    build_parser,
    prepare_output_directory,
    run_filter,
)


def metric_row(key, accepted=True):
    return {
        "mask_key": key,
        "pair_status": "mask_ineligible" if accepted else "unique_geometry_valid",
        "source_structure": "augmented",
        "member_count": 4,
        "readable_member_count": 4,
        "complete_read": True,
        "binary_control": False if accepted else True,
        "nonzero_label_count_min": 3 if accepted else 1,
        "edge_contact_member_count": 0,
        "member0_edge_contact": False,
        "label_component_mismatch_member_count": 0,
        "member0_label_component_match": True,
        "foreground_transform_min_pixel_agreement": 0.995 if accepted else 0.98,
        "instance_region_transform_min_iou": 0.995 if accepted else 0.98,
        "instance_region_match_rate_min": 1.0 if accepted else 0.0,
        "instance_region_count_min": 3 if accepted else 1,
        "instance_region_count_max": 3 if accepted else 1,
        "source_paths": f'/server/train_images/{key}_0.jpg|/server/train_images/{key}_1.jpg|/server/train_images/{key}_2.jpg|/server/train_images/{key}_3.jpg',
    }


def pairing_row(key, scope="train"):
    return {
        "mask_key": key,
        "member_paths": f'/server/tif/{key}_0.tif|/server/tif/{key}_1.tif|/server/tif/{key}_2.tif|/server/tif/{key}_3.tif',
        "source_paths": f'/server/{scope}_images/{key}_0.jpg|/server/{scope}_images/{key}_1.jpg|/server/{scope}_images/{key}_2.jpg|/server/{scope}_images/{key}_3.jpg',
        "source_structure": "augmented",
    }


class CandidateFilterCliTests(unittest.TestCase):
    def test_direct_script_help_is_runnable_from_repository_root(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "filter_nucleus_instance_candidates.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--metrics_csv", result.stdout)

    def test_parser_exposes_only_read_only_filter_arguments(self):
        actions = {action.dest for action in build_parser()._actions}

        self.assertEqual(
            actions,
            {
                "help",
                "metrics_csv",
                "pairing_csv",
                "out_dir",
                "foreground_min_agreement",
                "instance_min_iou",
                "instance_min_match_rate",
                "overwrite",
            },
        )

    def test_output_inside_input_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.csv"
            metrics.write_text("mask_key\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inside output directory"):
                prepare_output_directory(root, [metrics], overwrite=False)

    def test_end_to_end_writes_candidate_and_representative_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_csv = root / "metrics.csv"
            pairing_csv = root / "pairing.csv"
            out_dir = root / "candidate_output"
            pd.DataFrame([metric_row("keep"), metric_row("drop", accepted=False)]).to_csv(
                metrics_csv, index=False
            )
            pd.DataFrame([pairing_row("keep"), pairing_row("drop")]).to_csv(
                pairing_csv, index=False
            )

            decision = run_filter(metrics_csv, pairing_csv, out_dir)

            self.assertEqual(decision["route"], "PROVISIONAL_AUXILIARY_CANDIDATE_SET")
            self.assertEqual(decision["accepted_group_count"], 1)
            self.assertEqual(decision["representative_member_count"], 1)
            self.assertFalse(decision["production_training_manifest_generated"])
            candidates = pd.read_csv(out_dir / "candidate_groups.csv")
            representatives = pd.read_csv(out_dir / "representative_members.csv")
            rejected = pd.read_csv(out_dir / "rejected_groups.csv")
            self.assertEqual(candidates["mask_key"].tolist(), ["keep"])
            self.assertEqual(representatives["representative_member"].tolist(), [0])
            self.assertEqual(rejected["mask_key"].tolist(), ["drop"])
            self.assertEqual(
                representatives.loc[0, "representative_source_scope"], "external_train"
            )
            self.assertIn("rejection_reasons", rejected.columns)

            completed = json.loads((out_dir / "completed.json").read_text(encoding="utf-8"))
            for filename, expected in completed["artifact_sha256"].items():
                actual = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            self.assertTrue(set(ARTIFACT_FILENAMES) | {OWNER_FILENAME} <= {p.name for p in out_dir.iterdir()})

    def test_csv_round_trips_quotes_and_commas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_csv = root / "metrics.csv"
            pairing_csv = root / "pairing.csv"
            out_dir = root / "candidate_output"
            row = metric_row('group,"quoted"')
            row["source_paths"] = '/server/train_images/a, "quoted"_0.jpg|/server/train_images/a_1.jpg|/server/train_images/a_2.jpg|/server/train_images/a_3.jpg'
            pair = pairing_row(row["mask_key"])
            pair["source_paths"] = '/server/train_images/a, "quoted"_0.jpg|/server/train_images/a_1.jpg|/server/train_images/a_2.jpg|/server/train_images/a_3.jpg'
            pd.DataFrame([row]).to_csv(metrics_csv, index=False)
            pd.DataFrame([pair]).to_csv(pairing_csv, index=False)

            run_filter(metrics_csv, pairing_csv, out_dir)

            representatives = pd.read_csv(out_dir / "representative_members.csv")
            self.assertEqual(
                representatives.loc[0, "representative_source_path"],
                '/server/train_images/a, "quoted"_0.jpg',
            )

    def test_external_test_group_is_rejected_from_training_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_csv = root / "metrics.csv"
            pairing_csv = root / "pairing.csv"
            out_dir = root / "candidate_output"
            pd.DataFrame([metric_row("test_candidate")]).to_csv(metrics_csv, index=False)
            pd.DataFrame([pairing_row("test_candidate", scope="test")]).to_csv(
                pairing_csv, index=False
            )

            decision = run_filter(metrics_csv, pairing_csv, out_dir)

            self.assertEqual(decision["accepted_group_count"], 0)
            self.assertEqual(decision["representative_member_count"], 0)
            self.assertEqual(decision["rejected_group_count"], 1)
            self.assertEqual(
                decision["rejection_reason_counts"]["source_scope_not_allowed"], 1
            )


if __name__ == "__main__":
    unittest.main()
