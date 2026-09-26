import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from experiments.train_tbs_singleview import (
    _load_initial_weights,
    _sha256,
    resolve_stagewise_splits,
    validate_manifest_independence,
    validate_checkpoint_stage,
    validate_selection_role,
)


def s0r_checkpoint_payload():
    return {
        "schema_version": "xudata-tbs-s0r-final-checkpoint-v1",
        "stage": "s0r",
        "selection_role": "fivefold_fixed_epoch",
        "model_name": "caformer_s18",
        "epoch": 3,
        "pool_sha256": "pool",
        "select_s1_sha256": "sealed",
    }


class SingleViewTrainingContractTests(unittest.TestCase):
    def test_s0r_to_s1_requires_passed_hash_bound_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits"
            split_dir.mkdir()
            for name in ("train_fit.csv", "select_s0.csv", "select_s1.csv"):
                (split_dir / name).write_text(f"{name}\n", encoding="utf-8")
            run_dir = root / "final"
            run_dir.mkdir()
            checkpoint = run_dir / "final_model.pth"
            checkpoint.write_bytes(b"checkpoint")
            predictions = run_dir / "dev_predictions.csv"
            predictions.write_text("image_path,true_label\na.jpg,0\n", encoding="utf-8")
            payload = {
                **s0r_checkpoint_payload(),
                "train_fit_sha256": _sha256(split_dir / "train_fit.csv"),
                "select_s0_sha256": _sha256(split_dir / "select_s0.csv"),
                "select_s1_sha256": _sha256(split_dir / "select_s1.csv"),
                "model_state": {"backbone.weight": "value"},
            }
            model = Mock()
            args = SimpleNamespace(
                stage="s1",
                init_checkpoint=checkpoint,
                s0r_gate_json=None,
                split_dir=split_dir,
                model_name="caformer_s18",
            )
            with patch(
                "experiments.train_tbs_singleview._load_checkpoint",
                return_value=payload,
            ):
                with self.assertRaisesRegex(ValueError, "s0r_gate_json"):
                    _load_initial_weights(model, args, "cpu", "unused")

                gate = root / "gate.json"
                gate.write_text(
                    json.dumps(
                        {
                            "stage": "s0",
                            "passed": True,
                            "candidate_predictions_sha256": _sha256(predictions),
                        }
                    ),
                    encoding="utf-8",
                )
                args.s0r_gate_json = gate
                _load_initial_weights(model, args, "cpu", "unused")
            model.backbone.load_state_dict.assert_called_once_with(
                {"weight": "value"}, strict=True
            )

    def test_s1_accepts_only_strict_s0r_final_checkpoint_metadata(self):
        payload = s0r_checkpoint_payload()
        validate_checkpoint_stage("s1", payload)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_checkpoint_stage("s0", payload)
        with self.assertRaisesRegex(ValueError, "selection_role"):
            validate_checkpoint_stage(
                "s1", {**payload, "selection_role": "select_s0"}
            )
    def test_each_stage_accepts_only_its_own_selection_role(self):
        validate_selection_role("s0", "select_s0")
        validate_selection_role("s1", "select_s1")
        with self.assertRaisesRegex(ValueError, "select_s0"):
            validate_selection_role("s0", "select_s1")
        with self.assertRaisesRegex(ValueError, "select_s1"):
            validate_selection_role("s1", "select_s0")

    def test_checkpoint_stage_is_strict(self):
        valid = {
            "schema_version": "xudata-tbs-singleview-checkpoint-v1",
            "stage": "s0",
            "selection_role": "select_s0",
        }
        validate_checkpoint_stage("s0", valid)
        validate_checkpoint_stage("s1", valid)
        with self.assertRaisesRegex(ValueError, "s0"):
            validate_checkpoint_stage("s0", {**valid, "stage": "s1"})
        with self.assertRaisesRegex(ValueError, "s1"):
            validate_checkpoint_stage("s1", {**valid, "stage": "s1"})

    def test_s1_checkpoint_requires_owned_s0_metadata(self):
        valid = {
            "schema_version": "xudata-tbs-singleview-checkpoint-v1",
            "stage": "s0",
            "selection_role": "select_s0",
        }
        validate_checkpoint_stage("s1", valid)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_checkpoint_stage("s1", {**valid, "schema_version": "other"})
        with self.assertRaisesRegex(ValueError, "select_s0"):
            validate_checkpoint_stage("s1", {**valid, "selection_role": "select_s1"})

    def test_existing_split_locks_seed_and_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "image_path": f"/{label}_{index}.jpg",
                    "diagnosis_label": label,
                }
                for label in range(5)
                for index in range(40)
            ]
            train_csv = root / "train.csv"
            pd.DataFrame(rows).to_csv(train_csv, index=False)
            split_dir = root / "splits"
            resolve_stagewise_splits(train_csv, split_dir, 0.05, 42)
            with self.assertRaisesRegex(ValueError, "seed"):
                resolve_stagewise_splits(train_csv, split_dir, 0.05, 7)
            with self.assertRaisesRegex(ValueError, "select_fraction"):
                resolve_stagewise_splits(train_csv, split_dir, 0.10, 42)

    def test_duplicate_content_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "image_path": f"/{label}_{index}.jpg",
                    "diagnosis_label": label,
                    "content_sha256": f"{label}-{index}",
                }
                for label in range(5)
                for index in range(40)
            ]
            rows[-1]["content_sha256"] = rows[0]["content_sha256"]
            train_csv = root / "train.csv"
            pd.DataFrame(rows).to_csv(train_csv, index=False)
            with self.assertRaisesRegex(ValueError, "content_sha256"):
                resolve_stagewise_splits(train_csv, root / "splits", 0.05, 42)

    def test_train_dev_content_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            dev = root / "dev.csv"
            pd.DataFrame(
                [{"image_path": "/train/a.jpg", "content_sha256": "same"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"image_path": "/dev/b.jpg", "content_sha256": "same"}]
            ).to_csv(dev, index=False)
            with self.assertRaisesRegex(ValueError, "train/dev.*overlap"):
                validate_manifest_independence(train, dev)


if __name__ == "__main__":
    unittest.main()
