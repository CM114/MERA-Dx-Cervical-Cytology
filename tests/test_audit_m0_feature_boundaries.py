import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_m0_feature_boundaries import (
    AUDIT_OWNER_FILENAME,
    LOCKED_SEEDS,
    _analyse_seed,
    _build_sample_audit,
    _diagnostic_summary,
    _validate_feature_arrays,
    build_parser,
    parse_checkpoint_specs,
    prepare_output_directory,
    validate_checkpoint_contract,
    validate_manifest_hashes,
    validate_replayed_predictions,
    validate_split_counts,
    write_completed_marker,
)


class FrozenM0ContractTests(unittest.TestCase):
    @staticmethod
    def _synthetic_extraction(prefix):
        labels = np.array([0] + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
        features = np.array(
            [
                [0.0, 8.0],
                [-3.0, 0.0], [-2.5, 0.1], [-2.0, -0.1],
                [2.0, -0.1], [2.5, 0.1], [3.0, 0.0],
                [-3.0, 5.0], [-2.5, 5.1], [-2.0, 4.9],
                [2.0, 4.9], [2.5, 5.1], [3.0, 5.0],
            ],
            dtype=np.float32,
        )
        probabilities = np.full((len(labels), 5), 0.05, dtype=np.float64)
        probabilities[np.arange(len(labels)), labels] = 0.8
        return {
            "features": features,
            "probabilities": probabilities,
            "labels": labels,
            "image_paths": [f"{prefix}_{index}.jpg" for index in range(len(labels))],
            "maturity_labels": np.full(len(labels), 2, dtype=np.int64),
            "maturity_names": ["Parabasal"] * len(labels),
        }

    def test_checkpoint_specs_require_locked_seeds(self):
        parsed = parse_checkpoint_specs(
            [
                "42=/tmp/seed42/best_model.pth",
                "7=/tmp/seed7/best_model.pth",
                "2026=/tmp/seed2026/best_model.pth",
            ]
        )
        self.assertEqual(set(parsed), {42, 7, 2026})
        self.assertEqual(LOCKED_SEEDS, (42, 7, 2026))
        with self.assertRaisesRegex(ValueError, "42, 7, and 2026"):
            parse_checkpoint_specs(["42=/tmp/a.pth", "7=/tmp/b.pth"])

    def test_checkpoint_specs_reject_duplicates_and_bad_syntax(self):
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint seed"):
            parse_checkpoint_specs(
                ["42=/tmp/a.pth", "42=/tmp/b.pth", "7=x", "2026=y"]
            )
        with self.assertRaisesRegex(ValueError, "SEED=PATH"):
            parse_checkpoint_specs(["42", "7=x", "2026=y"])

    def test_checkpoint_contract_locks_m0_caformer_letterbox_and_seed(self):
        run_args = {
            "experiment_name": "m0_caformer_letterbox_clean_v2_seed42",
            "variant": "m0",
            "model_name": "caformer_s18",
            "input_mode": "letterbox",
            "img_size": 224,
            "epochs": 30,
            "batch_size": 64,
            "lr": 0.0001,
            "backbone_lr_multiplier": 1.0,
            "weight_decay": 0.0001,
            "label_smoothing": 0.0,
            "lambda_screen": 0.0,
            "pretrained": True,
            "seed": 42,
        }
        checkpoint = {
            "variant": "m0",
            "model_name": "caformer_s18",
            "model_state": {},
            "args": dict(run_args),
        }
        validated = validate_checkpoint_contract(checkpoint, run_args, 42)
        self.assertEqual(validated["seed"], 42)

        bad_args = dict(run_args, input_mode="crop")
        with self.assertRaisesRegex(ValueError, "input_mode=letterbox"):
            validate_checkpoint_contract(checkpoint, bad_args, 42)

        wrong_run = dict(
            run_args,
            experiment_name="m0_caformer_letterbox_dlr01_clean_v2_seed42",
        )
        with self.assertRaisesRegex(ValueError, "promoted M0 run"):
            validate_checkpoint_contract(checkpoint, wrong_run, 42)

        conflicting_checkpoint = dict(checkpoint)
        conflicting_checkpoint["args"] = dict(run_args, lr=0.001)
        with self.assertRaisesRegex(ValueError, "embedded args.*lr"):
            validate_checkpoint_contract(conflicting_checkpoint, run_args, 42)

    def test_split_counts_are_locked_to_clean_v2_train_dev(self):
        self.assertEqual(validate_split_counts(7985, 998), (7985, 998))
        with self.assertRaisesRegex(ValueError, "expected train/dev counts 7985/998"):
            validate_split_counts(7984, 998)

    def test_manifest_hash_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "train.csv"
            csv_path.write_text("x\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                validate_manifest_hashes(
                    {"train_manifest_sha256": "0" * 64},
                    train_csv=csv_path,
                    dev_csv=None,
                )

    def test_manifest_hash_requires_every_requested_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dev.csv"
            csv_path.write_text("x\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev_manifest_sha256"):
                validate_manifest_hashes({}, train_csv=None, dev_csv=csv_path)

    def test_replayed_predictions_require_exact_argmax_and_row_count(self):
        probabilities = np.array([[0.9, 0.1], [0.4, 0.6]])
        saved = pd.DataFrame({"pred_label": [0, 0]})
        with self.assertRaisesRegex(ValueError, "argmax mismatch"):
            validate_replayed_predictions(probabilities, saved)
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_replayed_predictions(probabilities[:1], saved)

    def test_parser_has_no_calibration_or_test_data_arguments(self):
        help_text = build_parser().format_help()
        self.assertNotIn("--calibration", help_text)
        self.assertNotIn("--test_csv", help_text)
        self.assertIn("--train-csv", help_text)
        self.assertIn("--dev-csv", help_text)

    def test_nonempty_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "audit"
            out_dir.mkdir()
            (out_dir / "old.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare_output_directory(out_dir, overwrite=False)
            with self.assertRaisesRegex(ValueError, "ownership marker"):
                prepare_output_directory(out_dir, overwrite=True)

            owned_dir = Path(directory) / "owned-audit"
            prepare_output_directory(owned_dir, overwrite=False)
            (owned_dir / "old.txt").write_text("old", encoding="utf-8")
            prepare_output_directory(owned_dir, overwrite=True)
            self.assertFalse((owned_dir / "old.txt").exists())
            self.assertTrue((owned_dir / AUDIT_OWNER_FILENAME).is_file())

    def test_output_may_not_contain_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "audit"
            checkpoint = out_dir / "seed42" / "best_model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "contains protected input"):
                prepare_output_directory(
                    out_dir,
                    overwrite=True,
                    protected_inputs=[checkpoint],
                )

    def test_write_completed_marker_contains_locked_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            marker_path = write_completed_marker(out_dir, seeds=[42, 7, 2026])
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seeds"], [42, 7, 2026])
            self.assertEqual(payload["data_scope"], "clean_v2 train/dev only")
            self.assertFalse(payload["calibration_used"])
            self.assertFalse(payload["test_used"])
            self.assertFalse(payload["expert_review_used"])

    def test_synthetic_seed_analysis_builds_wide_cross_seed_audit(self):
        train = self._synthetic_extraction("train")
        dev = self._synthetic_extraction("dev")
        all_metrics = []
        frames = {"low_grade": [], "high_grade": []}
        for seed in LOCKED_SEEDS:
            metrics, seed_frames = _analyse_seed(seed, train, dev)
            all_metrics.extend(metrics)
            for pair_name, frame in seed_frames.items():
                frames[pair_name].append(frame)

        pair_metrics = pd.DataFrame(all_metrics)
        sample_audit = _build_sample_audit(frames)
        diagnostic = _diagnostic_summary(pair_metrics, sample_audit)

        self.assertEqual(len(pair_metrics), 6)
        self.assertEqual(len(sample_audit), 12)
        self.assertIn("seed2026_probe_score", sample_audit.columns)
        self.assertIn("seed2026_five_class_entropy", sample_audit.columns)
        self.assertTrue((sample_audit["m0_correct_seeds"] == 3).all())
        self.assertEqual(set(diagnostic["pairs"]), {"low_grade", "high_grade"})

    def test_diagnostic_does_not_hide_heterogeneous_seeds_in_the_mean(self):
        rows = pd.DataFrame(
            {
                "seed": [42, 7, 2026],
                "pair_name": ["low_grade"] * 3,
                "m0_macro_f1": [0.60, 0.55, 0.52],
                "linear_macro_f1": [0.90, 0.40, 0.40],
                "knn_macro_f1": [0.50, 0.45, 0.42],
                "linear_minus_m0_macro_f1": [0.30, -0.15, -0.12],
                "knn_minus_m0_macro_f1": [-0.10, -0.10, -0.10],
            }
        )
        samples = pd.DataFrame(
            {
                "pair_name": ["low_grade"] * 2,
                "m0_seed_unstable": [True, False],
                "m0_persistent_error": [False, True],
            }
        )
        diagnostic = _diagnostic_summary(rows, samples)
        payload = diagnostic["pairs"]["low_grade"]
        self.assertEqual(payload["signal"], "mixed_or_inconclusive")
        self.assertNotIn("initialization_sensitivity_signal", payload)

    def test_feature_array_validation_reopens_every_saved_array(self):
        with tempfile.TemporaryDirectory() as directory:
            feature_dir = Path(directory)
            for seed in LOCKED_SEEDS:
                np.save(
                    feature_dir / f"seed{seed}_train_features.npy",
                    np.ones((4, 3), dtype=np.float32),
                    allow_pickle=False,
                )
                np.save(
                    feature_dir / f"seed{seed}_dev_features.npy",
                    np.ones((2, 3), dtype=np.float32),
                    allow_pickle=False,
                )
            self.assertEqual(_validate_feature_arrays(feature_dir, 4, 2), 3)
            broken = np.load(
                feature_dir / "seed7_dev_features.npy", allow_pickle=False
            )
            broken[0, 0] = np.nan
            np.save(
                feature_dir / "seed7_dev_features.npy",
                broken,
                allow_pickle=False,
            )
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                _validate_feature_arrays(feature_dir, 4, 2)


if __name__ == "__main__":
    unittest.main()
