import csv
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import experiments.summarize_m0_pb1_results as summarizer

from experiments.summarize_m0_pb1_results import (
    COMPLETION_ARTIFACTS,
    SEED42_THRESHOLDS,
    THREE_SEED_THRESHOLDS,
    atomic_write_json,
    build_seed_delta,
    evaluate_seed42_gate,
    evaluate_three_seed_gate,
    main,
    parse_seed_dirs,
    write_completed_marker,
)


class M0PB1GateTests(unittest.TestCase):
    def delta(self, seed=42, low=0.006, high=0.006, abnormal=0.006,
              macro=0.006, purity=0.011, screen=0.0, coverage=0.99):
        return {
            "seed": seed,
            "low_pair_f1_delta": low,
            "high_pair_f1_delta": high,
            "boundary_composite_f1_delta": (low + high) / 2,
            "abnormal_macro_f1_delta": abnormal,
            "macro_f1_delta": macro,
            "mean_local_purity_delta": purity,
            "screen_sensitivity_delta": screen,
            "low_anchor_coverage": coverage,
            "high_anchor_coverage": coverage,
        }

    def test_threshold_constants_are_locked(self):
        self.assertEqual(SEED42_THRESHOLDS, {
            "boundary_composite_f1_delta_min": 0.005,
            "each_pair_delta_min": -0.003,
            "abnormal_macro_f1_delta_min": 0.003,
            "macro_f1_delta_min": -0.002,
            "mean_local_purity_delta_min": 0.010,
            "screen_sensitivity_delta_min": -0.005,
            "anchor_coverage_min": 0.95,
        })
        self.assertEqual(THREE_SEED_THRESHOLDS["positive_boundary_seed_count_min"], 2)
        self.assertEqual(THREE_SEED_THRESHOLDS["mean_macro_f1_delta_min_exclusive"], 0.0)

    def test_seed42_each_inclusive_check_passes_at_boundary_and_fails_below(self):
        mapping = {
            "boundary_composite": ("boundary_composite_f1_delta", 0.005),
            "low_pair_guard": ("low_pair_f1_delta", -0.003),
            "high_pair_guard": ("high_pair_f1_delta", -0.003),
            "abnormal_macro_f1": ("abnormal_macro_f1_delta", 0.003),
            "macro_f1_guard": ("macro_f1_delta", -0.002),
            "local_purity": ("mean_local_purity_delta", 0.010),
            "screen_guard": ("screen_sensitivity_delta", -0.005),
            "low_anchor_coverage": ("low_anchor_coverage", 0.95),
            "high_anchor_coverage": ("high_anchor_coverage", 0.95),
        }
        for check, (field, boundary) in mapping.items():
            with self.subTest(check=check):
                good = self.delta()
                if field == "boundary_composite_f1_delta":
                    good["low_pair_f1_delta"] = boundary
                    good["high_pair_f1_delta"] = boundary
                else:
                    good[field] = boundary
                if field in {"boundary_composite_f1_delta", "low_pair_f1_delta", "high_pair_f1_delta"}:
                    good["boundary_composite_f1_delta"] = (
                        good["low_pair_f1_delta"] + good["high_pair_f1_delta"]
                    ) / 2
                self.assertTrue(evaluate_seed42_gate(good)["checks"][check])
                bad = dict(good)
                if field == "boundary_composite_f1_delta":
                    bad["low_pair_f1_delta"] = math.nextafter(boundary, -math.inf)
                    bad["high_pair_f1_delta"] = math.nextafter(boundary, -math.inf)
                else:
                    bad[field] = math.nextafter(boundary, -math.inf)
                if field in {"boundary_composite_f1_delta", "low_pair_f1_delta", "high_pair_f1_delta"}:
                    bad["boundary_composite_f1_delta"] = (
                        bad["low_pair_f1_delta"] + bad["high_pair_f1_delta"]
                    ) / 2
                self.assertFalse(evaluate_seed42_gate(bad)["checks"][check])

    def test_three_seed_boundary_rules_and_strict_positive_rules(self):
        rows = [self.delta(seed=s) for s in (42, 7, 2026)]
        self.assertTrue(evaluate_three_seed_gate(rows)["passed"])
        cases = (
            ("mean_boundary_composite", "boundary_composite_f1_delta", 0.005, False),
            ("positive_boundary_seeds", "boundary_composite_f1_delta", 0.0, True),
            ("low_pair_mean_positive", "low_pair_f1_delta", 0.0, True),
            ("high_pair_mean_positive", "high_pair_f1_delta", 0.0, True),
            ("mean_abnormal_macro_f1", "abnormal_macro_f1_delta", 0.005, False),
            ("positive_abnormal_seeds", "abnormal_macro_f1_delta", 0.0, True),
            ("mean_macro_f1_positive", "macro_f1_delta", 0.0, True),
            ("macro_f1_per_seed_guard", "macro_f1_delta", -0.002, False),
            ("mean_local_purity", "mean_local_purity_delta", 0.010, False),
            ("positive_purity_seeds", "mean_local_purity_delta", 0.0, True),
            ("mean_screen_guard", "screen_sensitivity_delta", -0.005, False),
        )
        for check, field, threshold, strict in cases:
            with self.subTest(check=check):
                changed = [dict(row) for row in rows]
                for row in changed:
                    if field == "boundary_composite_f1_delta":
                        row["low_pair_f1_delta"] = threshold
                        row["high_pair_f1_delta"] = threshold
                    else:
                        row[field] = threshold
                    if field in {"boundary_composite_f1_delta", "low_pair_f1_delta", "high_pair_f1_delta"}:
                        row["boundary_composite_f1_delta"] = (
                            row["low_pair_f1_delta"] + row["high_pair_f1_delta"]
                        ) / 2
                report = evaluate_three_seed_gate(changed)
                self.assertEqual(report["checks"][check], not strict)

    def test_three_seed_requires_exact_seed_set_once_each(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_three_seed_gate([self.delta(42), self.delta(42), self.delta(7)])

    def test_three_seed_decision_labels(self):
        passing = [self.delta(seed=s) for s in (42, 7, 2026)]
        self.assertEqual(evaluate_three_seed_gate(passing)["decision"], "PROMOTE_TO_MAINLINE_CANDIDATE")
        mechanism_only = [dict(row, macro_f1_delta=0.0) for row in passing]
        self.assertEqual(evaluate_three_seed_gate(mechanism_only)["decision"], "MECHANISM_POSITIVE_NOT_MAINLINE")
        negative = [dict(row, low_pair_f1_delta=-0.01,
                         boundary_composite_f1_delta=-0.002) for row in passing]
        self.assertEqual(evaluate_three_seed_gate(negative)["decision"], "NEGATIVE_SINGLE_VARIABLE_ABLATION")

    def test_gate_rejects_missing_nonfinite_and_inconsistent_delta(self):
        for altered in (
            {k: v for k, v in self.delta().items() if k != "macro_f1_delta"},
            dict(self.delta(), macro_f1_delta=float("nan")),
            dict(self.delta(), boundary_composite_f1_delta=0.1),
        ):
            with self.subTest(altered=altered):
                with self.assertRaises(ValueError):
                    evaluate_seed42_gate(altered)


class ArtifactAndCliTests(unittest.TestCase):
    EVIDENCE = {"calibration_used": False, "test_used": False}
    def write_json(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def metrics(self, macro=0.80, abnormal=(0.7, 0.7, 0.7, 0.7), screen=1.0, seed=None):
        payload = {
            "macro_f1": macro, "screen_sensitivity": screen,
            "asc_us_f1": abnormal[0], "lsil_f1": abnormal[1],
            "asc_h_f1": abnormal[2], "hsil_f1": abnormal[3],
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    def make_inputs(self, root, seed=42):
        m0 = root / "m0"; pb1 = root / "pb1"; m0.mkdir(); pb1.mkdir()
        train_csv = "data/local/csv_files_clean_v2/train_xudata_tbs5_clean_v2.csv"
        dev_csv = "data/local/csv_files_clean_v2/dev_xudata_tbs5_clean_v2.csv"
        m0_args = {
            "experiment_name": f"m0_caformer_letterbox_clean_v2_seed{seed}", "variant": "m0",
            "model_name": "caformer_s18", "train_csv": train_csv, "dev_csv": dev_csv,
            "out_dir": str(m0), "input_mode": "letterbox", "img_size": 224,
            "epochs": 30, "batch_size": 64, "lr": 0.0001, "backbone_lr_multiplier": 1.0,
            "weight_decay": 0.0001, "label_smoothing": 0.0, "num_workers": 8, "seed": seed,
            "lambda_screen": 0.0, "pretrained": True, "amp": True,
        }
        pb1_args = {
            "experiment_name": f"m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed{seed}",
            "variant": "m0", "model_name": "caformer_s18", "train_csv": train_csv,
            "dev_csv": dev_csv, "out_dir": str(pb1), "input_mode": "letterbox",
            "img_size": 224, "epochs": 30, "batch_size": 64, "lr": 0.0001,
            "backbone_lr_multiplier": 1.0, "weight_decay": 0.0001,
            "label_smoothing": 0.0, "num_workers": 8, "seed": seed,
            "lambda_screen": 0.0, "pretrained": True, "amp": True,
            "boundary_loss": "pair_boundary_supcon", "temperature": 0.1, "lambda_pb": 0.1,
        }
        source_root = Path(summarizer.__file__).resolve().parents[1]
        hashes = "a" * 64
        environment = {
            "train_manifest_sha256": hashes, "dev_manifest_sha256": "b" * 64,
            "code_sha256": {
                name: summarizer._sha256(source_root / name)
                for name in (
                    "experiments/train_tbs_stage1.py", "experiments/tbs/losses.py",
                    "experiments/tbs/pairboundary_evaluation.py",
                )
            },
        }
        self.write_json(m0 / "args.json", m0_args)
        self.write_json(pb1 / "args.json", pb1_args)
        self.write_json(m0 / "environment.json", environment)
        self.write_json(pb1 / "environment.json", environment)
        self.write_json(m0 / "dev_metrics.json", self.metrics(seed=seed))
        self.write_json(pb1 / "dev_metrics.json", self.metrics(
            macro=0.81, abnormal=(0.71, 0.71, 0.71, 0.71), screen=1.0, seed=seed))
        self.write_json(pb1 / "best_metrics.json", {
            "epoch": 2, **self.metrics(macro=0.81, abnormal=(0.71, 0.71, 0.71, 0.71), screen=1.0),
            "boundary_composite_macro_f1": 0.81, "boundary_mean_local_purity": 0.75,
        })
        self.write_csv(pb1 / "metrics.csv", ["epoch", "train_pb_low_coverage", "train_pb_high_coverage"], [
            {"epoch": 1, "train_pb_low_coverage": 0.5, "train_pb_high_coverage": 0.5},
            {"epoch": 2, "train_pb_low_coverage": 0.97, "train_pb_high_coverage": 0.98},
        ])
        self.write_csv(pb1 / "boundary_pair_metrics.csv", ["pair_name", "pairwise_macro_f1"], [
            {"pair_name": "low_grade", "pairwise_macro_f1": 0.80},
            {"pair_name": "high_grade", "pairwise_macro_f1": 0.82},
        ])
        self.write_csv(pb1 / "boundary_geometry_metrics.csv", ["pair_name", "mean_local_purity"], [
            {"pair_name": "low_grade", "mean_local_purity": 0.74},
            {"pair_name": "high_grade", "mean_local_purity": 0.76},
        ])
        audit = root / "pair_metrics.csv"
        self.write_csv(audit, ["seed", "pair_name", "m0_macro_f1", "mean_local_purity"], [
            {"seed": audit_seed, "pair_name": pair, "m0_macro_f1": value, "mean_local_purity": purity}
            for audit_seed in (42, 7, 2026)
            for pair, value, purity in (("low_grade", 0.77, 0.72), ("high_grade", 0.81, 0.73))
        ])
        return m0, pb1, audit

    def checkpoint(self, pb1, epoch=2, metrics=None, args=None):
        args = json.loads((pb1 / "args.json").read_text()) if args is None else args
        metrics = self.metrics(macro=0.81, abnormal=(0.71, 0.71, 0.71, 0.71), screen=1.0) if metrics is None else metrics
        return {"epoch": epoch, "variant": "m0", "model_name": "caformer_s18",
                "model_state": {"weight": object()}, "args": args, "metrics": metrics}

    def test_formal_contract_binds_seed_manifests_objective_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, _ = self.make_inputs(root, seed=42)
            loader = lambda path: self.checkpoint(pb1)
            manifest_hasher = lambda path: "a" * 64 if "train_" in str(path) else "b" * 64
            evidence = summarizer._validate_formal_run_contract(42, m0, pb1, loader, manifest_hasher)
            self.assertEqual(evidence, {"calibration_used": False, "test_used": False})
            archived_m0_args = json.loads((m0 / "args.json").read_text())
            archived_m0_args.pop("backbone_lr_multiplier")
            self.write_json(m0 / "args.json", archived_m0_args)
            self.assertEqual(
                summarizer._validate_formal_run_contract(42, m0, pb1, loader, manifest_hasher),
                {"calibration_used": False, "test_used": False},
            )
            archived_m0_args["backbone_lr_multiplier"] = 2.0
            self.write_json(m0 / "args.json", archived_m0_args)
            with self.assertRaises(ValueError):
                summarizer._validate_formal_run_contract(42, m0, pb1, loader, manifest_hasher)
            archived_m0_args["backbone_lr_multiplier"] = 1.0
            self.write_json(m0 / "args.json", archived_m0_args)
            with self.assertRaisesRegex(ValueError, "best_model"):
                summarizer._validate_formal_run_contract(
                    42, m0, pb1,
                    lambda path: (_ for _ in ()).throw(RuntimeError("corrupt checkpoint")),
                    manifest_hasher,
                )
            for checkpoint_metrics in ({}, {"macro_f1": 0.81}):
                with self.subTest(checkpoint_metrics=checkpoint_metrics), self.assertRaises(ValueError):
                    summarizer._validate_formal_run_contract(
                        42, m0, pb1,
                        lambda path, checkpoint_metrics=checkpoint_metrics: self.checkpoint(pb1, metrics=checkpoint_metrics),
                        manifest_hasher,
                    )
            for checkpoint in (
                self.checkpoint(pb1, epoch=1),
                self.checkpoint(pb1, args=dict(json.loads((pb1 / "args.json").read_text()), seed=7)),
                self.checkpoint(pb1, metrics=self.metrics(macro=0.80, abnormal=(0.71, 0.71, 0.71, 0.71), screen=1.0)),
            ):
                with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                    summarizer._validate_formal_run_contract(42, m0, pb1, lambda path, checkpoint=checkpoint: checkpoint, manifest_hasher)
            for path, mutate in (
                (m0 / "args.json", lambda p: p.update(seed=7)),
                (m0 / "args.json", lambda p: p.update(input_mode="crop")),
                (pb1 / "args.json", lambda p: p.update(boundary_loss="wrong")),
                (pb1 / "environment.json", lambda p: p.update(train_manifest_sha256="f" * 64)),
                (pb1 / "environment.json", lambda p: p["code_sha256"].update({"experiments/tbs/losses.py": "0" * 64})),
            ):
                payload = json.loads(path.read_text()); mutate(payload); self.write_json(path, payload)
                with self.subTest(path=path), self.assertRaises(ValueError):
                    summarizer._validate_formal_run_contract(42, m0, pb1, loader, manifest_hasher)
                if path.name == "args.json" and path.parent == m0:
                    self.write_json(path, {
                        "experiment_name": "m0_caformer_letterbox_clean_v2_seed42", "variant": "m0", "model_name": "caformer_s18",
                        "train_csv": "data/local/csv_files_clean_v2/train_xudata_tbs5_clean_v2.csv", "dev_csv": "data/local/csv_files_clean_v2/dev_xudata_tbs5_clean_v2.csv",
                        "out_dir": str(m0), "input_mode": "letterbox", "img_size": 224, "epochs": 30, "batch_size": 64, "lr": 0.0001,
                        "backbone_lr_multiplier": 1.0, "weight_decay": 0.0001, "label_smoothing": 0.0, "num_workers": 8, "seed": 42,
                        "lambda_screen": 0.0, "pretrained": True, "amp": True})
                elif path.name == "args.json":
                    self.make_inputs # restoration below avoids a test-only production path
                    payload = json.loads((root / "pb1" / "args.json").read_text())
                    payload["boundary_loss"] = "pair_boundary_supcon"; self.write_json(path, payload)
                else:
                    payload = json.loads((root / "m0" / "environment.json").read_text()); self.write_json(path, payload)

    def test_cli_never_seals_a_run_that_fails_formal_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root, seed=42)
            payload = json.loads((m0 / "args.json").read_text()); payload["seed"] = 7
            self.write_json(m0 / "args.json", payload)
            self.materialize_completion_artifacts(pb1)
            output = pb1 / "gate_report.json"
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"42={m0}",
                    "--pb1-run", f"42={pb1}", "--out-json", str(output)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertFalse((pb1 / "completed.json").exists())

    def test_procedural_seed_rejects_subthreshold_anchor_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root, seed=7)
            self.write_csv(pb1 / "metrics.csv", ["epoch", "train_pb_low_coverage", "train_pb_high_coverage"], [
                {"epoch": 1, "train_pb_low_coverage": 0.5, "train_pb_high_coverage": 0.5},
                {"epoch": 2, "train_pb_low_coverage": 0.94, "train_pb_high_coverage": 0.98},
            ])
            self.materialize_completion_artifacts(pb1)
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"7={m0}",
                    "--pb1-run", f"7={pb1}", "--out-json", str(pb1 / "gate_report.json")]
            with patch("sys.argv", argv), patch.object(summarizer, "_validate_formal_run_contract", return_value={"calibration_used": False, "test_used": False}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertFalse((pb1 / "completed.json").exists())

    def test_malformed_raw_run_spec_never_overwrites_its_rhs_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            for option, target in (("--m0-run", m0 / "dev_metrics.json"), ("--pb1-run", pb1 / "dev_metrics.json")):
                before = target.read_bytes()
                argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"42={m0}",
                        "--pb1-run", f"42={pb1}", "--out-json", str(target)]
                index = argv.index(option); argv[index + 1] = f"42={target}"
                with self.subTest(option=option), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 2)
                self.assertEqual(target.read_bytes(), before)

    def test_raw_run_token_without_equals_never_overwrites_the_token_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            target = m0 / "dev_metrics.json"; before = target.read_bytes()
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", str(target),
                    "--pb1-run", f"42={pb1}", "--out-json", str(target)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(target.read_bytes(), before)

    def test_three_seed_aggregate_requires_each_sealed_current_gate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mappings = []
            for seed in (42, 7, 2026):
                sub = root / str(seed); sub.mkdir(); m0, pb1, audit = self.make_inputs(sub, seed)
                self.materialize_completion_artifacts(pb1)
                summarizer.atomic_write_json(pb1 / "gate_report.json", summarizer._single_seed_report(build_seed_delta(seed, m0, pb1, audit)))
                write_completed_marker(pb1, seed, self.EVIDENCE)
                mappings.append((seed, m0, pb1))
            base = ["prog", "--m0-audit-pair-metrics", str(audit), "--out-json", str(root / "aggregate.json")]
            for seed, m0, _ in mappings: base += ["--m0-run", f"{seed}={m0}"]
            for seed, _, pb1 in mappings: base += ["--pb1-run", f"{seed}={pb1}"]
            failed = summarizer._single_seed_report(build_seed_delta(42, mappings[0][1], mappings[0][2], audit))
            failed.update(passed=False, decision="NEGATIVE_SINGLE_VARIABLE_ABLATION")
            (mappings[0][2] / "gate_report.json").write_text(json.dumps(failed), encoding="utf-8")
            (mappings[0][2] / "completed.json").unlink(); write_completed_marker(mappings[0][2], 42, self.EVIDENCE)
            with patch("sys.argv", base), patch.object(summarizer, "_validate_formal_run_contract", return_value={"calibration_used": False, "test_used": False}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads((root / "aggregate.json").read_text())["decision"], "INVALID_RUN")

    def materialize_completion_artifacts(self, run_dir):
        for name in COMPLETION_ARTIFACTS:
            path = run_dir / name
            if not path.exists():
                path.write_bytes(b"artifact:" + name.encode("ascii"))

    def test_parse_seed_dirs_accepts_any_single_seed_and_rejects_partial_multi_seed(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_seed_dirs(["42=a", "42=b"])
        with self.assertRaisesRegex(ValueError, "SEED=DIR"):
            parse_seed_dirs(["x=a"])
        self.assertEqual(set(parse_seed_dirs(["7=a"], require_dirs=False)), {7})
        self.assertEqual(set(parse_seed_dirs(["2026=a"], require_dirs=False)), {2026})
        with self.assertRaisesRegex(ValueError, "one seed or exactly"):
            parse_seed_dirs(["42=a", "7=b"])

    def test_build_delta_uses_frozen_audit_join_and_best_epoch_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            m0, pb1, audit = self.make_inputs(Path(tmp))
            delta = build_seed_delta(42, m0, pb1, audit)
            self.assertAlmostEqual(delta["low_pair_f1_delta"], 0.03)
            self.assertAlmostEqual(delta["high_pair_f1_delta"], 0.01)
            self.assertEqual(delta["low_anchor_coverage"], 0.97)
            self.assertEqual(delta["high_anchor_coverage"], 0.98)

    def test_build_delta_rejects_duplicate_wrong_missing_and_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            self.write_csv(pb1 / "boundary_pair_metrics.csv", ["pair_name", "pairwise_macro_f1"], [
                {"pair_name": "low_grade", "pairwise_macro_f1": 0.8},
                {"pair_name": "low_grade", "pairwise_macro_f1": 0.8},
            ])
            with self.assertRaises(ValueError):
                build_seed_delta(42, m0, pb1, audit)

    def test_build_delta_rejects_duplicate_best_epoch_and_global_audit_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            self.write_csv(pb1 / "metrics.csv", ["epoch", "train_pb_low_coverage", "train_pb_high_coverage"], [
                {"epoch": 2, "train_pb_low_coverage": 0.97, "train_pb_high_coverage": 0.98},
                {"epoch": 2, "train_pb_low_coverage": 0.97, "train_pb_high_coverage": 0.98},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate epoch"):
                build_seed_delta(42, m0, pb1, audit)
            second = root / "second"; second.mkdir()
            m0, pb1, audit = self.make_inputs(second)
            with audit.open("a", encoding="utf-8") as handle:
                handle.write("7,wrong,0.7,0.7\n")
            with self.assertRaisesRegex(ValueError, "wrong pair_name"):
                build_seed_delta(42, m0, pb1, audit)
            self.write_csv(pb1 / "boundary_pair_metrics.csv", ["pair_name", "pairwise_macro_f1"], [
                {"pair_name": "wrong", "pairwise_macro_f1": 0.8},
                {"pair_name": "high_grade", "pairwise_macro_f1": "nan"},
            ])
            with self.assertRaises(ValueError):
                build_seed_delta(42, m0, pb1, audit)

    def test_build_delta_requires_complete_audit_best_summary_and_all_valid_metric_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            with audit.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:-1]
            self.write_csv(audit, ["seed", "pair_name", "m0_macro_f1", "mean_local_purity"], rows)
            with self.assertRaisesRegex(ValueError, "exactly all"):
                build_seed_delta(42, m0, pb1, audit)
            self.write_csv(audit, ["seed", "pair_name", "m0_macro_f1", "mean_local_purity"], [
                {"seed": audit_seed, "pair_name": pair, "m0_macro_f1": value, "mean_local_purity": purity}
                for audit_seed in (42, 7, 2026)
                for pair, value, purity in (("low_grade", 0.77, 0.72), ("high_grade", 0.81, 0.73))
            ])
            self.write_json(pb1 / "best_metrics.json", {"epoch": 2})
            with self.assertRaisesRegex(ValueError, "boundary_composite_macro_f1"):
                build_seed_delta(42, m0, pb1, audit)
            for value in (float("nan"), 0.0):
                self.write_json(pb1 / "best_metrics.json", {
                    "epoch": 2, "boundary_composite_macro_f1": value,
                    "boundary_mean_local_purity": 0.75,
                })
                with self.assertRaises(ValueError):
                    build_seed_delta(42, m0, pb1, audit)
            self.write_json(pb1 / "best_metrics.json", {
                "epoch": 2, "boundary_composite_macro_f1": 0.81,
                "boundary_mean_local_purity": 0.75,
            })
            self.write_csv(pb1 / "metrics.csv", ["epoch", "train_pb_low_coverage", "train_pb_high_coverage"], [
                {"epoch": 1, "train_pb_low_coverage": "nan", "train_pb_high_coverage": 0.5},
                {"epoch": 2, "train_pb_low_coverage": 0.97, "train_pb_high_coverage": 0.98},
            ])
            with self.assertRaises(ValueError):
                build_seed_delta(42, m0, pb1, audit)

    def test_build_delta_rejects_duplicate_nonbest_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            self.write_csv(pb1 / "metrics.csv", ["epoch", "train_pb_low_coverage", "train_pb_high_coverage"], [
                {"epoch": 1, "train_pb_low_coverage": 0.5, "train_pb_high_coverage": 0.5},
                {"epoch": 1, "train_pb_low_coverage": 0.6, "train_pb_high_coverage": 0.6},
                {"epoch": 2, "train_pb_low_coverage": 0.97, "train_pb_high_coverage": 0.98},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate epoch"):
                build_seed_delta(42, m0, pb1, audit)

    def test_atomic_json_replaces_target_with_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"; target.write_text("old", encoding="utf-8")
            atomic_write_json(target, {"b": 2, "a": 1})
            self.assertEqual(target.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertFalse(any(path.name.startswith(".report.json.") for path in target.parent.iterdir()))

    def test_completion_marker_requires_exact_nonempty_regular_artifacts_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); self.materialize_completion_artifacts(run)
            marker = write_completed_marker(run, 42, self.EVIDENCE)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(list(payload), ["artifact_sha256", "calibration_used", "seed", "status", "test_used"])
            self.assertEqual(set(payload["artifact_sha256"]), set(COMPLETION_ARTIFACTS))
            self.assertEqual(len(payload["artifact_sha256"]), 13)
            self.assertEqual(payload["seed"], 42)
            self.assertFalse(payload["calibration_used"])
            self.assertFalse(payload["test_used"])
            self.assertEqual(write_completed_marker(run, 42, self.EVIDENCE), marker)
            (run / "metrics.csv").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "refuse"):
                write_completed_marker(run, 42, self.EVIDENCE)

    def test_marker_rejects_tampered_fields_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); self.materialize_completion_artifacts(run)
            marker = write_completed_marker(run, 42, self.EVIDENCE)
            original = json.loads(marker.read_text(encoding="utf-8"))
            for field, value in (("status", "BAD"), ("seed", 7),
                                 ("calibration_used", True), ("test_used", True)):
                payload = dict(original); payload[field] = value
                marker.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    summarizer._validate_current_marker(marker, run, 42)
            payload = dict(original); payload["artifact_sha256"] = {}
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                summarizer._validate_current_marker(marker, run, 42)

    def test_completion_marker_requires_validated_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); self.materialize_completion_artifacts(run)
            for evidence in (None, {}, {"calibration_used": False}, {"calibration_used": True, "test_used": False}):
                with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                    write_completed_marker(run, 42, evidence)

    def test_completion_marker_rejects_empty_and_symlink_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); self.materialize_completion_artifacts(run)
            (run / "console.log").write_bytes(b"")
            with self.assertRaises(ValueError):
                write_completed_marker(run, 42, self.EVIDENCE)
            (run / "console.log").write_bytes(b"log")
            target = run / "real.txt"; target.write_bytes(b"real")
            link = run / "args.json"; link.unlink()
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink unsupported: {exc}")
            with self.assertRaises(ValueError):
                write_completed_marker(run, 42, self.EVIDENCE)

    def test_completed_marker_protects_gate_report_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); self.materialize_completion_artifacts(run)
            write_completed_marker(run, 42, self.EVIDENCE)
            from experiments.summarize_m0_pb1_results import _write_single_report_then_marker
            with self.assertRaisesRegex(ValueError, "protected"):
                _write_single_report_then_marker(run, 42, {"passed": False})

    def test_cli_single_writes_gate_report_before_completion_and_invalid_writes_only_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            self.materialize_completion_artifacts(pb1)
            output = pb1 / "gate_report.json"
            buffer = io.StringIO()
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"42={m0}",
                    "--pb1-run", f"42={pb1}", "--out-json", str(output)]
            with patch("sys.argv", argv), patch.object(summarizer, "_validate_formal_run_contract", return_value={"calibration_used": False, "test_used": False}), patch.object(summarizer, "atomic_write_json", wraps=atomic_write_json) as writer, redirect_stdout(buffer):
                self.assertEqual(main(), 0)
            self.assertTrue((pb1 / "completed.json").exists())
            self.assertEqual(json.loads(output.read_text())["decision"], "ELIGIBLE_FOR_SEEDS_7_2026")
            self.assertEqual([Path(call.args[0]).name for call in writer.call_args_list], ["gate_report.json", "completed.json"])
            (pb1 / "metrics.csv").write_text("bad", encoding="utf-8")
            invalid = root / "invalid.json"
            argv[-1] = str(invalid)
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads(invalid.read_text())["decision"], "INVALID_RUN")
            self.assertFalse((pb1 / "completed.json").read_bytes() == b"")

    def test_cli_seed7_and_seed2026_write_procedural_completion_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (7, 2026):
                with self.subTest(seed=seed):
                    root = Path(tmp) / str(seed); root.mkdir()
                    m0, pb1, audit = self.make_inputs(root, seed)
                    self.materialize_completion_artifacts(pb1)
                    output = pb1 / "gate_report.json"
                    argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"{seed}={m0}",
                            "--pb1-run", f"{seed}={pb1}", "--out-json", str(output)]
                    with patch("sys.argv", argv), patch.object(summarizer, "_validate_formal_run_contract", return_value={"calibration_used": False, "test_used": False}), redirect_stdout(io.StringIO()):
                        self.assertEqual(main(), 0)
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(payload["decision"], "ELIGIBLE_FOR_THREE_SEED_AGGREGATION")
                    self.assertFalse(payload["final_promotion_decision"])
                    self.assertTrue((pb1 / "completed.json").exists())

    def test_cli_invalid_run_never_creates_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, pb1, audit = self.make_inputs(root)
            out = root / "invalid.json"
            (pb1 / "metrics.csv").write_text("epoch\n2\n", encoding="utf-8")
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"42={m0}",
                    "--pb1-run", f"42={pb1}", "--out-json", str(out)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads(out.read_text())["decision"], "INVALID_RUN")
            self.assertFalse((pb1 / "completed.json").exists())

    def test_cli_three_seed_requires_current_markers_and_writes_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mappings = []; audits = []
            for seed in (42, 7, 2026):
                sub = root / str(seed); sub.mkdir(); m0, pb1, audit = self.make_inputs(sub, seed)
                self.materialize_completion_artifacts(pb1)
                summarizer.atomic_write_json(pb1 / "gate_report.json", summarizer._single_seed_report(build_seed_delta(seed, m0, pb1, audit)))
                write_completed_marker(pb1, seed, self.EVIDENCE)
                mappings.append((seed, m0, pb1)); audits.append(audit)
            audit = audits[0]
            out = root / "aggregate.json"
            argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--out-json", str(out)]
            for seed, m0, _ in mappings: argv += ["--m0-run", f"{seed}={m0}"]
            for seed, _, pb1 in mappings: argv += ["--pb1-run", f"{seed}={pb1}"]
            reports_before = {
                seed: (pb1 / "gate_report.json").read_bytes()
                for seed, _, pb1 in mappings
            }
            with patch("sys.argv", argv), patch.object(summarizer, "_validate_formal_run_contract", return_value={"calibration_used": False, "test_used": False}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertTrue(out.exists())
            self.assertEqual(reports_before, {
                seed: (pb1 / "gate_report.json").read_bytes()
                for seed, _, pb1 in mappings
            })

    def test_three_seed_rejects_missing_stale_markers_and_malicious_output_paths_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mappings = []
            for seed in (42, 7, 2026):
                sub = root / str(seed); sub.mkdir(); m0, pb1, audit = self.make_inputs(sub, seed)
                self.materialize_completion_artifacts(pb1); write_completed_marker(pb1, seed, self.EVIDENCE)
                mappings.append((seed, m0, pb1))
            audit = root / "audit.csv"
            with (root / "42" / "pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.write_csv(audit, ["seed", "pair_name", "m0_macro_f1", "mean_local_purity"], rows)
            base = ["prog", "--m0-audit-pair-metrics", str(audit)]
            for seed, m0, _ in mappings: base += ["--m0-run", f"{seed}={m0}"]
            for seed, _, pb1 in mappings: base += ["--pb1-run", f"{seed}={pb1}"]
            protected = mappings[0][2]
            for name in ("gate_report.json", "completed.json", "metrics.csv"):
                target = protected / name
                before = {path: path.read_bytes() for _, _, run in mappings for path in (run / "gate_report.json", run / "completed.json", run / "metrics.csv")}
                with patch("sys.argv", base + ["--out-json", str(target)]), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 2)
                self.assertEqual(before, {path: path.read_bytes() for path in before})
                for seed, _, run in mappings:
                    summarizer._validate_current_marker(run / "completed.json", run, seed)
            (mappings[1][2] / "completed.json").unlink()
            with patch("sys.argv", base + ["--out-json", str(root / "aggregate.json")]), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads((root / "aggregate.json").read_text())["decision"], "INVALID_RUN")

    def test_three_seed_never_overwrites_m0_or_audit_inputs_even_for_invalid_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mappings = []
            for seed in (42, 7, 2026):
                sub = root / str(seed); sub.mkdir(); m0, pb1, audit = self.make_inputs(sub, seed)
                self.materialize_completion_artifacts(pb1); write_completed_marker(pb1, seed, self.EVIDENCE)
                mappings.append((seed, m0, pb1))
            audit = root / "audit.csv"
            with (root / "42" / "pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
                self.write_csv(audit, ["seed", "pair_name", "m0_macro_f1", "mean_local_purity"], list(csv.DictReader(handle)))
            base = ["prog", "--m0-audit-pair-metrics", str(audit)]
            for seed, m0, _ in mappings: base += ["--m0-run", f"{seed}={m0}"]
            for seed, _, pb1 in mappings: base += ["--pb1-run", f"{seed}={pb1}"]
            for target in (audit, mappings[0][1] / "dev_metrics.json", mappings[0][1] / "nested" / "out.json"):
                if target.parent != mappings[0][1]: target.parent.mkdir(exist_ok=True)
                if not target.exists(): target.write_bytes(b"m0-input")
                before = target.read_bytes()
                with patch("sys.argv", base + ["--out-json", str(target)]), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 2)
                self.assertEqual(target.read_bytes(), before)
            audit.write_text("malformed", encoding="utf-8")
            before = audit.read_bytes()
            with patch("sys.argv", base + ["--out-json", str(audit)]), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(audit.read_bytes(), before)

    def test_parse_failure_after_valid_m0_never_overwrites_audit_or_m0_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m0, _, audit = self.make_inputs(root)
            for target in (audit, m0 / "nested.json"):
                if not target.exists(): target.write_bytes(b"m0-input")
                before = target.read_bytes()
                argv = ["prog", "--m0-audit-pair-metrics", str(audit), "--m0-run", f"42={m0}",
                        "--pb1-run", "7=does-not-exist", "--out-json", str(target)]
                with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 2)
                self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
