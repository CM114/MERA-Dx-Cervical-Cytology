"""Audit frozen M0 feature separability on clean_v2 train/dev only.

The script deliberately exposes no calibration- or test-set arguments.  It
replays three locked M0 checkpoints, verifies their original dev predictions,
and then runs fixed train-only probes on the two prespecified abnormal-class
boundaries.  The probes are diagnostics; they do not promote or replace M0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.boundary_audit import (  # noqa: E402
    PAIR_SPECS,
    aggregate_sample_frames,
    binary_metrics,
    build_seed_sample_frame,
    fit_knn_probe,
    fit_linear_probe,
    geometry_metrics,
    select_pair,
)
from experiments.tbs.labels import DIAGNOSIS_NAMES  # noqa: E402


LOCKED_SEEDS = (42, 7, 2026)
LOCKED_VARIANT = "m0"
LOCKED_MODEL_NAME = "caformer_s18"
LOCKED_INPUT_MODE = "letterbox"
LOCKED_IMAGE_SIZE = 224
ANALYSIS_SEED = 0
EXPECTED_TRAIN_SAMPLES = 7985
EXPECTED_DEV_SAMPLES = 998
AUDIT_OWNER_FILENAME = ".m0_frozen_boundary_audit_owner.json"
AUDIT_OWNER_SCHEMA = "xudata-tbs5-m0-frozen-boundary-audit-v1"
HEAD_GAIN_THRESHOLD = 0.02
LOW_PROBE_F1_MAX = 0.80
STABLE_PROBE_F1_SPREAD_MAX = 0.05


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def parse_checkpoint_specs(values):
    parsed = {}
    for value in values:
        seed_text, separator, path_text = str(value).partition("=")
        if not separator or not seed_text.strip() or not path_text.strip():
            raise ValueError("checkpoint must use SEED=PATH syntax")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("checkpoint seed must be an integer in SEED=PATH") from exc
        if seed in parsed:
            raise ValueError(f"duplicate checkpoint seed: {seed}")
        parsed[seed] = Path(path_text)
    if set(parsed) != set(LOCKED_SEEDS):
        raise ValueError("checkpoints must contain exactly seeds 42, 7, and 2026")
    return parsed


def validate_checkpoint_contract(checkpoint, run_args, expected_seed):
    """Validate metadata without constructing a model or reading any images."""

    if not isinstance(checkpoint, dict) or not isinstance(run_args, dict):
        raise ValueError("checkpoint and args.json must both be dictionaries")
    if checkpoint.get("variant") != LOCKED_VARIANT:
        raise ValueError("checkpoint must declare variant=m0")
    if checkpoint.get("model_name") != LOCKED_MODEL_NAME:
        raise ValueError("checkpoint must declare model_name=caformer_s18")
    expected_experiment = f"m0_caformer_letterbox_clean_v2_seed{expected_seed}"
    locked_values = {
        "experiment_name": expected_experiment,
        "variant": LOCKED_VARIANT,
        "model_name": LOCKED_MODEL_NAME,
        "input_mode": LOCKED_INPUT_MODE,
        "img_size": LOCKED_IMAGE_SIZE,
        "seed": int(expected_seed),
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-4,
        "backbone_lr_multiplier": 1.0,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "lambda_screen": 0.0,
        "pretrained": True,
    }

    def validate_args_payload(payload, source):
        if not isinstance(payload, dict):
            raise ValueError(f"{source} must be a dictionary")
        for key, expected in locked_values.items():
            default = 1.0 if key == "backbone_lr_multiplier" else None
            actual = payload.get(key, default)
            if isinstance(expected, float):
                try:
                    matches = bool(
                        np.isclose(float(actual), expected, rtol=0.0, atol=1e-12)
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                if key == "input_mode":
                    raise ValueError(f"{source} must declare input_mode=letterbox")
                if key == "experiment_name":
                    raise ValueError(
                        f"{source} does not identify the promoted M0 run: "
                        f"expected {expected_experiment}"
                    )
                raise ValueError(
                    f"{source} {key}={actual!r} does not match promoted M0 value "
                    f"{expected!r}"
                )

    validate_args_payload(run_args, "args.json")
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_state")
    embedded_args = checkpoint.get("args")
    if embedded_args is None:
        raise ValueError("checkpoint is missing embedded args")
    validate_args_payload(embedded_args, "checkpoint embedded args")
    return {
        "seed": int(expected_seed),
        "variant": LOCKED_VARIANT,
        "model_name": LOCKED_MODEL_NAME,
        "input_mode": LOCKED_INPUT_MODE,
        "img_size": LOCKED_IMAGE_SIZE,
    }


def validate_manifest_hashes(environment, *, train_csv=None, dev_csv=None):
    if not isinstance(environment, dict):
        raise ValueError("environment.json must contain an object")
    checks = (
        ("train_manifest_sha256", train_csv),
        ("dev_manifest_sha256", dev_csv),
    )
    actual = {}
    for key, path in checks:
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {path}")
        expected_hash = environment.get(key)
        if not expected_hash:
            raise ValueError(f"environment.json is missing {key}")
        actual_hash = file_sha256(path)
        actual[key] = actual_hash
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(
                f"manifest SHA-256 mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return actual


def validate_split_counts(train_count, dev_count):
    train_count = int(train_count)
    dev_count = int(dev_count)
    if (train_count, dev_count) != (EXPECTED_TRAIN_SAMPLES, EXPECTED_DEV_SAMPLES):
        raise ValueError(
            "expected train/dev counts 7985/998 for clean_v2, got "
            f"{train_count}/{dev_count}"
        )
    return train_count, dev_count


def validate_replayed_predictions(
    probabilities,
    saved_predictions,
    *,
    image_paths=None,
    true_labels=None,
):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape [samples, classes]")
    if not np.isfinite(probabilities).all():
        raise ValueError("replayed probabilities contain nonfinite values")
    if np.any(probabilities < -1e-8):
        raise ValueError("replayed probabilities contain negative values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("replayed probability rows do not sum to one")
    if len(saved_predictions) != len(probabilities):
        raise ValueError("replayed and saved prediction row count differs")
    if "pred_label" not in saved_predictions.columns:
        raise ValueError("saved predictions are missing pred_label")

    replayed = probabilities.argmax(axis=1)
    saved_labels = saved_predictions["pred_label"].to_numpy(dtype=np.int64)
    mismatch = np.flatnonzero(replayed != saved_labels)
    if mismatch.size:
        raise ValueError(
            "replayed dev argmax mismatch with saved predictions: "
            f"{mismatch.size} rows; first row={int(mismatch[0])}"
        )

    if image_paths is not None:
        if "image_path" not in saved_predictions.columns:
            raise ValueError("saved predictions are missing image_path")
        current_paths = np.asarray([str(path) for path in image_paths], dtype=object)
        saved_paths = saved_predictions["image_path"].astype(str).to_numpy()
        if not np.array_equal(current_paths, saved_paths):
            raise ValueError("replayed dev image path/order mismatch")
    if true_labels is not None:
        if "true_label" not in saved_predictions.columns:
            raise ValueError("saved predictions are missing true_label")
        current_labels = np.asarray(true_labels, dtype=np.int64)
        saved_true = saved_predictions["true_label"].to_numpy(dtype=np.int64)
        if not np.array_equal(current_labels, saved_true):
            raise ValueError("replayed dev true-label/order mismatch")
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen M0 feature separability using clean_v2 train/dev only. "
            "No new deep model is trained."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="repeat exactly for seeds 42, 7, and 2026",
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing audit output directory",
    )
    return parser


def prepare_output_directory(out_dir, overwrite=False, protected_inputs=()):
    out_dir = Path(out_dir)
    resolved = out_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.cwd().resolve():
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    for protected in protected_inputs:
        protected = Path(protected).resolve()
        try:
            protected.relative_to(resolved)
        except ValueError:
            continue
        raise ValueError(
            f"output directory contains protected input {protected}: {resolved}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory is nonempty; use --overwrite to replace it: {out_dir}"
        )
    if existing and overwrite:
        owner_path = out_dir / AUDIT_OWNER_FILENAME
        if not owner_path.is_file():
            raise ValueError(
                f"refusing --overwrite without audit ownership marker: {owner_path}"
            )
        owner_payload = _read_json(owner_path)
        if owner_payload.get("schema") != AUDIT_OWNER_SCHEMA:
            raise ValueError(f"invalid audit ownership marker: {owner_path}")
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    _write_json(
        out_dir / AUDIT_OWNER_FILENAME,
        {
            "schema": AUDIT_OWNER_SCHEMA,
            "purpose": "safe overwrite ownership for frozen M0 boundary audit",
        },
    )
    return out_dir


def write_completed_marker(out_dir, seeds, **extra):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(seed) for seed in seeds],
        "data_scope": "clean_v2 train/dev only",
        "calibration_used": False,
        "test_used": False,
        "expert_review_used": False,
        "new_deep_model_trained": False,
        **extra,
    }
    path = out_dir / "completed.json"
    _write_json(path, payload)
    return path


def _load_checkpoint_bundle(checkpoint_path, expected_seed, train_csv, dev_csv, torch):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != "best_model.pth":
        raise ValueError(f"checkpoint must be named best_model.pth: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    run_dir = checkpoint_path.parent
    expected_run_name = f"m0_caformer_letterbox_clean_v2_seed{expected_seed}"
    if run_dir.name != expected_run_name:
        raise ValueError(
            f"checkpoint directory must be the promoted M0 run {expected_run_name}: "
            f"{run_dir}"
        )
    required = {
        "args": run_dir / "args.json",
        "environment": run_dir / "environment.json",
        "dev_predictions": run_dir / "dev_predictions.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "checkpoint sidecar files are missing: " + ", ".join(missing)
        )
    run_args = _read_json(required["args"])
    environment = _read_json(required["environment"])
    validate_manifest_hashes(environment, train_csv=train_csv, dev_csv=dev_csv)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "safe weights-only checkpoint load failed; refusing unsafe pickle fallback: "
            f"{checkpoint_path}"
        ) from exc
    validate_checkpoint_contract(checkpoint, run_args, expected_seed)
    predictions = pd.read_csv(required["dev_predictions"])
    return {
        "checkpoint": checkpoint,
        "args": run_args,
        "environment": environment,
        "dev_predictions": predictions,
        "checkpoint_path": checkpoint_path,
        "run_dir": run_dir,
    }


def _make_loader(dataset, batch_size, num_workers, device, torch):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _extract_split(model, loader, device, amp, torch):
    model.eval()
    features = []
    probabilities = []
    labels = []
    image_paths = []
    maturity_labels = []
    maturity_names = []
    amp_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
            features.append(output["features"].float().cpu().numpy())
            probabilities.append(output["diagnosis_probs"].float().cpu().numpy())
            labels.append(batch["diagnosis_label"].numpy())
            image_paths.extend(str(path) for path in batch["image_path"])
            maturity_labels.extend(batch["maturity_label"].numpy().tolist())
            maturity_names.extend(str(name) for name in batch["maturity_name"])

    result = {
        "features": np.concatenate(features).astype(np.float32, copy=False),
        "probabilities": np.concatenate(probabilities).astype(np.float64, copy=False),
        "labels": np.concatenate(labels).astype(np.int64, copy=False),
        "image_paths": image_paths,
        "maturity_labels": np.asarray(maturity_labels, dtype=np.int64),
        "maturity_names": maturity_names,
    }
    sample_count = len(result["labels"])
    if result["features"].shape[0] != sample_count:
        raise ValueError("feature row count differs from label count")
    if result["probabilities"].shape != (sample_count, 5):
        raise ValueError("diagnosis probabilities must have shape [samples, 5]")
    if not np.isfinite(result["features"]).all():
        raise ValueError("extracted features contain nonfinite values")
    if not np.isfinite(result["probabilities"]).all():
        raise ValueError("extracted probabilities contain nonfinite values")
    if not np.allclose(
        result["probabilities"].sum(axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("extracted probability rows do not sum to one")
    if len(set(result["image_paths"])) != sample_count:
        raise ValueError("extracted split contains duplicate image paths")
    if set(result["labels"].tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("extracted split must contain all five diagnosis classes")
    return result


def _index_frame(extraction):
    return pd.DataFrame(
        {
            "row_index": np.arange(len(extraction["labels"]), dtype=np.int64),
            "image_path": extraction["image_paths"],
            "true_label": extraction["labels"],
            "true_name": [DIAGNOSIS_NAMES[label] for label in extraction["labels"]],
            "maturity_label": extraction["maturity_labels"],
            "maturity_name": extraction["maturity_names"],
        }
    )


def _validate_split_consistency(reference, current, split_name):
    if not np.array_equal(reference["labels"], current["labels"]):
        raise ValueError(f"cross-seed {split_name} label/order mismatch")
    if reference["image_paths"] != current["image_paths"]:
        raise ValueError(f"cross-seed {split_name} image path/order mismatch")
    if not np.array_equal(reference["maturity_labels"], current["maturity_labels"]):
        raise ValueError(f"cross-seed {split_name} maturity/order mismatch")


def _pair_head_scores(probabilities, pair_labels):
    pair_probabilities = probabilities[:, pair_labels]
    pair_mass = pair_probabilities.sum(axis=1)
    if np.any(pair_mass <= 0.0):
        raise ValueError("pair probability mass must be positive")
    return pair_probabilities[:, 1] / pair_mass


def _analyse_seed(seed, train, dev):
    metric_rows = []
    sample_frames = {}
    for pair_name, spec in PAIR_SPECS.items():
        train_x, train_y, train_indices = select_pair(
            train["features"], train["labels"], spec["labels"]
        )
        dev_x, dev_y, dev_indices = select_pair(
            dev["features"], dev["labels"], spec["labels"]
        )
        pair_probabilities = dev["probabilities"][dev_indices]
        m0_scores = _pair_head_scores(pair_probabilities, spec["labels"])
        m0_predictions = (m0_scores >= 0.5).astype(np.int64)
        m0_metrics = binary_metrics(dev_y, m0_scores, m0_predictions)
        linear = fit_linear_probe(train_x, train_y, dev_x, dev_y)
        knn = fit_knn_probe(train_x, train_y, dev_x, dev_y, n_neighbors=5)
        geometry = geometry_metrics(dev_x, dev_y, n_neighbors=5)

        row = {
            "seed": int(seed),
            "pair_name": pair_name,
            "class_0": spec["names"][0],
            "class_1": spec["names"][1],
            "n_train": int(len(train_indices)),
            "n_dev": int(len(dev_indices)),
            **{f"m0_{key}": value for key, value in m0_metrics.items()},
            **{f"linear_{key}": value for key, value in linear["metrics"].items()},
            **{f"knn_{key}": value for key, value in knn["metrics"].items()},
            **geometry["metrics"],
        }
        for metric in ("macro_f1", "balanced_accuracy", "roc_auc"):
            row[f"linear_minus_m0_{metric}"] = (
                linear["metrics"][metric] - m0_metrics[metric]
            )
            row[f"knn_minus_m0_{metric}"] = (
                knn["metrics"][metric] - m0_metrics[metric]
            )
        metric_rows.append(row)

        sample_frame = build_seed_sample_frame(
            seed=seed,
            image_paths=dev["image_paths"],
            true_labels=dev["labels"],
            probabilities=dev["probabilities"],
            pair_name=pair_name,
            probe_predictions=linear["predictions"],
            knn_predictions=knn["predictions"],
            local_purity=geometry["local_purity"],
        )
        sample_frame["probe_score"] = linear["scores"]
        sample_frame["knn_score"] = knn["scores"]
        sample_frame["five_class_prediction"] = dev["probabilities"][
            dev_indices
        ].argmax(axis=1)
        sample_frame["five_class_confidence"] = dev["probabilities"][
            dev_indices
        ].max(axis=1)
        full_probabilities = np.clip(
            dev["probabilities"][dev_indices], 1e-12, 1.0
        )
        sample_frame["five_class_entropy"] = -np.sum(
            full_probabilities * np.log(full_probabilities), axis=1
        )
        sample_frame["maturity_label"] = dev["maturity_labels"][dev_indices]
        sample_frame["maturity_name"] = np.asarray(
            dev["maturity_names"], dtype=object
        )[dev_indices]
        sample_frames[pair_name] = sample_frame
    return metric_rows, sample_frames


def _build_sample_audit(sample_frames_by_pair):
    outputs = []
    for pair_name, seed_frames in sample_frames_by_pair.items():
        aggregate = aggregate_sample_frames(seed_frames).reset_index()
        reference = seed_frames[0][
            ["image_path", "maturity_label", "maturity_name"]
        ].copy()
        aggregate = aggregate.merge(reference, on="image_path", validate="one_to_one")
        aggregate["true_name"] = [
            DIAGNOSIS_NAMES[label] for label in aggregate["true_label"]
        ]
        for frame in sorted(seed_frames, key=lambda item: int(item["seed"].iloc[0])):
            seed = int(frame["seed"].iloc[0])
            columns = [
                "image_path",
                "m0_pair_score",
                "m0_pair_prediction",
                "m0_pair_margin",
                "m0_correct",
                "probe_score",
                "probe_prediction",
                "probe_correct",
                "knn_score",
                "knn_prediction",
                "knn_correct",
                "local_purity",
                "five_class_prediction",
                "five_class_confidence",
                "five_class_entropy",
            ]
            renamed = frame[columns].rename(
                columns={column: f"seed{seed}_{column}" for column in columns[1:]}
            )
            aggregate = aggregate.merge(renamed, on="image_path", validate="one_to_one")
        outputs.append(aggregate)
    result = pd.concat(outputs, ignore_index=True)
    first_columns = [
        "image_path",
        "true_label",
        "true_name",
        "maturity_label",
        "maturity_name",
        "pair_name",
    ]
    remaining = [column for column in result.columns if column not in first_columns]
    return result[first_columns + remaining]


def _summarize_pair_metrics(pair_metrics):
    numeric = [
        column
        for column in pair_metrics.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    summary = pair_metrics.groupby("pair_name")[numeric].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary.insert(0, "seed_count", pair_metrics.groupby("pair_name")["seed"].nunique())
    return summary.reset_index()


def _diagnostic_summary(pair_metrics, sample_audit):
    summary = {
        "scope": "clean_v2 train/dev only",
        "seeds": list(LOCKED_SEEDS),
        "analysis_seed": ANALYSIS_SEED,
        "decision_rules": {
            "head_gain_threshold": HEAD_GAIN_THRESHOLD,
            "head_positive_seed_minimum": 2,
            "representation_no_gain_required_for_every_seed": True,
            "representation_probe_f1_max": LOW_PROBE_F1_MAX,
            "representation_probe_f1_spread_max": STABLE_PROBE_F1_SPREAD_MAX,
        },
        "calibration_used": False,
        "test_used": False,
        "expert_review_used": False,
        "interpretation_warning": (
            "Automated diagnostic signals are not proof of label error, an absolute "
            "information ceiling, or patient/WSI-level generalization."
        ),
        "pairs": {},
    }
    for pair_name, rows in pair_metrics.groupby("pair_name"):
        pair_samples = sample_audit[sample_audit["pair_name"] == pair_name]
        linear_gains = rows["linear_minus_m0_macro_f1"]
        knn_gains = rows["knn_minus_m0_macro_f1"]
        positive_linear_seeds = int((linear_gains > 0.0).sum())
        head_underuse = bool(
            linear_gains.mean() >= HEAD_GAIN_THRESHOLD
            and positive_linear_seeds >= 2
        )
        linear_f1 = rows["linear_macro_f1"]
        knn_f1 = rows["knn_macro_f1"]
        representation_overlap = bool(
            not head_underuse
            and (linear_gains < HEAD_GAIN_THRESHOLD).all()
            and (knn_gains < HEAD_GAIN_THRESHOLD).all()
            and linear_f1.max() < LOW_PROBE_F1_MAX
            and knn_f1.max() < LOW_PROBE_F1_MAX
            and linear_f1.max() - linear_f1.min() <= STABLE_PROBE_F1_SPREAD_MAX
            and knn_f1.max() - knn_f1.min() <= STABLE_PROBE_F1_SPREAD_MAX
        )
        if head_underuse:
            signal = "possible_head_or_objective_underuse"
        elif representation_overlap:
            signal = "possible_frozen_representation_overlap"
        else:
            signal = "mixed_or_inconclusive"
        unstable_fraction = float(pair_samples["m0_seed_unstable"].mean())
        summary["pairs"][pair_name] = {
            "signal": signal,
            "mean_m0_macro_f1": float(rows["m0_macro_f1"].mean()),
            "mean_linear_macro_f1": float(rows["linear_macro_f1"].mean()),
            "mean_knn_macro_f1": float(rows["knn_macro_f1"].mean()),
            "mean_linear_minus_m0_macro_f1": float(linear_gains.mean()),
            "mean_knn_minus_m0_macro_f1": float(knn_gains.mean()),
            "positive_linear_gain_seeds": positive_linear_seeds,
            "m0_seed_unstable_fraction": unstable_fraction,
            "persistent_m0_pair_errors": int(
                pair_samples["m0_persistent_error"].sum()
            ),
            "sample_count": int(len(pair_samples)),
        }
    return summary


def _write_report(path, diagnostic):
    lines = [
        "# M0 Frozen Boundary Audit Report",
        "",
        "本报告仅使用 `clean_v2` train/dev 与三个冻结 M0 checkpoint；未使用 "
        "calibration/test，未进行专家复核，也未训练新的深度模型。",
        "",
        "## 自动诊断信号",
        "",
    ]
    display_names = {
        "low_grade": "ASC-US / LSIL",
        "high_grade": "ASC-H / HSIL",
    }
    for pair_name, payload in diagnostic["pairs"].items():
        lines.extend(
            [
                f"### {display_names[pair_name]}",
                "",
                f"- 信号分类：`{payload['signal']}`",
                f"- M0 pairwise Macro F1（三 seed 均值）：{payload['mean_m0_macro_f1']:.4f}",
                f"- 线性 probe Macro F1（三 seed 均值）：{payload['mean_linear_macro_f1']:.4f}",
                f"- 5-NN Macro F1（三 seed 均值）：{payload['mean_knn_macro_f1']:.4f}",
                f"- 线性 probe 相对 M0 增益：{payload['mean_linear_minus_m0_macro_f1']:+.4f}",
                f"- M0 跨 seed 不稳定样本比例：{payload['m0_seed_unstable_fraction']:.2%}",
                f"- 三 seed 均错误样本数：{payload['persistent_m0_pair_errors']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "这些结果只能区分‘冻结特征中存在但原诊断头未充分利用的信号’与"
            "‘冻结表征仍明显重叠的信号’。它们不能自动证明标签错误、图像信息达到"
            "绝对上限，也不能支持患者级或 WSI 级泛化结论。",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _runtime_environment(torch, train_csv, dev_csv, device):
    try:
        import sklearn
        import timm
        import torchvision
    except ImportError as exc:
        raise RuntimeError("scikit-learn, timm, and torchvision are required") from exc
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "train_manifest_sha256": file_sha256(train_csv),
        "dev_manifest_sha256": file_sha256(dev_csv),
        "analysis_seed": ANALYSIS_SEED,
    }
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["torch_cuda"] = torch.version.cuda
    return payload


def _validate_feature_arrays(feature_dir, expected_train_count, expected_dev_count):
    feature_dir = Path(feature_dir)
    feature_dim = None
    for seed in LOCKED_SEEDS:
        for split, expected_count in (
            ("train", expected_train_count),
            ("dev", expected_dev_count),
        ):
            path = feature_dir / f"seed{seed}_{split}_features.npy"
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise RuntimeError(f"cannot reopen feature array: {path}") from exc
            if array.ndim != 2 or array.shape[0] != expected_count:
                raise RuntimeError(
                    f"feature array shape mismatch: {path} has {array.shape}"
                )
            if array.dtype != np.float32:
                raise RuntimeError(
                    f"feature array dtype mismatch: {path} has {array.dtype}"
                )
            if not np.isfinite(array).all():
                raise RuntimeError(f"feature array contains nonfinite values: {path}")
            if feature_dim is None:
                feature_dim = int(array.shape[1])
            elif array.shape[1] != feature_dim:
                raise RuntimeError(f"feature dimension mismatch: {path}")
    return feature_dim


def _verify_artifacts(out_dir, expected_train_count, expected_dev_count):
    required = [
        "args.json",
        "environment.json",
        "checkpoint_manifest.csv",
        "features/train_index.csv",
        "features/dev_index.csv",
        "pair_metrics.csv",
        "pair_metrics_summary.csv",
        "sample_audit.csv",
        "persistent_pair_errors.csv",
        "probe_persistent_errors.csv",
        "seed_unstable_samples.csv",
        "head_probe_rescues.csv",
        "diagnostic_summary.json",
        "REPORT.md",
    ]
    required.extend(
        f"features/seed{seed}_{split}_features.npy"
        for seed in LOCKED_SEEDS
        for split in ("train", "dev")
    )
    for relative in required:
        path = out_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required artifact missing or empty: {path}")
    if len(pd.read_csv(out_dir / "features/train_index.csv")) != expected_train_count:
        raise RuntimeError("saved train index row count mismatch")
    if len(pd.read_csv(out_dir / "features/dev_index.csv")) != expected_dev_count:
        raise RuntimeError("saved dev index row count mismatch")
    _validate_feature_arrays(
        out_dir / "features",
        expected_train_count,
        expected_dev_count,
    )
    return required


def run_audit(args):
    try:
        import torch
        from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
        from experiments.tbs.models import build_stage1_model
    except ImportError as exc:
        raise RuntimeError(
            "server runtime requires PyTorch, torchvision, and timm"
        ) from exc

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    checkpoint_paths = parse_checkpoint_specs(args.checkpoint)
    protected_inputs = [args.train_csv, args.dev_csv, *checkpoint_paths.values()]
    out_dir = prepare_output_directory(
        args.out_dir,
        overwrite=args.overwrite,
        protected_inputs=protected_inputs,
    )
    feature_dir = out_dir / "features"
    feature_dir.mkdir()

    random.seed(ANALYSIS_SEED)
    np.random.seed(ANALYSIS_SEED)
    torch.manual_seed(ANALYSIS_SEED)
    torch.cuda.manual_seed_all(ANALYSIS_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    resolved_args = {
        "checkpoints": {str(seed): str(path) for seed, path in checkpoint_paths.items()},
        "train_csv": str(args.train_csv),
        "dev_csv": str(args.dev_csv),
        "out_dir": str(args.out_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "amp": bool(args.amp),
        "overwrite": bool(args.overwrite),
        "analysis_seed": ANALYSIS_SEED,
        "data_scope": "clean_v2 train/dev only",
        "calibration_used": False,
        "test_used": False,
        "expert_review_used": False,
    }
    _write_json(out_dir / "args.json", resolved_args)
    environment = _runtime_environment(torch, args.train_csv, args.dev_csv, device)
    _write_json(out_dir / "environment.json", environment)

    _, eval_transform = build_transforms(
        LOCKED_IMAGE_SIZE, input_mode=LOCKED_INPUT_MODE
    )
    train_dataset = XUDataTBS5Dataset(args.train_csv, transform=eval_transform)
    dev_dataset = XUDataTBS5Dataset(args.dev_csv, transform=eval_transform)
    validate_split_counts(len(train_dataset), len(dev_dataset))
    train_paths = {str(record["image_path"]) for record in train_dataset.records}
    dev_paths = {str(record["image_path"]) for record in dev_dataset.records}
    if len(train_paths) != len(train_dataset) or len(dev_paths) != len(dev_dataset):
        raise ValueError("clean_v2 train/dev manifests contain duplicate image paths")
    overlap = train_paths.intersection(dev_paths)
    if overlap:
        raise ValueError(
            "clean_v2 train/dev manifests overlap in image_path: "
            f"{len(overlap)} rows"
        )
    train_loader = _make_loader(
        train_dataset, args.batch_size, args.num_workers, device, torch
    )
    dev_loader = _make_loader(
        dev_dataset, args.batch_size, args.num_workers, device, torch
    )

    checkpoint_rows = []
    metric_rows = []
    sample_frames_by_pair = {name: [] for name in PAIR_SPECS}
    reference_train = None
    reference_dev = None

    for seed in LOCKED_SEEDS:
        bundle = _load_checkpoint_bundle(
            checkpoint_paths[seed], seed, args.train_csv, args.dev_csv, torch
        )
        model = build_stage1_model(
            LOCKED_VARIANT, model_name=LOCKED_MODEL_NAME, pretrained=False
        )
        model.load_state_dict(bundle["checkpoint"]["model_state"], strict=True)
        model.requires_grad_(False)
        model.to(device)

        print(f"[seed {seed}] extracting deterministic train features", flush=True)
        train = _extract_split(model, train_loader, device, args.amp, torch)
        print(f"[seed {seed}] extracting deterministic dev features", flush=True)
        dev = _extract_split(model, dev_loader, device, args.amp, torch)
        validate_replayed_predictions(
            dev["probabilities"],
            bundle["dev_predictions"],
            image_paths=dev["image_paths"],
            true_labels=dev["labels"],
        )

        if reference_train is None:
            reference_train = train
            reference_dev = dev
            _index_frame(train).to_csv(feature_dir / "train_index.csv", index=False)
            _index_frame(dev).to_csv(feature_dir / "dev_index.csv", index=False)
        else:
            _validate_split_consistency(reference_train, train, "train")
            _validate_split_consistency(reference_dev, dev, "dev")

        np.save(
            feature_dir / f"seed{seed}_train_features.npy",
            train["features"],
            allow_pickle=False,
        )
        np.save(
            feature_dir / f"seed{seed}_dev_features.npy",
            dev["features"],
            allow_pickle=False,
        )
        seed_metrics, seed_frames = _analyse_seed(seed, train, dev)
        metric_rows.extend(seed_metrics)
        for pair_name, frame in seed_frames.items():
            sample_frames_by_pair[pair_name].append(frame)

        checkpoint_rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(bundle["checkpoint_path"]),
                "checkpoint_sha256": file_sha256(bundle["checkpoint_path"]),
                "args_sha256": file_sha256(bundle["run_dir"] / "args.json"),
                "environment_sha256": file_sha256(
                    bundle["run_dir"] / "environment.json"
                ),
                "dev_predictions_sha256": file_sha256(
                    bundle["run_dir"] / "dev_predictions.csv"
                ),
                "checkpoint_epoch": int(bundle["checkpoint"].get("epoch", -1)),
                "replayed_dev_argmax_exact": True,
                "variant": LOCKED_VARIANT,
                "model_name": LOCKED_MODEL_NAME,
                "input_mode": LOCKED_INPUT_MODE,
            }
        )
        del model, bundle, train, dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_manifest = pd.DataFrame(checkpoint_rows)
    checkpoint_manifest.to_csv(out_dir / "checkpoint_manifest.csv", index=False)
    pair_metrics = pd.DataFrame(metric_rows)
    pair_metrics.to_csv(out_dir / "pair_metrics.csv", index=False)
    _summarize_pair_metrics(pair_metrics).to_csv(
        out_dir / "pair_metrics_summary.csv", index=False
    )

    sample_audit = _build_sample_audit(sample_frames_by_pair)
    sample_audit.to_csv(out_dir / "sample_audit.csv", index=False)
    sample_audit[sample_audit["m0_persistent_error"]].to_csv(
        out_dir / "persistent_pair_errors.csv", index=False
    )
    sample_audit[sample_audit["probe_correct_seeds"] == 0].to_csv(
        out_dir / "probe_persistent_errors.csv", index=False
    )
    sample_audit[sample_audit["m0_seed_unstable"]].to_csv(
        out_dir / "seed_unstable_samples.csv", index=False
    )
    sample_audit[
        (sample_audit["m0_correct_seeds"] <= 1)
        & (sample_audit["probe_correct_seeds"] >= 2)
    ].to_csv(out_dir / "head_probe_rescues.csv", index=False)

    diagnostic = _diagnostic_summary(pair_metrics, sample_audit)
    _write_json(out_dir / "diagnostic_summary.json", diagnostic)
    _write_report(out_dir / "REPORT.md", diagnostic)

    required_artifacts = _verify_artifacts(
        out_dir,
        expected_train_count=len(reference_train["labels"]),
        expected_dev_count=len(reference_dev["labels"]),
    )
    artifact_hashes = {
        relative: file_sha256(out_dir / relative) for relative in required_artifacts
    }
    marker = write_completed_marker(
        out_dir,
        seeds=LOCKED_SEEDS,
        train_samples=int(len(reference_train["labels"])),
        dev_samples=int(len(reference_dev["labels"])),
        artifact_sha256=artifact_hashes,
    )
    print(f"Audit completed: {marker}", flush=True)
    return marker


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_audit(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
