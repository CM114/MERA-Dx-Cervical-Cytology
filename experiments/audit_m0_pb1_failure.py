"""Frozen train/dev failure-mechanism audit for M0 versus failed PB1.

This CLI intentionally has no calibration or test arguments and never updates
model parameters. It routes the next diagnostic action; it cannot promote PB1.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_m0_feature_boundaries import (  # noqa: E402
    _extract_split,
    _index_frame,
    _make_loader,
    _runtime_environment,
    _validate_split_consistency,
    file_sha256,
    validate_manifest_hashes,
    validate_replayed_predictions,
    validate_split_counts,
)
from experiments.tbs.pb1_failure_audit import (  # noqa: E402
    PAIR_SPECS,
    audit_model_split,
    build_maturity_audit,
    build_review_list,
    build_transition_audit,
    compare_metric_tables,
    route_decision,
)


LOCKED_SEED = 42
LOCKED_MODEL_NAME = "caformer_s18"
LOCKED_IMAGE_SIZE = 224
LOCKED_INPUT_MODE = "letterbox"
ANALYSIS_SEED = 0
EXPECTED_TRAIN_SAMPLES = 7985
EXPECTED_DEV_SAMPLES = 998
AUDIT_OWNER_FILENAME = ".m0_pb1_failure_audit_owner.json"
AUDIT_OWNER_SCHEMA = "xudata-tbs5-m0-pb1-failure-audit-v1"

RUN_NAMES = {
    "m0": "m0_caformer_letterbox_clean_v2_seed42",
    "pb1": "m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed42",
}

COMMON_LOCKS = {
    "variant": "m0",
    "model_name": LOCKED_MODEL_NAME,
    "input_mode": LOCKED_INPUT_MODE,
    "img_size": LOCKED_IMAGE_SIZE,
    "seed": LOCKED_SEED,
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-4,
    "backbone_lr_multiplier": 1.0,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
    "lambda_screen": 0.0,
    "pretrained": True,
}


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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-checkpoint", type=Path, required=True)
    parser.add_argument("--pb1-checkpoint", type=Path, required=True)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _matches(actual, expected):
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_checkpoint_contract(checkpoint, run_args, run_kind):
    """Validate the locked scientific identity before model construction."""

    if run_kind not in RUN_NAMES:
        raise ValueError(f"unknown run kind: {run_kind}")
    if not isinstance(checkpoint, dict) or not isinstance(run_args, dict):
        raise ValueError("checkpoint and args.json must contain objects")
    locked = {**COMMON_LOCKS, "experiment_name": RUN_NAMES[run_kind]}
    if run_kind == "pb1":
        locked.update(
            {
                "boundary_loss": "pair_boundary_supcon",
                "temperature": 0.1,
                "lambda_pb": 0.1,
            }
        )
    for source, payload in (("args.json", run_args), ("checkpoint.args", checkpoint.get("args"))):
        if not isinstance(payload, dict):
            raise ValueError(f"{source} must contain an object")
        for key, expected in locked.items():
            default = None
            if run_kind == "m0" and key == "backbone_lr_multiplier":
                default = 1.0
            actual = payload.get(key, default)
            if not _matches(actual, expected):
                raise ValueError(
                    f"{source} {key}={actual!r} does not match locked {run_kind} value {expected!r}"
                )
        if run_kind == "m0":
            boundary_loss = payload.get("boundary_loss", "none")
            lambda_pb = payload.get("lambda_pb", 0.0)
            if boundary_loss != "none" or not _matches(lambda_pb, 0.0):
                raise ValueError("M0 must not enable a pair-boundary loss")
    if checkpoint.get("variant") != "m0":
        raise ValueError("checkpoint variant must be m0")
    if checkpoint.get("model_name") != LOCKED_MODEL_NAME:
        raise ValueError("checkpoint model_name must be caformer_s18")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model_state must be a nonempty mapping")
    if checkpoint.get("args") != run_args:
        raise ValueError("checkpoint.args must exactly match args.json")
    return {
        "run_kind": run_kind,
        "seed": LOCKED_SEED,
        "variant": "m0",
        "model_name": LOCKED_MODEL_NAME,
        "input_mode": LOCKED_INPUT_MODE,
        "img_size": LOCKED_IMAGE_SIZE,
    }


def prepare_output_directory(out_dir, overwrite=False, protected_inputs=()):
    """Create or replace only a directory owned by this exact audit."""

    out_dir = Path(out_dir)
    resolved = out_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.cwd().resolve():
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    for protected in protected_inputs:
        protected = Path(protected).resolve()
        if protected == resolved or resolved in protected.parents:
            raise ValueError(f"output directory contains protected input: {protected}")
    if out_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory is nonempty; use --overwrite only for this audit: {out_dir}"
        )
    if existing:
        marker = out_dir / AUDIT_OWNER_FILENAME
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(f"refusing overwrite without ownership marker: {marker}")
        if _read_json(marker).get("schema") != AUDIT_OWNER_SCHEMA:
            raise ValueError(f"invalid ownership marker: {marker}")
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    _write_json(
        out_dir / AUDIT_OWNER_FILENAME,
        {
            "schema": AUDIT_OWNER_SCHEMA,
            "purpose": "safe overwrite ownership for frozen M0-PB1 failure audit",
        },
    )
    return out_dir


def derive_decision(comparison):
    required = {
        "split",
        "pair_name",
        "delta_mean_local_purity",
        "delta_pair_macro_f1",
    }
    if not isinstance(comparison, pd.DataFrame) or not required.issubset(
        comparison.columns
    ):
        raise ValueError("comparison table is missing decision columns")
    identities = set(zip(comparison["split"], comparison["pair_name"]))
    expected = {(split, pair) for split in ("train", "dev") for pair in PAIR_SPECS}
    if identities != expected or len(comparison) != 4:
        raise ValueError("comparison must contain exactly two pairs for train and dev")
    train = comparison[comparison["split"] == "train"]
    dev = comparison[comparison["split"] == "dev"]
    pair_deltas = {
        row.pair_name: float(row.delta_pair_macro_f1)
        for row in dev.itertuples(index=False)
    }
    return route_decision(
        float(train["delta_mean_local_purity"].mean()),
        float(dev["delta_mean_local_purity"].mean()),
        float(dev["delta_pair_macro_f1"].mean()),
        pair_deltas,
    )


def write_completed_marker(out_dir, artifacts):
    out_dir = Path(out_dir)
    artifact_hashes = {}
    for relative in artifacts:
        path = out_dir / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"required artifact is missing or invalid: {path}")
        artifact_hashes[str(relative)] = file_sha256(path)
    payload = {
        "status": "completed",
        "schema": AUDIT_OWNER_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": LOCKED_SEED,
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "dev_samples": EXPECTED_DEV_SAMPLES,
        "data_scope": "clean_v2 train/dev only",
        "calibration_used": False,
        "test_used": False,
        "new_deep_model_trained": False,
        "pb1_promoted": False,
        "artifact_sha256": artifact_hashes,
    }
    path = out_dir / "completed.json"
    _write_json(path, payload)
    return path


def _load_run(checkpoint_path, run_kind, train_csv, dev_csv, torch):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != "best_model.pth" or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing best_model.pth: {checkpoint_path}")
    run_dir = checkpoint_path.parent
    if run_dir.name != RUN_NAMES[run_kind]:
        raise ValueError(
            f"{run_kind} checkpoint must be under {RUN_NAMES[run_kind]}: {run_dir}"
        )
    sidecars = {
        "args": run_dir / "args.json",
        "environment": run_dir / "environment.json",
        "predictions": run_dir / "dev_predictions.csv",
    }
    missing = [str(path) for path in sidecars.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint sidecars are missing: " + ", ".join(missing))
    run_args = _read_json(sidecars["args"])
    environment = _read_json(sidecars["environment"])
    validate_manifest_hashes(environment, train_csv=train_csv, dev_csv=dev_csv)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"safe weights-only checkpoint load failed: {checkpoint_path}"
        ) from exc
    identity = validate_checkpoint_contract(checkpoint, run_args, run_kind)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "run_dir": run_dir,
        "args": run_args,
        "environment": environment,
        "predictions": pd.read_csv(sidecars["predictions"]),
        "identity": identity,
    }


def _transition_summary(frame, split_name):
    counts = frame["transition"].value_counts()
    rows = [
        {"split": split_name, "scope": "all", "transition": key, "count": int(counts.get(key, 0))}
        for key in ("both_correct", "harmed", "rescued", "persistent_error")
    ]
    for pair_name in PAIR_SPECS:
        pair = frame[frame["pair_name"] == pair_name]
        pair_counts = pair["transition"].value_counts()
        rows.extend(
            {
                "split": split_name,
                "scope": pair_name,
                "transition": key,
                "count": int(pair_counts.get(key, 0)),
            }
            for key in ("both_correct", "harmed", "rescued", "persistent_error")
        )
    rows.append(
        {
            "split": split_name,
            "scope": "Normal",
            "transition": "harmed",
            "count": int(frame["normal_harmed"].sum()),
        }
    )
    return rows


def _write_report(path, decision, comparison, transition_summary):
    lines = [
        "# Frozen M0-PB1 Failure Audit",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "This is a diagnostic routing result. PB1 remains rejected and no new model was trained.",
        "",
        "## Geometry And Classification",
        "",
    ]
    for row in comparison.sort_values(["split", "pair_name"]).itertuples(index=False):
        lines.append(
            f"- {row.split}/{row.pair_name}: local-purity delta "
            f"{row.delta_mean_local_purity:+.6f}; pair Macro F1 delta "
            f"{row.delta_pair_macro_f1:+.6f}."
        )
    lines.extend(["", "## Sample Transitions", ""])
    dev_all = transition_summary[
        (transition_summary["split"] == "dev")
        & (transition_summary["scope"] == "all")
    ]
    for row in dev_all.itertuples(index=False):
        lines.append(f"- dev/{row.transition}: {row.count}")
    lines.extend(
        [
            "",
            "## Data Controls",
            "",
            "- Data: clean_v2 train/dev only.",
            "- Calibration used: false.",
            "- Test used: false.",
            "- New deep model trained: false.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_saved_arrays(feature_dir):
    feature_dim = None
    for model_name in ("m0", "pb1"):
        for split, expected in (("train", EXPECTED_TRAIN_SAMPLES), ("dev", EXPECTED_DEV_SAMPLES)):
            feature_path = feature_dir / f"{model_name}_{split}_features.npy"
            probability_path = feature_dir / f"{model_name}_{split}_probabilities.npy"
            features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
            if features.ndim != 2 or features.shape[0] != expected or features.dtype != np.float32:
                raise RuntimeError(f"saved feature contract failed: {feature_path}")
            if feature_dim is None:
                feature_dim = features.shape[1]
            elif features.shape[1] != feature_dim:
                raise RuntimeError("M0 and PB1 feature dimensions differ")
            if probabilities.shape != (expected, 5) or probabilities.dtype != np.float64:
                raise RuntimeError(f"saved probability contract failed: {probability_path}")


def run_audit(args):
    try:
        import torch
        from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
        from experiments.tbs.models import build_stage1_model
    except ImportError as exc:
        raise RuntimeError("server runtime requires PyTorch, torchvision, and timm") from exc

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    protected = [
        args.m0_checkpoint,
        args.pb1_checkpoint,
        args.train_csv,
        args.dev_csv,
    ]
    out_dir = prepare_output_directory(args.out_dir, args.overwrite, protected)
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

    _write_json(
        out_dir / "args.json",
        {
            "m0_checkpoint": str(args.m0_checkpoint),
            "pb1_checkpoint": str(args.pb1_checkpoint),
            "train_csv": str(args.train_csv),
            "dev_csv": str(args.dev_csv),
            "out_dir": str(args.out_dir),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "amp": bool(args.amp),
            "analysis_seed": ANALYSIS_SEED,
            "data_scope": "clean_v2 train/dev only",
            "calibration_used": False,
            "test_used": False,
            "new_deep_model_trained": False,
        },
    )
    _write_json(
        out_dir / "environment.json",
        _runtime_environment(torch, args.train_csv, args.dev_csv, device),
    )

    _, eval_transform = build_transforms(LOCKED_IMAGE_SIZE, LOCKED_INPUT_MODE)
    train_dataset = XUDataTBS5Dataset(args.train_csv, transform=eval_transform)
    dev_dataset = XUDataTBS5Dataset(args.dev_csv, transform=eval_transform)
    validate_split_counts(len(train_dataset), len(dev_dataset))
    train_paths = {str(record["image_path"]) for record in train_dataset.records}
    dev_paths = {str(record["image_path"]) for record in dev_dataset.records}
    if len(train_paths) != len(train_dataset) or len(dev_paths) != len(dev_dataset):
        raise ValueError("clean_v2 manifests contain duplicate image paths")
    if train_paths.intersection(dev_paths):
        raise ValueError("clean_v2 train/dev image paths overlap")
    train_loader = _make_loader(train_dataset, args.batch_size, args.num_workers, device, torch)
    dev_loader = _make_loader(dev_dataset, args.batch_size, args.num_workers, device, torch)

    checkpoint_paths = {"m0": args.m0_checkpoint, "pb1": args.pb1_checkpoint}
    extractions = {}
    checkpoint_rows = []
    reference = None
    for run_kind in ("m0", "pb1"):
        bundle = _load_run(
            checkpoint_paths[run_kind], run_kind, args.train_csv, args.dev_csv, torch
        )
        model = build_stage1_model("m0", LOCKED_MODEL_NAME, pretrained=False)
        model.load_state_dict(bundle["checkpoint"]["model_state"], strict=True)
        model.requires_grad_(False)
        model.to(device)
        print(f"[{run_kind}] extracting frozen train features", flush=True)
        train = _extract_split(model, train_loader, device, args.amp, torch)
        print(f"[{run_kind}] extracting frozen dev features", flush=True)
        dev = _extract_split(model, dev_loader, device, args.amp, torch)
        validate_replayed_predictions(
            dev["probabilities"],
            bundle["predictions"],
            image_paths=dev["image_paths"],
            true_labels=dev["labels"],
        )
        if reference is None:
            reference = {"train": train, "dev": dev}
            _index_frame(train).to_csv(feature_dir / "train_index.csv", index=False)
            _index_frame(dev).to_csv(feature_dir / "dev_index.csv", index=False)
        else:
            _validate_split_consistency(reference["train"], train, "train")
            _validate_split_consistency(reference["dev"], dev, "dev")
        extractions[run_kind] = {"train": train, "dev": dev}
        for split_name, extraction in (("train", train), ("dev", dev)):
            np.save(
                feature_dir / f"{run_kind}_{split_name}_features.npy",
                extraction["features"].astype(np.float32, copy=False),
                allow_pickle=False,
            )
            np.save(
                feature_dir / f"{run_kind}_{split_name}_probabilities.npy",
                extraction["probabilities"].astype(np.float64, copy=False),
                allow_pickle=False,
            )
        checkpoint_rows.append(
            {
                "run_kind": run_kind,
                "run_name": bundle["run_dir"].name,
                "checkpoint_path": str(bundle["checkpoint_path"]),
                "checkpoint_sha256": file_sha256(bundle["checkpoint_path"]),
                "args_sha256": file_sha256(bundle["run_dir"] / "args.json"),
                "environment_sha256": file_sha256(bundle["run_dir"] / "environment.json"),
                "dev_predictions_sha256": file_sha256(bundle["run_dir"] / "dev_predictions.csv"),
                "checkpoint_epoch": int(bundle["checkpoint"].get("epoch", -1)),
                "replayed_dev_argmax_exact": True,
            }
        )
        del model, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(checkpoint_rows).to_csv(out_dir / "checkpoint_manifest.csv", index=False)
    metric_frames = []
    local_frames = []
    maturity_frames = []
    for run_kind in ("m0", "pb1"):
        for split_name in ("train", "dev"):
            extraction = extractions[run_kind][split_name]
            metrics, local = audit_model_split(
                run_kind,
                split_name,
                extraction["features"],
                extraction["labels"],
                extraction["probabilities"],
                n_neighbors=5,
            )
            metric_frames.append(metrics)
            local_frames.append(local)
            maturity_frames.append(
                build_maturity_audit(
                    run_kind,
                    split_name,
                    extraction["features"],
                    extraction["labels"],
                    extraction["probabilities"],
                    extraction["maturity_labels"],
                    extraction["maturity_names"],
                    n_neighbors=5,
                )
            )
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_local = pd.concat(local_frames, ignore_index=True)
    all_maturity = pd.concat(maturity_frames, ignore_index=True)
    all_metrics.to_csv(out_dir / "model_split_metrics.csv", index=False)
    all_local.to_csv(out_dir / "local_purity_samples.csv", index=False)
    all_maturity.to_csv(out_dir / "maturity_metrics.csv", index=False)
    comparison = compare_metric_tables(
        all_metrics[all_metrics["model_name"] == "m0"],
        all_metrics[all_metrics["model_name"] == "pb1"],
    )
    comparison.to_csv(out_dir / "metric_comparison.csv", index=False)

    transition_frames = {}
    summary_rows = []
    for split_name in ("train", "dev"):
        index = _index_frame(reference[split_name])
        m0_local = all_local[
            (all_local["model_name"] == "m0") & (all_local["split"] == split_name)
        ]
        pb1_local = all_local[
            (all_local["model_name"] == "pb1") & (all_local["split"] == split_name)
        ]
        transitions = build_transition_audit(
            index,
            extractions["m0"][split_name]["probabilities"],
            extractions["pb1"][split_name]["probabilities"],
            m0_local,
            pb1_local,
        )
        transition_frames[split_name] = transitions
        transitions.to_csv(out_dir / f"transition_audit_{split_name}.csv", index=False)
        summary_rows.extend(_transition_summary(transitions, split_name))
    transition_summary = pd.DataFrame(summary_rows)
    transition_summary.to_csv(out_dir / "transition_summary.csv", index=False)
    review = build_review_list(transition_frames["dev"], per_group_limit=20)
    review.to_csv(out_dir / "pathology_review_list.csv", index=False)

    decision = derive_decision(comparison)
    decision.update(
        {
            "data_scope": "clean_v2 train/dev only",
            "calibration_used": False,
            "test_used": False,
            "new_deep_model_trained": False,
            "review_list_samples": int(len(review)),
        }
    )
    _write_json(out_dir / "decision.json", decision)
    _write_report(out_dir / "REPORT.md", decision, comparison, transition_summary)
    _validate_saved_arrays(feature_dir)

    artifacts = sorted(
        str(path.relative_to(out_dir)).replace("\\", "/")
        for path in out_dir.rglob("*")
        if path.is_file()
        and path.name not in {AUDIT_OWNER_FILENAME, "completed.json"}
    )
    marker = write_completed_marker(out_dir, artifacts)
    print(f"Audit completed: {marker}", flush=True)
    print(f"Decision: {decision['decision']}", flush=True)
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
