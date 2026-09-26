"""Train leakage-resistant single-view S0/S1 xudata TBS models."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.singleview_protocol import make_stagewise_train_splits
from experiments.xudata_gain_common import (
    compute_locked_candidate_metrics,
    reject_forbidden_data_path,
    validate_train_dev_paths,
    write_safety_artifacts,
)


STAGES = ("s0", "s1")
SPLIT_FILENAMES = ("train_fit.csv", "select_s0.csv", "select_s1.csv")
LOSS_NAMES = (
    "loss",
    "diagnosis_loss",
    "screen_loss",
    "morph_loss",
    "evidence_loss",
    "decorr_loss",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_selection_role(stage, role):
    if stage not in STAGES:
        raise ValueError(f"unknown single-view stage: {stage}")
    expected = f"select_{stage}"
    if role != expected:
        raise ValueError(f"{stage} requires selection role {expected}, got {role!r}")


def validate_checkpoint_stage(target_stage, payload):
    if target_stage not in STAGES:
        raise ValueError(f"unknown single-view stage: {target_stage}")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a metadata dictionary")
    schema = payload.get("schema_version")
    if schema == "xudata-tbs-s0r-final-checkpoint-v1":
        if target_stage != "s1":
            raise ValueError("S0-R checkpoint schema is accepted only for s1")
        if payload.get("stage") != "s0r":
            raise ValueError("S0-R checkpoint stage must be s0r")
        if payload.get("selection_role") != "fivefold_fixed_epoch":
            raise ValueError("S0-R checkpoint selection_role must be fivefold_fixed_epoch")
        return
    expected_schema = "xudata-tbs-singleview-checkpoint-v1"
    if schema != expected_schema:
        raise ValueError(
            f"checkpoint schema must be {expected_schema!r}; "
            f"got {schema!r}"
        )
    source_stage = payload.get("stage")
    expected_source = "s0"
    if source_stage != expected_source:
        raise ValueError(
            f"{target_stage} requires an {expected_source} checkpoint; "
            f"checkpoint stage is {source_stage!r}"
        )
    if payload.get("selection_role") != "select_s0":
        raise ValueError("S0 checkpoint selection_role must be select_s0")


def _nonempty_values(frame, column):
    if column not in frame.columns:
        return set()
    values = frame[column].dropna().astype(str).str.strip()
    return set(values[values != ""])


def validate_manifest_independence(train_csv, dev_csv):
    train = pd.read_csv(train_csv)
    dev = pd.read_csv(dev_csv)
    for name, frame in (("train", train), ("dev", dev)):
        if "image_path" not in frame.columns:
            raise ValueError(f"{name} CSV is missing image_path")
        paths = frame["image_path"].astype(str)
        if paths.duplicated().any():
            raise ValueError(f"{name} CSV contains duplicate image_path values")
        content = frame.get("content_sha256")
        if content is not None:
            nonempty = content.dropna().astype(str).str.strip()
            nonempty = nonempty[nonempty != ""]
            if nonempty.duplicated().any():
                raise ValueError(f"{name} CSV contains duplicate content_sha256 values")
    path_overlap = _nonempty_values(train, "image_path") & _nonempty_values(
        dev, "image_path"
    )
    content_overlap = _nonempty_values(
        train, "content_sha256"
    ) & _nonempty_values(dev, "content_sha256")
    if path_overlap or content_overlap:
        raise ValueError(
            "train/dev identity overlap detected: "
            f"image_path={len(path_overlap)}, content_sha256={len(content_overlap)}"
        )


def _validate_split_integrity(train_csv, paths):
    source = pd.read_csv(train_csv)
    required = {"image_path", "diagnosis_label"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"train CSV is missing columns: {missing}")
    if source["image_path"].astype(str).duplicated().any():
        raise ValueError("train CSV contains duplicate image_path values")
    source_content = source.get("content_sha256")
    if source_content is not None:
        nonempty = source_content.dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty != ""]
        if nonempty.duplicated().any():
            raise ValueError("train CSV contains duplicate content_sha256 values")

    partitions = [pd.read_csv(path) for path in paths]
    path_sets = [set(frame["image_path"].astype(str)) for frame in partitions]
    if any(path_sets[left] & path_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("stagewise split partitions overlap")
    source_paths = set(source["image_path"].astype(str))
    partition_paths = set().union(*path_sets)
    if source_paths != partition_paths or sum(map(len, path_sets)) != len(source):
        raise ValueError("stagewise split does not exactly partition train CSV")
    for path, frame in zip(paths, partitions):
        if frame.empty:
            raise ValueError(f"stagewise split is empty: {path}")
        if not set(frame["diagnosis_label"].astype(int)) == set(range(5)):
            raise ValueError(f"stagewise split lacks one or more classes: {path}")


def resolve_stagewise_splits(train_csv, split_dir, select_fraction=0.05, seed=42):
    split_dir = Path(split_dir)
    reject_forbidden_data_path(split_dir)
    source = pd.read_csv(train_csv)
    source_content = source.get("content_sha256")
    if source["image_path"].astype(str).duplicated().any():
        raise ValueError("train CSV contains duplicate image_path values")
    if source_content is not None:
        nonempty = source_content.dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty != ""]
        if nonempty.duplicated().any():
            raise ValueError("train CSV contains duplicate content_sha256 values")
    paths = tuple(split_dir / name for name in SPLIT_FILENAMES)
    present = [path.is_file() for path in paths]
    if any(present) and not all(present):
        raise FileExistsError(
            f"incomplete stagewise split directory; refusing repair/overwrite: {split_dir}"
        )
    if not any(present):
        if split_dir.exists() and any(split_dir.iterdir()):
            raise FileExistsError(
                f"non-empty split directory has no complete owned split: {split_dir}"
            )
        paths = tuple(
            make_stagewise_train_splits(
                train_csv,
                split_dir,
                select_fraction=select_fraction,
                seed=seed,
            )
        )
        metadata = {
            "schema_version": "xudata-tbs-singleview-stagewise-split-v1",
            "source_train_csv": str(Path(train_csv).resolve()),
            "source_train_sha256": _sha256(train_csv),
            "select_fraction_per_stage": float(select_fraction),
            "seed": int(seed),
            "partitions": {path.name: _sha256(path) for path in paths},
        }
        (split_dir / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    metadata_path = split_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"stagewise split metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_train_sha256") != _sha256(train_csv):
        raise ValueError("stagewise split was generated from a different train CSV")
    if int(metadata.get("seed", -1)) != int(seed):
        raise ValueError("stagewise split seed does not match the requested seed")
    recorded_fraction = float(metadata.get("select_fraction_per_stage", -1.0))
    if not np.isclose(recorded_fraction, float(select_fraction), rtol=0.0, atol=1e-12):
        raise ValueError(
            "stagewise split select_fraction does not match the requested value"
        )
    for path in paths:
        expected = metadata.get("partitions", {}).get(path.name)
        if expected != _sha256(path):
            raise ValueError(f"stagewise split checksum mismatch: {path}")
    _validate_split_integrity(train_csv, paths)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--s0r_gate_json", type=Path)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--select_fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lambda_screen", type=float, default=0.2)
    parser.add_argument("--lambda_morph", type=float, default=0.3)
    parser.add_argument("--lambda_evidence", type=float, default=0.3)
    parser.add_argument("--lambda_decorr", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        validate_train_dev_paths(args.train_csv, args.dev_csv)
        validate_manifest_independence(args.train_csv, args.dev_csv)
        for path in (args.split_dir, args.out_dir):
            reject_forbidden_data_path(path)
        if args.out_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {args.out_dir}"
            )
        if args.stage == "s1" and args.init_checkpoint is None:
            raise ValueError("s1 requires --init_checkpoint from S0")
        if args.init_checkpoint is not None:
            reject_forbidden_data_path(args.init_checkpoint)
            if not args.init_checkpoint.is_file():
                raise FileNotFoundError(args.init_checkpoint)
        if args.s0r_gate_json is not None:
            reject_forbidden_data_path(args.s0r_gate_json)
            if not args.s0r_gate_json.is_file():
                raise FileNotFoundError(args.s0r_gate_json)
        if min(args.epochs, args.batch_size, args.patience, args.img_size) <= 0:
            raise ValueError("epochs, batch_size, patience, and img_size must be positive")
        if args.semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        if not 0.0 < args.select_fraction < 0.25:
            raise ValueError("select_fraction must be between 0 and 0.25")
        if not np.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not 0.0 < args.backbone_lr_multiplier <= 1.0:
            raise ValueError("backbone_lr_multiplier must be within (0, 1]")
        for name in ("lambda_screen", "lambda_morph", "lambda_evidence", "lambda_decorr"):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset, batch_size, num_workers, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _targets(batch, device):
    return {
        "diagnosis_labels": batch["diagnosis_label"].to(device, non_blocking=True),
        "screen_labels": batch["screen_label"].to(device, non_blocking=True),
        "morph_labels": batch["morph_label"].to(device, non_blocking=True),
        "evidence_labels": batch["evidence_label"].to(device, non_blocking=True),
        "semantic_mask": batch["semantic_mask"].to(device, non_blocking=True).bool(),
    }


def _loss_kwargs(args):
    return {
        "stage": args.stage,
        "lambda_screen": args.lambda_screen,
        "lambda_morph": args.lambda_morph,
        "lambda_evidence": args.lambda_evidence,
        "lambda_decorr": args.lambda_decorr,
    }


def _build_optimizer(model, args):
    import torch

    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    task = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if not backbone or not task:
        raise RuntimeError("optimizer requires backbone and task parameters")
    return torch.optim.AdamW(
        [
            {"name": "backbone", "params": backbone, "lr": args.lr * args.backbone_lr_multiplier},
            {"name": "task", "params": task, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


def _load_checkpoint(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_initial_weights(model, args, device, train_fit_sha256):
    if args.init_checkpoint is None:
        return None
    payload = _load_checkpoint(args.init_checkpoint, device)
    validate_checkpoint_stage(args.stage, payload)
    if payload.get("schema_version") == "xudata-tbs-s0r-final-checkpoint-v1":
        if args.s0r_gate_json is None:
            raise ValueError("S1 from S0-R requires --s0r_gate_json")
        gate = json.loads(args.s0r_gate_json.read_text(encoding="utf-8"))
        if gate.get("stage") != "s0" or not gate.get("passed"):
            raise ValueError("S0-R gate does not authorize S1")
        candidate_predictions = args.init_checkpoint.parent / "dev_predictions.csv"
        if not candidate_predictions.is_file():
            raise FileNotFoundError(candidate_predictions)
        if gate.get("candidate_predictions_sha256") != _sha256(candidate_predictions):
            raise ValueError("S0-R gate candidate prediction checksum mismatch")
        if payload.get("select_s1_sha256") != _sha256(args.split_dir / "select_s1.csv"):
            raise ValueError("S0-R checkpoint sealed select_s1 checksum mismatch")
        if payload.get("train_fit_sha256") != _sha256(args.split_dir / "train_fit.csv"):
            raise ValueError("S0-R checkpoint train_fit checksum mismatch")
        if payload.get("select_s0_sha256") != _sha256(args.split_dir / "select_s0.csv"):
            raise ValueError("S0-R checkpoint select_s0 checksum mismatch")
        if payload.get("model_name") != args.model_name:
            raise ValueError("checkpoint model_name does not match the requested model")
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain model_state")
        prefix = "backbone."
        backbone_state = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        if not backbone_state:
            raise ValueError("S0-R checkpoint contains no backbone parameters")
        model.backbone.load_state_dict(backbone_state, strict=True)
        return payload
    source_fit_sha = payload.get("train_fit_sha256")
    if source_fit_sha != train_fit_sha256:
        raise ValueError("checkpoint train_fit does not match the current stagewise split")
    if payload.get("model_name") != args.model_name:
        raise ValueError("checkpoint model_name does not match the requested model")
    expected_select_sha = _sha256(args.split_dir / "select_s0.csv")
    if payload.get("select_manifest_sha256") != expected_select_sha:
        raise ValueError("checkpoint select_s0 manifest does not match the current split")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state")
    if args.stage == "s0":
        model.load_state_dict(state, strict=True)
    else:
        prefix = "backbone."
        backbone_state = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        if not backbone_state:
            raise ValueError("S0 checkpoint contains no backbone parameters")
        model.backbone.load_state_dict(backbone_state, strict=True)
    return payload


def _train_epoch(model, loader, optimizer, scaler, device, args):
    import torch
    from experiments.tbs.singleview_model import singleview_tbs_loss

    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            losses = singleview_tbs_loss(model(images), targets, **_loss_kwargs(args))
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        size = len(images)
        count += size
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach()) * size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _evaluate(model, loader, device, args):
    import torch
    from experiments.tbs.metrics import compute_semantic_metrics
    from experiments.tbs.singleview_model import singleview_tbs_loss

    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    labels, probabilities = [], []
    paths, maturity_labels, maturity_names = [], [], []
    semantic_masks, morph_true, morph_prob = [], [], []
    evidence_true, evidence_prob = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = _targets(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
                losses = singleview_tbs_loss(output, targets, **_loss_kwargs(args))
            size = len(images)
            count += size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * size
            labels.append(targets["diagnosis_labels"].cpu().numpy())
            probabilities.append(output["diagnosis_probs"].cpu().numpy())
            paths.extend(batch["image_path"])
            maturity_labels.append(batch["maturity_label"].cpu().numpy())
            maturity_names.extend(batch["maturity_name"])
            semantic_masks.append(targets["semantic_mask"].cpu().numpy())
            if args.stage == "s1":
                morph_true.append(targets["morph_labels"].cpu().numpy())
                morph_prob.append(output["morph_probs"][:, 1].cpu().numpy())
                evidence_true.append(targets["evidence_labels"].cpu().numpy())
                evidence_prob.append(output["evidence_probs"][:, 1].cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    metrics = compute_locked_candidate_metrics(y_true, y_prob)
    metrics["loss"] = totals["loss"] / max(count, 1)
    audit = {
        "maturity_label": np.concatenate(maturity_labels),
        "maturity_name": maturity_names,
        "semantic_mask": np.concatenate(semantic_masks).astype(bool),
    }
    if args.stage == "s1":
        audit.update(
            {
                "morph_true": np.concatenate(morph_true),
                "morph_prob_high": np.concatenate(morph_prob),
                "evidence_true": np.concatenate(evidence_true),
                "evidence_prob_definitive": np.concatenate(evidence_prob),
            }
        )
        metrics.update(
            compute_semantic_metrics(
                audit["morph_true"],
                audit["morph_prob_high"],
                audit["evidence_true"],
                audit["evidence_prob_definitive"],
                audit["semantic_mask"],
                y_prob,
            )
        )
    return metrics, y_true, y_prob, paths, audit


def _write_dev_artifacts(out_dir, metrics, y_true, y_prob, image_paths, audit, stage):
    from sklearn.metrics import classification_report, confusion_matrix
    from experiments.tbs.labels import DIAGNOSIS_NAMES

    y_pred = y_prob.argmax(axis=1)
    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true,
            "true_name": [DIAGNOSIS_NAMES[index] for index in y_true],
            "pred_label": y_pred,
            "pred_name": [DIAGNOSIS_NAMES[index] for index in y_pred],
            "maturity_label": audit["maturity_label"],
            "maturity_name": audit["maturity_name"],
            "semantic_mask": audit["semantic_mask"].astype(int),
            "screen_prob": y_prob[:, 1:].sum(axis=1),
        }
    )
    for index, name in enumerate(DIAGNOSIS_NAMES):
        predictions[f"prob_{name}"] = y_prob[:, index]
    if stage == "s1":
        for key in (
            "morph_true",
            "morph_prob_high",
            "evidence_true",
            "evidence_prob_definitive",
        ):
            predictions[key] = audit[key]
    predictions.to_csv(out_dir / "dev_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(5))),
        index=DIAGNOSIS_NAMES,
        columns=DIAGNOSIS_NAMES,
    ).to_csv(out_dir / "dev_confusion_matrix.csv")
    pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(5)),
            target_names=DIAGNOSIS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(out_dir / "dev_classification_report.csv")
    (out_dir / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_training(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.singleview_model import build_tbs_singleview_model

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU smoke test")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    fit_csv, select_s0_csv, select_s1_csv = resolve_stagewise_splits(
        args.train_csv,
        args.split_dir,
        select_fraction=args.select_fraction,
        seed=args.seed,
    )
    selection_role = f"select_{args.stage}"
    validate_selection_role(args.stage, selection_role)
    select_csv = select_s0_csv if args.stage == "s0" else select_s1_csv
    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    fit_loader = _loader(
        XUDataTBS5Dataset(fit_csv, train_transform),
        args.batch_size,
        args.num_workers,
        True,
        args.seed,
    )
    select_loader = _loader(
        XUDataTBS5Dataset(select_csv, eval_transform),
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
    )
    model = build_tbs_singleview_model(
        args.stage,
        model_name=args.model_name,
        pretrained=args.pretrained and args.init_checkpoint is None,
        semantic_dim=args.semantic_dim,
    ).to(device)
    fit_sha = _sha256(fit_csv)
    _load_initial_weights(model, args, device, fit_sha)
    optimizer = _build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    best_score = -1.0
    stale_epochs = 0
    best_path = args.out_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(model, fit_loader, optimizer, scaler, device, args)
        select_metrics, _, _, _, _ = _evaluate(model, select_loader, device, args)
        row = {
            "epoch": epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "task_lr": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"select_{key}": value for key, value in select_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / "metrics.csv", index=False, lineterminator="\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = float(select_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": "xudata-tbs-singleview-checkpoint-v1",
                    "stage": args.stage,
                    "epoch": epoch,
                    "model_name": args.model_name,
                    "semantic_dim": args.semantic_dim,
                    "train_fit_sha256": fit_sha,
                    "selection_role": selection_role,
                    "select_manifest_sha256": _sha256(select_csv),
                    "model_state": model.state_dict(),
                    "select_metrics": select_metrics,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            break

    payload = _load_checkpoint(best_path, device)
    if payload.get("stage") != args.stage or payload.get("selection_role") != selection_role:
        raise RuntimeError("best checkpoint stage/selection metadata is inconsistent")
    model.load_state_dict(payload["model_state"], strict=True)

    # Dev is instantiated and evaluated only after model selection is frozen.
    dev_loader = _loader(
        XUDataTBS5Dataset(args.dev_csv, eval_transform),
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
    )
    dev_metrics, y_true, y_prob, image_paths, audit = _evaluate(
        model, dev_loader, device, args
    )
    _write_dev_artifacts(
        args.out_dir, dev_metrics, y_true, y_prob, image_paths, audit, args.stage
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-singleview-s0-s1-v1",
            "stage": args.stage,
            "route": "TRAIN_FIT_STAGE_SELECT_DEV_ONCE",
            "train_split": "train_fit",
            "selection_split": selection_role,
            "dev_evaluation_count": 1,
            "checkpoint_selection_metric": "macro_f1",
            "s1_checkpoint_transfer": "backbone_only" if args.stage == "s1" else None,
            "model_trained": True,
            "external_data_used": False,
            "labels_are_ontology_derived": True,
        },
    )
    return dev_metrics


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_training(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
