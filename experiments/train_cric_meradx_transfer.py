"""Two-stage MERA-Dx transfer training for the CRIC five-class benchmark.

Stage ``pretrain`` trains the paper-facing MERA-Dx model on XUData only by
delegating to the source-only trainer.  Stage ``finetune`` loads that frozen
source checkpoint and fine-tunes a fresh copy on each CRIC training fold.
CRIC validation folds are used only for model selection and final reporting;
the script never reads a separate CRIC test set or uses target labels outside
the current training/validation manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CLASS_NAMES = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
SOURCE_CHECKPOINT_SCHEMA = "xudata-to-cric-zeroshot-checkpoint-v1"
TRANSFER_CHECKPOINT_SCHEMA = "xudata-to-cric-supervised-transfer-checkpoint-v1"
TRANSFER_RUN_SCHEMA = "xudata-to-cric-supervised-transfer-v1"
SOURCE_MODEL_NAME = "swin_tiny_patch4_window7_224"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_checkpoint(payload) -> bool:
    """Reject checkpoints that are not the expected XUData MERA-Dx source."""
    if not isinstance(payload, dict):
        raise ValueError("source checkpoint must be a dictionary payload")
    if payload.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "source checkpoint must be produced by the XUData source-only trainer"
        )
    if payload.get("model_name") != "mera_dx":
        raise ValueError("source checkpoint model_name must be mera_dx")
    if payload.get("backbone_name") != SOURCE_MODEL_NAME:
        raise ValueError(
            f"source checkpoint backbone_name must be {SOURCE_MODEL_NAME}"
        )
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError("source checkpoint lacks a model_state dictionary")
    return True


def inverse_sqrt_sample_weights(labels):
    """Return one inverse-square-root class-frequency weight per sample."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("labels must be a non-empty one-dimensional non-negative array")
    classes, counts = np.unique(values, return_counts=True)
    class_weights = {int(cls): 1.0 / np.sqrt(float(count)) for cls, count in zip(classes, counts)}
    weights = np.asarray([class_weights[int(value)] for value in values], dtype=np.float64)
    return weights / weights.mean()


def transfer_metadata_flags():
    """Fixed provenance flags used by the supervised-transfer report."""
    return {
        "target_training_performed": True,
        "target_parameter_updates": True,
        "target_validation_used_for_selection": True,
        "target_test_accessed_during_training": False,
        "target_test_labels_used_for_selection": False,
    }


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_scaler(enabled: bool):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_loader(frame, transform, batch_size, num_workers, weighted=False):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from experiments import train_cric_fiveclass as cric

    dataset = cric.CRICCellDataset(frame, transform)
    sampler = None
    shuffle = False
    if weighted:
        weights = inverse_sqrt_sample_weights(frame["label"].astype(int).to_numpy())
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
    else:
        shuffle = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def _set_backbone_trainable(model, trainable: bool):
    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            parameter.requires_grad = bool(trainable)


def call_meradx_loss(loss_fn, output, labels, objective="full"):
    """Call both legacy (3-argument) and current loss-function interfaces."""
    if objective == "direct_ce":
        # Direct CE is defined locally so it remains available even when the
        # server still has the legacy CRIC trainer without an ``objective``
        # keyword in ``_loss_for_output``.
        if not isinstance(output, dict) or "diagnosis_log_probs" not in output:
            raise ValueError("MERA-Dx output lacks diagnosis_log_probs for direct_ce")
        import torch.nn.functional as F

        diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], labels)
        return diagnosis_loss, {"diagnosis_loss": float(diagnosis_loss.detach())}
    parameters = inspect.signature(loss_fn).parameters
    if "objective" in parameters:
        return loss_fn("mera_dx", output, labels, objective=objective)
    if objective != "full":
        raise TypeError(
            "the installed CRIC trainer has no objective argument; "
            "only objective=full is supported with this legacy trainer"
        )
    return loss_fn("mera_dx", output, labels)


def make_parameter_groups(model, lr: float, weight_decay: float, backbone_lr_multiplier: float):
    """Use a smaller learning rate for the transferred visual backbone."""
    if lr <= 0.0 or weight_decay < 0.0 or backbone_lr_multiplier <= 0.0:
        raise ValueError("learning rates and weight decay must be valid positive values")
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            backbone.append(parameter)
        else:
            heads.append(parameter)
    groups = []
    if heads:
        groups.append({"params": heads, "lr": float(lr), "weight_decay": float(weight_decay)})
    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": float(lr) * float(backbone_lr_multiplier),
                "weight_decay": float(weight_decay),
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def _load_source_model(checkpoint_path: Path, device):
    import torch

    from experiments import train_cric_fiveclass as cric

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_source_checkpoint(checkpoint)
    model = cric.build_model("mera_dx", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def _train_epoch(model, loader, optimizer, scaler, device, objective):
    import torch

    from experiments import train_cric_fiveclass as cric

    model.train()
    total_loss, total_count = 0.0, 0
    amp_enabled = bool(scaler.is_enabled())
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(images)
            loss, _ = call_meradx_loss(
                cric._loss_for_output, output, labels, objective=objective
            )
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_count += len(labels)
    if total_count == 0:
        raise RuntimeError("CRIC training loader is empty")
    return total_loss / total_count


def _parse_folds(value: str, n_splits: int):
    folds = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not folds or any(fold < 0 or fold >= n_splits for fold in folds):
        raise ValueError(f"folds must be a comma-separated subset of 0..{n_splits - 1}")
    return folds


def pretrain_source(args):
    """Delegate source pretraining to the tested source-only implementation."""
    from experiments.xudata_to_cric_zeroshot import train_source

    return train_source(args)


def finetune_cric(args):
    import torch

    from experiments import train_cric_fiveclass as cric
    from experiments.tbs.dataset import build_transforms

    data_dir = Path(args.data_dir)
    source_checkpoint_path = Path(args.init_checkpoint)
    out_dir = Path(args.out_dir)
    metadata_path = data_dir / "preparation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(source_checkpoint_path)
    prep = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(prep.get("class_names", [])) != CLASS_NAMES:
        raise ValueError("CRIC preparation does not use the expected five-class mapping")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source_checkpoint_sha256 = sha256_file(source_checkpoint_path)
    folds = _parse_folds(args.folds, args.n_splits)
    _, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    train_transform, _ = build_transforms(args.img_size, input_mode="letterbox")
    fold_rows, all_true, all_prob = [], [], []
    registry = []

    for fold in folds:
        fold_seed = int(args.seed) + int(fold)
        seed_everything(fold_seed)
        fold_csv = data_dir / f"fold_{fold}.csv"
        if not fold_csv.is_file():
            raise FileNotFoundError(fold_csv)
        frame = cric.resolve_manifest_paths(pd.read_csv(fold_csv), data_dir)
        if "split" not in frame.columns:
            raise ValueError(f"CRIC fold manifest lacks split column: {fold_csv}")
        train_frame = frame.loc[frame["split"].astype(str) == "train"].copy()
        val_frame = frame.loc[frame["split"].astype(str) == "val"].copy()
        if train_frame.empty or val_frame.empty:
            raise ValueError(f"CRIC fold {fold} must have non-empty train and val splits")
        train_ids = set(train_frame["image_id"].astype(int))
        val_ids = set(val_frame["image_id"].astype(int))
        if train_ids & val_ids:
            raise ValueError(f"slide leakage detected in CRIC fold {fold}")
        train_loader = _make_loader(
            train_frame,
            train_transform,
            args.batch_size,
            args.num_workers,
            weighted=args.sampler == "inverse_sqrt",
        )
        eval_train_loader = _make_loader(
            train_frame, eval_transform, args.batch_size, args.num_workers, weighted=False
        )
        val_loader = _make_loader(
            val_frame, eval_transform, args.batch_size, args.num_workers, weighted=False
        )
        model, source_payload = _load_source_model(source_checkpoint_path, device)
        cric._initialize_meradx_prototypes(model, eval_train_loader, device)
        _set_backbone_trainable(model, args.freeze_backbone_epochs == 0)
        optimizer = torch.optim.AdamW(
            make_parameter_groups(
                model,
                args.lr,
                args.weight_decay,
                args.backbone_lr_multiplier,
            )
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1)
        )
        scaler = _make_scaler(enabled=device.type == "cuda" and args.amp)
        model_dir = out_dir / "mera_dx"
        model_dir.mkdir(parents=True, exist_ok=True)
        fold_dir = model_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_path = model_dir / f"fold_{fold}_best.pt"
        best_score = -float("inf")
        bad_epochs = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_backbone_epochs + 1:
                _set_backbone_trainable(model, True)
            train_loss = _train_epoch(
                model, train_loader, optimizer, scaler, device, args.objective
            )
            val_metrics, *_ = cric._evaluate("mera_dx", model, val_loader, device)
            scheduler.step()
            row = {
                "epoch": int(epoch),
                "fold": int(fold),
                "train_loss": float(train_loss),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "backbone_frozen": bool(epoch <= args.freeze_backbone_epochs),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            score = float(val_metrics["macro_f1"])
            if score > best_score:
                best_score = score
                bad_epochs = 0
                torch.save(
                    {
                        "schema_version": TRANSFER_CHECKPOINT_SCHEMA,
                        "model_state": model.state_dict(),
                        "model_name": "mera_dx",
                        "fold": int(fold),
                        "best_epoch": int(epoch),
                        "source_checkpoint": str(source_checkpoint_path.resolve()),
                        "source_checkpoint_sha256": source_checkpoint_sha256,
                        "target_data_dir": str(data_dir.resolve()),
                    },
                    best_path,
                )
            else:
                bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                break

        if not best_path.is_file():
            raise RuntimeError(f"no best checkpoint written for CRIC fold {fold}")
        best_payload = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_payload["model_state"], strict=True)
        metrics, y_true, y_prob, image_ids, cell_keys, image_paths = cric._evaluate(
            "mera_dx", model, val_loader, device
        )
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False, lineterminator="\n")
        cric._write_predictions(
            fold_dir / "predictions.csv",
            y_true,
            y_prob,
            image_ids,
            cell_keys,
            image_paths,
        )
        (fold_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        fold_rows.append(
            {
                "model": "mera_dx",
                "paper_label": "MERA-Dx (proposed)",
                "category": "Proposed",
                "fold": int(fold),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
        all_true.append(y_true)
        all_prob.append(y_prob)
        registry.append(
            {
                "fold": int(fold),
                "best_epoch": int(best_payload["best_epoch"]),
                "train_samples": int(len(train_frame)),
                "validation_samples": int(len(val_frame)),
                "source_checkpoint_sha256": source_checkpoint_sha256,
            }
        )
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(out_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    summary = {
        "model": "mera_dx",
        "paper_label": "MERA-Dx (proposed)",
        "category": "Proposed",
        "fold_count": int(len(folds_frame)),
    }
    for metric in (
        "accuracy",
        "macro_f1",
        "macro_sensitivity",
        "macro_specificity",
        "macro_auc",
    ):
        values = folds_frame[metric].astype(float).to_numpy()
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    pooled = cric.compute_metrics(np.concatenate(all_true), np.concatenate(all_prob))
    (out_dir / "pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    (out_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "schema_version": TRANSFER_RUN_SCHEMA,
        "model": "MERA-Dx",
        "backbone": SOURCE_MODEL_NAME,
        "source_checkpoint": str(source_checkpoint_path.resolve()),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_checkpoint_schema": source_payload.get("schema_version"),
        "target_data_dir": str(data_dir.resolve()),
        "target_preparation_metadata_sha256": sha256_file(metadata_path),
        "class_names": list(CLASS_NAMES),
        "excluded_label": "SCC",
        "folds_run": [int(fold) for fold in folds],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "backbone_lr_multiplier": float(args.backbone_lr_multiplier),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
        "patience": int(args.patience),
        "sampler": args.sampler,
        "objective": args.objective,
        "seed": int(args.seed),
        "img_size": int(args.img_size),
        "input_mode": "letterbox",
        "amp": bool(args.amp),
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        **transfer_metadata_flags(),
    }
    (out_dir / "transfer_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "pooled": pooled}, indent=2, ensure_ascii=False))
    return summary


def _add_amp_flags(parser):
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pretrain = subparsers.add_parser("pretrain", help="train MERA-Dx on XUData only")
    pretrain.add_argument("--fold_dir", type=Path, required=True)
    pretrain.add_argument("--out_dir", type=Path, required=True)
    pretrain.add_argument("--epochs", type=int, default=30)
    pretrain.add_argument("--batch_size", type=int, default=64)
    pretrain.add_argument("--lr", type=float, default=1e-4)
    pretrain.add_argument("--weight_decay", type=float, default=1e-4)
    pretrain.add_argument("--img_size", type=int, default=224)
    pretrain.add_argument("--seed", type=int, default=42)
    pretrain.add_argument("--device", default="cuda:0")
    pretrain.add_argument("--num_workers", type=int, default=8)
    _add_amp_flags(pretrain)
    pretrain.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrain.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    pretrain.set_defaults(pretrained=False)

    finetune = subparsers.add_parser(
        "finetune", help="fine-tune the source checkpoint on CRIC folds"
    )
    finetune.add_argument("--data_dir", type=Path, required=True)
    finetune.add_argument("--init_checkpoint", type=Path, required=True)
    finetune.add_argument("--out_dir", type=Path, required=True)
    finetune.add_argument("--epochs", type=int, default=30)
    finetune.add_argument("--batch_size", type=int, default=64)
    finetune.add_argument("--lr", type=float, default=3e-5)
    finetune.add_argument("--weight_decay", type=float, default=1e-4)
    finetune.add_argument("--backbone_lr_multiplier", type=float, default=0.2)
    finetune.add_argument("--freeze_backbone_epochs", type=int, default=3)
    finetune.add_argument("--patience", type=int, default=8)
    finetune.add_argument("--img_size", type=int, default=224)
    finetune.add_argument("--n_splits", type=int, default=5)
    finetune.add_argument("--folds", default="0,1,2,3,4")
    finetune.add_argument("--seed", type=int, default=42)
    finetune.add_argument("--device", default="cuda:0")
    finetune.add_argument("--num_workers", type=int, default=8)
    finetune.add_argument("--sampler", choices=("inverse_sqrt", "shuffle"), default="inverse_sqrt")
    finetune.add_argument("--objective", choices=("full", "direct_ce"), default="full")
    _add_amp_flags(finetune)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "pretrain":
        pretrain_source(args)
    else:
        finetune_cric(args)


if __name__ == "__main__":
    main()
