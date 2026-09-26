"""Train MERA-Dx on XUData only, then evaluate it on CRIC without updates.

The two stages are intentionally separate.  ``train`` never opens CRIC, and
``evaluate`` loads a frozen source checkpoint and never creates an optimizer or
executes a backward pass.
"""

from __future__ import annotations

import argparse
import hashlib
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
SOURCE_MODEL_NAME = "swin_tiny_patch4_window7_224"
CHECKPOINT_SCHEMA = "xudata-to-cric-zeroshot-checkpoint-v1"
TARGET_SCHEMA = "xudata-to-cric-zeroshot-evaluation-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_class_order(class_names) -> bool:
    observed = tuple(str(value) for value in class_names)
    if observed != CLASS_NAMES:
        raise ValueError(
            f"class order must be {list(CLASS_NAMES)}, got {list(observed)}"
        )
    return True


def make_checkpoint_payload(model_state, source_pool, epochs, seed, pretrained):
    """Build provenance for a source-only checkpoint before serialization."""
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_name": "mera_dx",
        "backbone_name": SOURCE_MODEL_NAME,
        "model_state": model_state,
        "source_pool": str(source_pool),
        "source_pool_sha256": sha256_file(Path(source_pool))
        if Path(source_pool).is_file()
        else None,
        "epochs": int(epochs),
        "seed": int(seed),
        "pretrained": bool(pretrained),
        "input_mode": "letterbox",
        "img_size": 224,
        "target_data_accessed": False,
        "target_labels_accessed": False,
        "selection_source": "fixed_epoch_budget",
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compute_zero_shot_metrics(y_true, y_prob):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 2 or y_prob.shape != (len(y_true), 5):
        raise ValueError("expected y_true=[n] and y_prob=[n,5]")
    if len(y_true) == 0 or not np.isfinite(y_prob).all():
        raise ValueError("zero-shot evaluation inputs must be non-empty and finite")
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(5))
    recalls, specificities = [], []
    recall_values, specificity_values = [], []
    for index in range(5):
        tp = float(cm[index, index])
        fn = float(cm[index, :].sum() - tp)
        fp = float(cm[:, index].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        recalls.append(_finite_or_none(recall))
        specificities.append(_finite_or_none(specificity))
        recall_values.append(recall)
        specificity_values.append(specificity)
    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
            labels=np.arange(5),
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = np.nan
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=np.arange(5), average="macro", zero_division=0)
        ),
        "macro_sensitivity": _finite_or_none(np.nanmean(recall_values)),
        "macro_specificity": _finite_or_none(np.nanmean(specificity_values)),
        "macro_auc": _finite_or_none(auc),
        "per_class_recall": recalls,
        "per_class_specificity": specificities,
        "confusion_matrix": cm.tolist(),
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


def _source_loader(csv_path, transform, batch_size, shuffle, num_workers):
    import torch
    from torch.utils.data import DataLoader

    from experiments.tbs.dataset import XUDataTBS5Dataset

    dataset = XUDataTBS5Dataset(csv_path, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def _train_source_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.train_cric_fiveclass import _loss_for_output

    model.train()
    total_loss, total_count = 0.0, 0
    amp_enabled = bool(scaler.is_enabled())
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["diagnosis_label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(images)
            # ``objective`` was added after the original source-only trainer;
            # the full loss is the default in both interfaces.
            loss, _ = _loss_for_output("mera_dx", output, labels)
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
        raise RuntimeError("source training loader is empty")
    return total_loss / total_count


def train_source(args):
    import torch

    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.c2_dualprototype_training import initialize_prototypes
    from experiments.tbs.tbs_fv_factorized_model import build_tbs_dualprototype_model

    fold_dir = Path(args.fold_dir)
    pool_csv = fold_dir / "pool.csv"
    if not pool_csv.is_file():
        raise FileNotFoundError(pool_csv)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed)
    train_transform, eval_transform = build_transforms(
        args.img_size, input_mode="letterbox"
    )
    train_loader = _source_loader(
        pool_csv, train_transform, args.batch_size, True, args.num_workers
    )
    prototype_loader = _source_loader(
        pool_csv, eval_transform, args.batch_size, False, args.num_workers
    )
    model = build_tbs_dualprototype_model(
        model_name=SOURCE_MODEL_NAME,
        pretrained=args.pretrained,
        semantic_dim=128,
        residual_logit_bound=0.10,
        prototype_temperature=10.0,
        prototype_residual_bound=0.10,
    ).to(device)
    initialize_prototypes(model, prototype_loader, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = _make_scaler(enabled=device.type == "cuda" and args.amp)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_source_epoch(
            model, train_loader, optimizer, scaler, device
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "target_data_accessed": False,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            out_dir / "source_train_history.csv", index=False, lineterminator="\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        scheduler.step()

    checkpoint_payload = make_checkpoint_payload(
        model.state_dict(), pool_csv, args.epochs, args.seed, args.pretrained
    )
    checkpoint_payload.update(
        {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "model_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    torch.save(checkpoint_payload, out_dir / "final_model.pth")
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "source_pool": str(pool_csv.resolve()),
        "source_pool_sha256": sha256_file(pool_csv),
        "checkpoint": str((out_dir / "final_model.pth").resolve()),
        "checkpoint_sha256": sha256_file(out_dir / "final_model.pth"),
        "model": "MERA-Dx",
        "backbone": SOURCE_MODEL_NAME,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "pretrained": bool(args.pretrained),
        "input_mode": "letterbox",
        "img_size": int(args.img_size),
        "target_data_accessed": False,
        "target_labels_accessed": False,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    (out_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def _write_predictions(path, image_ids, cell_keys, image_paths, y_true, y_prob):
    frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "cell_key": cell_keys,
            "image_path": image_paths,
            "true_label": y_true,
            "pred_label": y_prob.argmax(axis=1),
        }
    )
    for index in range(5):
        frame[f"prob_{index}"] = y_prob[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def evaluate_target(args):
    import torch
    from torch.utils.data import DataLoader

    from experiments import train_cric_fiveclass as cric
    from experiments.tbs.dataset import build_transforms
    from experiments.tbs.tbs_fv_factorized_model import build_tbs_dualprototype_model

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.cric_data_dir)
    out_dir = Path(args.out_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    metadata_path = data_dir / "preparation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    prep = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_class_order(prep.get("class_names", []))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a source-only zero-shot checkpoint")
    model = build_tbs_dualprototype_model(
        model_name=SOURCE_MODEL_NAME,
        pretrained=False,
        semantic_dim=128,
        residual_logit_bound=0.10,
        prototype_temperature=10.0,
        prototype_residual_bound=0.10,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    _, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    fold_metrics = []
    all_true, all_prob = [], []
    seen_cell_keys = set()
    fold_count = int(prep.get("n_splits", args.n_splits))
    with torch.inference_mode():
        for fold in range(fold_count):
            fold_csv = data_dir / f"fold_{fold}.csv"
            if not fold_csv.is_file():
                raise FileNotFoundError(fold_csv)
            frame = pd.read_csv(fold_csv)
            if "split" not in frame.columns:
                raise ValueError(f"CRIC fold manifest lacks split column: {fold_csv}")
            val_frame = frame.loc[frame["split"].astype(str) == "val"].copy()
            if val_frame.empty:
                raise ValueError(f"CRIC validation split is empty: {fold_csv}")
            val_frame = cric.resolve_manifest_paths(val_frame, data_dir)
            cell_keys = set(val_frame["cell_key"].astype(str))
            overlap = seen_cell_keys & cell_keys
            if overlap:
                raise ValueError(f"CRIC validation cells overlap across folds: {len(overlap)}")
            seen_cell_keys.update(cell_keys)
            dataset = cric.CRICCellDataset(val_frame, eval_transform)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=bool(args.num_workers),
            )
            labels, probs, image_ids, keys, paths = [], [], [], [], []
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                output = model(images)
                labels.append(batch["label"].numpy())
                probs.append(output["diagnosis_probs"].float().cpu().numpy())
                image_ids.extend(batch["image_id"].numpy().tolist())
                keys.extend(batch["cell_key"])
                paths.extend(batch["image_path"])
            y_true = np.concatenate(labels)
            y_prob = np.concatenate(probs)
            metrics = compute_zero_shot_metrics(y_true, y_prob)
            fold_dir = out_dir / f"fold_{fold}"
            _write_predictions(fold_dir / "predictions.csv", image_ids, keys, paths, y_true, y_prob)
            (fold_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            fold_metrics.append({"fold": fold, **{key: metrics[key] for key in (
                "accuracy", "macro_f1", "macro_sensitivity", "macro_specificity", "macro_auc"
            )}})
            all_true.append(y_true)
            all_prob.append(y_prob)

    pooled = compute_zero_shot_metrics(np.concatenate(all_true), np.concatenate(all_prob))
    folds = pd.DataFrame(fold_metrics)
    summary = {
        "model": "mera_dx",
        "paper_label": "MERA-Dx (proposed)",
        "category": "Proposed",
        "fold_count": int(fold_count),
    }
    for metric in ("accuracy", "macro_f1", "macro_sensitivity", "macro_specificity", "macro_auc"):
        values = folds[metric].astype(float).to_numpy()
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"pooled_{metric}"] = pooled[metric]
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    folds.to_csv(out_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    (out_dir / "pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target_metadata = {
        "schema_version": TARGET_SCHEMA,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_pool": checkpoint.get("source_pool"),
        "source_pool_sha256": checkpoint.get("source_pool_sha256"),
        "target_data_dir": str(data_dir.resolve()),
        "target_preparation_metadata_sha256": sha256_file(metadata_path),
        "class_names": list(CLASS_NAMES),
        "fold_count": int(fold_count),
        "target_data_accessed": True,
        "target_training_performed": False,
        "target_parameter_updates": False,
        "target_threshold_tuning": False,
        "target_early_stopping": False,
        "source_pretrained": bool(checkpoint.get("pretrained", False)),
    }
    (out_dir / "zeroshot_metadata.json").write_text(
        json.dumps(target_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "pooled": pooled}, indent=2, ensure_ascii=False))
    return target_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train", help="train MERA-Dx on XUData only")
    train.add_argument("--fold_dir", type=Path, required=True)
    train.add_argument("--out_dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--img_size", type=int, default=224)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--num_workers", type=int, default=8)
    train.add_argument("--amp", dest="amp", action="store_true")
    train.add_argument("--no-amp", dest="amp", action="store_false")
    train.set_defaults(amp=True)
    train.add_argument("--pretrained", dest="pretrained", action="store_true")
    train.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    train.set_defaults(pretrained=False)

    evaluate = subparsers.add_parser("evaluate", help="evaluate frozen checkpoint on CRIC")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cric_data_dir", type=Path, required=True)
    evaluate.add_argument("--out_dir", type=Path, required=True)
    evaluate.add_argument("--img_size", type=int, default=224)
    evaluate.add_argument("--batch_size", type=int, default=64)
    evaluate.add_argument("--n_splits", type=int, default=5)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "train":
        train_source(args)
    else:
        evaluate_target(args)


if __name__ == "__main__":
    main()
