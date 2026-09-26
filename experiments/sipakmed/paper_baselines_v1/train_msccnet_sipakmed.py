#!/usr/bin/env python3
"""Grouped five-fold MSCCNet paper-guided adaptation on SIPaKMeD."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.sipakmed.paper_baselines_v1.msccnet import CONFIG, JointMSCCLoss, MSCCNet


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "external_data/SIPAKMED/manifests_v2/manifest.csv"
DEFAULT_RESULTS = ROOT / "results/sipakmed_paper_arch_baselines_v1"
EXPECTED_CLASS_NAMES = (
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",
)
IMAGE_SIZE = 224
AUGMENTATION_VARIANTS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_safe_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "class_name", "label", "group_id", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks required columns: {sorted(missing)}")
    if len(manifest) != 4049:
        raise ValueError(f"expected the verified 4049-image SIPaKMeD manifest, found {len(manifest)}")
    class_rows = manifest.sort_values("label").drop_duplicates("label")
    if tuple(class_rows["class_name"].tolist()) != EXPECTED_CLASS_NAMES:
        raise ValueError("SIPaKMeD class-id mapping differs from the verified five-class mapping")
    if set(manifest["label"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly labels 0 through 4")
    if set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly outer folds 0 through 4")
    missing_files = [p for p in manifest["image_path"].tolist() if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest


def make_internal_split(outer_train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_path", "label", "group_id"}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"outer training frame is missing columns: {sorted(missing)}")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, select_indices = next(
        splitter.split(outer_train["image_path"], outer_train["label"], outer_train["group_id"])
    )
    select = outer_train.iloc[select_indices].copy()
    fit = outer_train.drop(outer_train.index[select_indices]).copy()
    for name, frame in (("fit", fit), ("select", select)):
        if frame.empty or set(frame["label"].astype(int)) != set(range(5)):
            raise RuntimeError(f"internal {name} split does not contain all five classes")
    if set(fit["group_id"]) & set(select["group_id"]):
        raise RuntimeError("internal split has source-group leakage")
    return fit.reset_index(drop=True), select.reset_index(drop=True)


class MSCCTrainingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.multiplier = len(AUGMENTATION_VARIANTS) if augment else 1

    def __len__(self) -> int:
        return len(self.frame) * self.multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.augment:
            row_index, variant_index = divmod(index, len(AUGMENTATION_VARIANTS))
            variant = AUGMENTATION_VARIANTS[variant_index]
        else:
            row_index, variant = index, "identity"
        row = self.frame.iloc[row_index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if variant == "rotate_90":
            image = image.rotate(90, expand=True)
        elif variant == "rotate_180":
            image = image.rotate(180, expand=True)
        elif variant == "rotate_270":
            image = image.rotate(270, expand=True)
        elif variant == "flip_horizontal":
            image = ImageOps.mirror(image)
        elif variant == "flip_vertical":
            image = ImageOps.flip(image)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor, int(row.label)


def make_loader(
    frame: pd.DataFrame,
    *,
    augment: bool,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        MSCCTrainingDataset(frame, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[int], list[int]]:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        truths.extend(labels.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(truths, predictions), truths, predictions


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metric: dict,
) -> None:
    torch.save(
        {
            "method": "MSCCNet",
            "epoch": epoch,
            "selection_metrics": metric,
            "model_config": CONFIG,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def train_fold(
    fold: int,
    manifest: pd.DataFrame,
    run_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    fold_root.mkdir(parents=False, exist_ok=False)
    outer_train = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select = make_internal_split(outer_train, fold_seed)
    for name, frame in (("fit", fit), ("select", select), ("heldout", heldout)):
        if frame.empty or set(frame.label.astype(int)) != set(range(5)):
            raise RuntimeError(f"fold {fold} {name} split is empty or missing a class")
    groups = {name: set(frame.group_id) for name, frame in (("fit", fit), ("select", select), ("heldout", heldout))}
    intersections = {
        f"{left}_x_{right}": len(groups[left] & groups[right])
        for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))
    }
    if any(intersections.values()):
        raise RuntimeError(f"fold {fold} contains source-group leakage: {intersections}")
    split_dir = fold_root / "split"
    split_dir.mkdir()
    fit.to_csv(split_dir / "fit.csv", index=False)
    select.to_csv(split_dir / "select.csv", index=False)
    heldout.to_csv(split_dir / "heldout.csv", index=False)
    atomic_json(
        fold_root / "split_audit.json",
        {
            "fold": fold,
            "counts": {name: len(frame) for name, frame in (("fit", fit), ("select", select), ("heldout", heldout))},
            "augmented_fit_samples_per_epoch": len(MSCCTrainingDataset(fit, augment=True)),
            "groups": {name: len(values) for name, values in groups.items()},
            "pairwise_group_intersections": intersections,
            "classes": EXPECTED_CLASS_NAMES,
        },
    )
    fit_loader = make_loader(fit, augment=True, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    heldout_loader = make_loader(heldout, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)

    model = MSCCNet(num_classes=5, routing_iterations=args.routing_iterations).to(device)
    criterion = JointMSCCLoss(margin_weight=args.margin_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total_epochs = min(args.epochs, args.max_epochs) if args.max_epochs else args.epochs
    best_score = (-1.0, -1.0)
    best_record: dict | None = None
    epoch_log = fold_root / "epochs.jsonl"
    status_path = run_root / "status.json"

    for epoch in range(1, total_epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (images, labels) in enumerate(fit_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(images, return_capsules=True)
            loss = criterion(logits, lengths, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fold {fold} epoch {epoch} produced non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
        select_metrics, _, _ = evaluate(model, select_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "fold": fold,
            "epoch": epoch,
            "learning_rate": current_lr,
            "fit_loss": loss_sum / max(seen, 1),
            "fit_samples_seen": seen,
            "select": select_metrics,
            "seconds": time.time() - started,
        }
        with epoch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = (select_metrics["macro_f1"], select_metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_record = record
            save_checkpoint(fold_root / "best.pt", model, optimizer, scheduler, epoch, select_metrics)
        save_checkpoint(fold_root / "latest.pt", model, optimizer, scheduler, epoch, select_metrics)
        scheduler.step()
        atomic_json(
            status_path,
            {
                "state": "training",
                "method": "MSCCNet",
                "fold": fold,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "configured_epochs": args.epochs,
                "latest_select_macro_f1": select_metrics["macro_f1"],
                "latest_select_accuracy": select_metrics["accuracy"],
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"fold={fold} epoch={epoch}/{total_epochs} loss={record['fit_loss']:.4f} "
            f"lr={current_lr:.3g} select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}",
            flush=True,
        )

    if best_record is None:
        raise RuntimeError(f"no checkpoint was selected for fold {fold}")
    checkpoint = torch.load(fold_root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    heldout_metrics, truths, predictions = evaluate(model, heldout_loader, device)
    result = {
        "method": "MSCCNet",
        "dataset": "SIPaKMeD",
        "fold": fold,
        "seed": fold_seed,
        "n_fit": len(fit),
        "n_select": len(select),
        "n_heldout": len(heldout),
        "augmented_fit_samples_per_epoch": len(MSCCTrainingDataset(fit, augment=True)),
        "best_epoch": int(best_record["epoch"]),
        "best_selection": best_record["select"],
        "heldout_metrics": heldout_metrics,
        "heldout_evaluation_count": 1,
        "class_names": EXPECTED_CLASS_NAMES,
        "predictions_recorded": len(truths) == len(predictions) == len(heldout),
    }
    atomic_json(fold_root / "result.json", result)
    print(
        f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} "
        f"macro_f1={heldout_metrics['macro_f1']:.4f}",
        flush=True,
    )
    atomic_json(
        status_path,
        {
            "state": "fold_complete",
            "method": "MSCCNet",
            "fold": fold,
            "result": heldout_metrics,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del model, optimizer, scheduler, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = [{"fold": result["fold"], **result["heldout_metrics"]} for result in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary = {
        "method": "MSCCNet",
        "dataset": "SIPaKMeD",
        "completed_folds": len(results),
        "metrics_mean": {name: float(table[name].mean()) for name in metric_names},
        "metrics_sample_sd": {name: float(table[name].std(ddof=1)) for name in metric_names} if len(table) > 1 else {},
        "folds": rows,
    }
    atomic_json(run_root / "summary.json", summary)
    return summary


def run_smoke(manifest: pd.DataFrame, device: torch.device, args: argparse.Namespace, smoke_root: Path) -> None:
    row_ids = [int(manifest.index[manifest.label == label][0]) for label in range(5)]
    frame = manifest.loc[row_ids].copy()
    loader = make_loader(frame, augment=False, batch_size=5, workers=0, shuffle=False, seed=args.seed)
    images, labels = next(iter(loader))
    model = MSCCNet(num_classes=5, routing_iterations=args.routing_iterations).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    logits, lengths = model(images.to(device), return_capsules=True)
    loss = JointMSCCLoss(margin_weight=args.margin_weight)(logits, lengths, labels.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("MSCCNet smoke produced non-finite loss")
    loss.backward()
    optimizer.step()
    smoke_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        smoke_root / "smoke.json",
        {
            "state": "passed",
            "input_batch": list(images.shape),
            "output_shape": list(logits.shape),
            "capsule_lengths_shape": list(lengths.shape),
            "loss": float(loss.detach()),
            "num_classes": 5,
            "augmentation_variants": AUGMENTATION_VARIANTS,
            "heldout_scored": False,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--routing-iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = validate_manifest(args.manifest)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)
    source_fidelity = {
        "method": "MSCCNet",
        "dataset": "SIPaKMeD",
        "paper_doi": "10.1109/BIBM58861.2023.10385911",
        "implementation_status": "paper-guided PyTorch adaptation; author code/full text unavailable in current workspace",
        "recoverable_mechanisms": [
            "multi-scale convolution features",
            "cross-layer attention-based feature fusion",
            "spatial relationship modeling with capsule representations",
            "joint loss with increased penalty for misclassified samples",
        ],
        "explicit_adaptations": [
            "three 3x3/5x5/dilated-3x3 stem branches",
            "three convolution stages with 96/160/224 channels",
            "channel and spatial attention after cross-layer alignment",
            "16 primary capsules of dimension 8 and five digit capsules of dimension 16",
            "three dynamic-routing iterations",
            "CrossEntropy plus 0.5 times capsule margin loss",
            "six deterministic rotation/flip fit views; no augmentation on select or held-out",
        ],
        "paper_reported_sipakmed_accuracy": 0.9790,
        "paper_reported_score_is_not_rerun_result": True,
        "run_config": json_safe_args(args) | {"device_resolved": str(device), "model_config": CONFIG},
    }
    atomic_json(run_root / "source_fidelity.json", source_fidelity)
    atomic_json(
        run_root / "status.json",
        {
            "state": "initialized",
            "method": "MSCCNet",
            "dataset": "SIPaKMeD",
            "configured_folds": [int(item) for item in args.folds.split(",") if item.strip()],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    smoke_root = run_root / "smoke"
    run_smoke(manifest, device, args, smoke_root)
    if args.smoke_only:
        atomic_json(run_root / "status.json", {"state": "smoke_complete", "method": "MSCCNet", "updated_utc": datetime.now(timezone.utc).isoformat()})
        return
    results = []
    for fold in [int(item) for item in args.folds.split(",") if item.strip()]:
        results.append(train_fold(fold, manifest, run_root, device, args))
        write_summary(run_root, results)
    summary = write_summary(run_root, results)
    atomic_json(
        run_root / "status.json",
        {
            "state": "completed",
            "method": "MSCCNet",
            "dataset": "SIPaKMeD",
            "completed_folds": len(results),
            "summary": summary,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
