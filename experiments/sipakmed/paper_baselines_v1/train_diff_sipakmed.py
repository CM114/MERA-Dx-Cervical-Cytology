#!/usr/bin/env python3
"""Grouped five-fold DIFF adaptation on SIPaKMeD."""

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

from experiments.sipakmed.paper_baselines_v1.diff import DIFFNet, MODEL_CONFIG


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
AUGMENTATION_VARIANTS = ("identity", "rotate_90", "rotate_180", "rotate_270", "flip_horizontal", "flip_vertical")


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


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_safe_args(args: argparse.Namespace) -> dict:
    return json_safe(vars(args))


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
    if not manifest["image_path"].is_unique:
        raise ValueError("manifest image paths must be unique")
    missing_files = [path for path in manifest["image_path"].tolist() if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest


def make_internal_split(outer_train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, select_indices = next(splitter.split(outer_train["image_path"], outer_train["label"], outer_train["group_id"]))
    select = outer_train.iloc[select_indices].copy()
    fit = outer_train.drop(outer_train.index[select_indices]).copy()
    for name, frame in (("fit", fit), ("select", select)):
        if frame.empty or set(frame["label"].astype(int)) != set(range(5)):
            raise RuntimeError(f"internal {name} split does not contain all five classes")
    intersections = {
        "fit_x_select": len(set(fit["group_id"]) & set(select["group_id"])),
    }
    if any(intersections.values()):
        raise RuntimeError(f"internal split has source-group leakage: {intersections}")
    audit = {
        "seed": seed,
        "counts": {"fit": len(fit), "select": len(select)},
        "groups": {"fit": int(fit["group_id"].nunique()), "select": int(select["group_id"].nunique())},
        "pairwise_group_intersections": intersections,
    }
    return fit.reset_index(drop=True), select.reset_index(drop=True), audit


class DIFFTrainingDataset(Dataset):
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
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))), int(row.label)


def make_loader(frame: pd.DataFrame, *, augment: bool, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        DIFFTrainingDataset(frame, augment),
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
def evaluate_model(model: DIFFNet, loader: DataLoader, device: torch.device) -> tuple[dict, list[int], list[int]]:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        predictions.extend(model(images.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        truths.extend(labels.tolist())
    return classification_metrics(truths, predictions), truths, predictions


def save_checkpoint(path: Path, model: DIFFNet, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    torch.save(
        {
            "method": "DIFF",
            "epoch": epoch,
            "selection_metrics": metrics,
            "model_config": {**MODEL_CONFIG, "channels": args.channels, "depth": args.depth, "patch_size": args.patch_size},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_fold(fold: int, manifest: pd.DataFrame, run_root: Path, device: torch.device, args: argparse.Namespace) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    fold_root.mkdir(parents=False, exist_ok=False)
    outer_train = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select, internal_audit = make_internal_split(outer_train, fold_seed)
    frames = {"fit": fit, "select": select, "heldout": heldout}
    groups = {name: set(frame.group_id.astype(str)) for name, frame in frames.items()}
    intersections = {f"{left}_x_{right}": len(groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))}
    if any(intersections.values()):
        raise RuntimeError(f"fold {fold} contains source-group leakage: {intersections}")
    split_dir = fold_root / "split"
    split_dir.mkdir()
    for name, frame in frames.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    atomic_json(
        fold_root / "split_audit.json",
        {
            "fold": fold,
            "counts": {name: len(frame) for name, frame in frames.items()},
            "groups": {name: len(values) for name, values in groups.items()},
            "pairwise_group_intersections": intersections,
            "internal_split": internal_audit,
            "classes": EXPECTED_CLASS_NAMES,
            "fit_views": len(AUGMENTATION_VARIANTS),
        },
    )

    fit_loader = make_loader(fit, augment=True, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    heldout_loader = make_loader(heldout, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    model = DIFFNet(num_classes=5, channels=args.channels, depth=args.depth, patch_size=args.patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            loss = nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fold {fold} epoch {epoch} produced non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
        select_metrics, _, _ = evaluate_model(model, select_loader, device)
        record = {
            "fold": fold,
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
            save_checkpoint(fold_root / "best.pt", model, optimizer, epoch, select_metrics, args)
        save_checkpoint(fold_root / "latest.pt", model, optimizer, epoch, select_metrics, args)
        atomic_json(
            status_path,
            {
                "state": "training",
                "method": "DIFF",
                "fold": fold,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "latest_select_macro_f1": select_metrics["macro_f1"],
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"fold={fold} epoch={epoch}/{total_epochs} loss={record['fit_loss']:.4f} select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}", flush=True)

    if best_record is None:
        raise RuntimeError(f"no checkpoint was selected for fold {fold}")
    checkpoint = torch.load(fold_root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    heldout_metrics, heldout_truths, heldout_predictions = evaluate_model(model, heldout_loader, device)
    np.savez_compressed(fold_root / "predictions.npz", y_true=np.asarray(heldout_truths, dtype=np.int64), y_pred=np.asarray(heldout_predictions, dtype=np.int64))
    result = {
        "method": "DIFF",
        "dataset": "SIPaKMeD",
        "fold": fold,
        "seed": fold_seed,
        "n_fit": len(fit),
        "n_select": len(select),
        "n_heldout": len(heldout),
        "fit_views": len(AUGMENTATION_VARIANTS),
        "best_epoch": int(best_record["epoch"]),
        "heldout_metrics": heldout_metrics,
        "heldout_evaluation_count": 1,
        "class_names": EXPECTED_CLASS_NAMES,
        "predictions_recorded": len(heldout_truths) == len(heldout_predictions) == len(heldout),
        "model_config": {**MODEL_CONFIG, "channels": args.channels, "depth": args.depth, "patch_size": args.patch_size},
    }
    atomic_json(fold_root / "result.json", result)
    atomic_json(status_path, {"state": "fold_complete", "method": "DIFF", "fold": fold, "result": heldout_metrics, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} macro_f1={heldout_metrics['macro_f1']:.4f}", flush=True)
    del model, optimizer, checkpoint, fit_loader, select_loader, heldout_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = [{"fold": result["fold"], **result["heldout_metrics"]} for result in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary = {
        "method": "DIFF",
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
    loader = make_loader(manifest.loc[row_ids].copy(), augment=False, batch_size=5, workers=0, shuffle=False, seed=args.seed)
    images, labels = next(iter(loader))
    model = DIFFNet(num_classes=5, channels=args.channels, depth=args.depth, patch_size=args.patch_size).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("DIFF smoke produced non-finite loss")
    loss.backward()
    optimizer.step()
    smoke_root.mkdir(parents=False, exist_ok=False)
    atomic_json(smoke_root / "smoke.json", {"state": "passed", "input_batch": list(images.shape), "loss": float(loss.detach()), "model_config": {**MODEL_CONFIG, "channels": args.channels, "depth": args.depth, "patch_size": args.patch_size}, "updated_utc": datetime.now(timezone.utc).isoformat()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
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
        "method": "DIFF",
        "dataset": "SIPaKMeD",
        "paper_doi": "10.1016/j.compbiomed.2024.108153",
        "implementation_status": "paper-guided PyTorch adaptation; author code and complete source configuration unavailable in current workspace",
        "paper_reported_sipakmed_accuracy": 0.9602,
        "paper_reported_score_is_not_rerun_result": True,
        "recoverable_mechanisms": [
            "local CNN branch",
            "global visual-transformer branch",
            "repeated Deep Integrated Feature Fusion blocks",
            "cross-channel 1x1 fusion with residual connections",
        ],
        "explicit_adaptations": [
            "224x224 input and 32-channel stem for the shared SIPaKMeD protocol",
            "three repeated branch/fusion blocks",
            "transformer patch size 4 with embedding dimension P^2*C=512",
            "30-epoch bounded default instead of the source material's 100-epoch description",
            "pretrained=False because no exact local checkpoint was available",
            "six deterministic rotation/flip views for fit-side training only",
        ],
        "run_config": json_safe_args(args) | {"device_resolved": str(device), "model_config": {**MODEL_CONFIG, "channels": args.channels, "depth": args.depth, "patch_size": args.patch_size}},
    }
    atomic_json(run_root / "source_fidelity.json", source_fidelity)
    atomic_json(run_root / "status.json", {"state": "initialized", "method": "DIFF", "dataset": "SIPaKMeD", "configured_folds": [int(item) for item in args.folds.split(",") if item.strip()], "updated_utc": datetime.now(timezone.utc).isoformat()})
    run_smoke(manifest, device, args, run_root / "smoke")
    if args.smoke_only:
        atomic_json(run_root / "status.json", {"state": "smoke_complete", "method": "DIFF", "updated_utc": datetime.now(timezone.utc).isoformat()})
        return
    results = []
    for fold in [int(item) for item in args.folds.split(",") if item.strip()]:
        results.append(train_fold(fold, manifest, run_root, device, args))
        write_summary(run_root, results)
    summary = write_summary(run_root, results)
    atomic_json(run_root / "status.json", {"state": "completed", "method": "DIFF", "dataset": "SIPaKMeD", "completed_folds": len(results), "summary": summary, "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
