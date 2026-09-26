#!/usr/bin/env python3
"""Grouped five-fold MTFM adaptation on SIPaKMeD."""

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

from experiments.sipakmed.paper_baselines_v1.mtfm import FEATURE_NAMES, MTFM, manual_feature_vector, mtfm_loss


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "external_data/SIPAKMED/manifests_v2/manifest.csv"
DEFAULT_RESULTS = ROOT / "results/sipakmed_paper_arch_baselines_v1"
EXPECTED_CLASS_NAMES = ("Superficial-Intermediate", "Parabasal", "Koilocytotic", "Dyskeratotic", "Metaplastic")
IMAGE_SIZE = 224
AUGMENTATION_VARIANTS = ("identity", "rotate_90", "rotate_180", "flip_horizontal")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


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
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


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
    if set(manifest["label"].astype(int)) != set(range(5)) or set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly labels and folds 0 through 4")
    missing_files = [path for path in manifest["image_path"].tolist() if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest.reset_index(drop=True)


def make_internal_split(outer_train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, select_indices = next(splitter.split(outer_train["image_path"], outer_train["label"], outer_train["group_id"]))
    select = outer_train.iloc[select_indices].copy()
    fit = outer_train.drop(outer_train.index[select_indices]).copy()
    for name, frame in (("fit", fit), ("select", select)):
        if frame.empty or set(frame["label"].astype(int)) != set(range(5)):
            raise RuntimeError(f"internal {name} split does not contain all five classes")
    if set(fit["group_id"]) & set(select["group_id"]):
        raise RuntimeError("internal split has source-group leakage")
    return fit.reset_index(drop=True), select.reset_index(drop=True)


def image_tensor(path: str, variant: str) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if variant == "rotate_90":
        image = image.rotate(90, expand=True)
    elif variant == "rotate_180":
        image = image.rotate(180, expand=True)
    elif variant == "flip_horizontal":
        image = ImageOps.mirror(image)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


class MTFMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, augment: bool, feature_table: dict[str, np.ndarray]):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.variants = AUGMENTATION_VARIANTS if augment else ("identity",)
        self.feature_table = feature_table

    def __len__(self) -> int:
        return len(self.frame) * len(self.variants)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        row_index, variant_index = divmod(index, len(self.variants))
        row = self.frame.iloc[row_index]
        image = image_tensor(row.image_path, self.variants[variant_index])
        features = torch.from_numpy(self.feature_table[str(row.image_path)])
        return image, int(row.label), features


def make_loader(frame: pd.DataFrame, *, augment: bool, feature_table: dict[str, np.ndarray], batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        MTFMDataset(frame, augment=augment, feature_table=feature_table),
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
def evaluate_model(model: MTFM, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    binary_truths: list[int] = []
    binary_predictions: list[int] = []
    for images, labels, _ in loader:
        outputs = model(images.to(device, non_blocking=True))
        predictions.extend(outputs["five_logits"].argmax(1).cpu().tolist())
        binary_predictions.extend(outputs["binary_logits"].argmax(1).cpu().tolist())
        truths.extend(labels.tolist())
        binary_truths.extend((~torch.isin(labels, torch.tensor([0, 1, 4]))).long().tolist())
    metrics = classification_metrics(truths, predictions)
    metrics["binary_accuracy"] = float(accuracy_score(binary_truths, binary_predictions))
    metrics["binary_macro_f1"] = float(f1_score(binary_truths, binary_predictions, average="macro", zero_division=0))
    return metrics


def save_checkpoint(path: Path, model: MTFM, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, config: dict) -> None:
    torch.save({"method": "MTFM", "epoch": epoch, "selection_metrics": metrics, "config": config, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, path)


def build_feature_table(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for path in manifest["image_path"].tolist():
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
        table[str(path)] = manual_feature_vector(image)
    return table


def train_fold(fold: int, manifest: pd.DataFrame, feature_table: dict[str, np.ndarray], run_root: Path, device: torch.device, args: argparse.Namespace) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    fold_root.mkdir(parents=False, exist_ok=False)
    outer_train = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select = make_internal_split(outer_train, fold_seed)
    frames = {"fit": fit, "select": select, "heldout": heldout}
    groups = {name: set(frame.group_id.astype(str)) for name, frame in frames.items()}
    intersections = {f"{left}_x_{right}": len(groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))}
    if any(intersections.values()):
        raise RuntimeError(f"fold {fold} contains source-group leakage: {intersections}")
    split_dir = fold_root / "split"
    split_dir.mkdir()
    for name, frame in frames.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    atomic_json(fold_root / "split_audit.json", {"fold": fold, "counts": {name: len(frame) for name, frame in frames.items()}, "groups": {name: len(values) for name, values in groups.items()}, "pairwise_group_intersections": intersections, "classes": EXPECTED_CLASS_NAMES, "fit_views": len(AUGMENTATION_VARIANTS)})

    fit_loader = make_loader(fit, augment=True, feature_table=feature_table, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, feature_table=feature_table, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    model = MTFM().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    total_epochs = args.epochs

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / float(args.warmup_epochs)
        progress = (epoch_index - args.warmup_epochs) / max(1, total_epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    best_score = (-1.0, -1.0)
    best_record: dict | None = None
    epoch_log = fold_root / "epochs.jsonl"
    status_path = run_root / "status.json"
    for epoch in range(1, total_epochs + 1):
        started = time.time()
        model.train()
        sums = {"loss": 0.0, "five_class_loss": 0.0, "binary_loss": 0.0, "manual_loss": 0.0}
        seen = 0
        for batch_index, (images, labels, manual_targets) in enumerate(fit_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            manual_targets = manual_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = mtfm_loss(model(images), labels, manual_targets, lambda_manual=args.lambda_manual, alpha=args.alpha, beta=args.beta)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"fold {fold} epoch {epoch} produced non-finite loss")
            losses["loss"].backward()
            optimizer.step()
            batch_size = len(labels)
            for name in sums:
                sums[name] += float(losses[name].detach()) * batch_size
            seen += batch_size
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
        scheduler.step()
        select_metrics = evaluate_model(model, select_loader, device)
        record = {"fold": fold, "epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), **{name: value / max(seen, 1) for name, value in sums.items()}, "fit_samples_seen": seen, "select": select_metrics, "seconds": time.time() - started}
        with epoch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = (select_metrics["macro_f1"], select_metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_record = record
            save_checkpoint(fold_root / "best.pt", model, optimizer, epoch, select_metrics, json_safe_args(args))
        save_checkpoint(fold_root / "latest.pt", model, optimizer, epoch, select_metrics, json_safe_args(args))
        atomic_json(status_path, {"state": "training", "method": "MTFM", "fold": fold, "epoch": epoch, "total_epochs": total_epochs, "latest_select_macro_f1": select_metrics["macro_f1"], "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"fold={fold} epoch={epoch}/{total_epochs} loss={record['loss']:.4f} select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}", flush=True)

    if best_record is None:
        raise RuntimeError(f"no checkpoint was selected for fold {fold}")
    checkpoint = torch.load(fold_root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    heldout_loader = make_loader(heldout, augment=False, feature_table=feature_table, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    heldout_metrics = evaluate_model(model, heldout_loader, device)
    result = {"method": "MTFM", "dataset": "SIPaKMeD", "fold": fold, "seed": fold_seed, "n_fit": len(fit), "n_select": len(select), "n_heldout": len(heldout), "fit_views": len(AUGMENTATION_VARIANTS), "best_epoch": int(best_record["epoch"]), "heldout_metrics": heldout_metrics, "heldout_evaluation_count": 1, "class_names": EXPECTED_CLASS_NAMES, "predictions_recorded": True}
    atomic_json(fold_root / "result.json", result)
    atomic_json(status_path, {"state": "fold_complete", "method": "MTFM", "fold": fold, "result": heldout_metrics, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} macro_f1={heldout_metrics['macro_f1']:.4f}", flush=True)
    del model, optimizer, scheduler, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = [{"fold": result["fold"], **result["heldout_metrics"]} for result in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall", "binary_accuracy", "binary_macro_f1")
    summary = {"method": "MTFM", "dataset": "SIPaKMeD", "completed_folds": len(results), "metrics_mean": {name: float(table[name].mean()) for name in metric_names}, "metrics_sample_sd": {name: float(table[name].std(ddof=1)) for name in metric_names} if len(table) > 1 else {}, "folds": rows}
    atomic_json(run_root / "summary.json", summary)
    return summary


def run_smoke(manifest: pd.DataFrame, device: torch.device, args: argparse.Namespace, smoke_root: Path) -> None:
    rows = manifest.iloc[[0, 1]].copy()
    feature_table = build_feature_table(rows)
    loader = make_loader(rows, augment=False, feature_table=feature_table, batch_size=2, workers=0, shuffle=False, seed=args.seed)
    images, labels, manual_targets = next(iter(loader))
    model = MTFM(width_multiplier=0.25).to(device).eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        losses = mtfm_loss(outputs, labels.to(device), manual_targets.to(device))
    if not all(torch.isfinite(value).all() for value in list(outputs.values()) + list(losses.values())):
        raise FloatingPointError("MTFM smoke produced non-finite output")
    smoke_root.mkdir(parents=True, exist_ok=False)
    atomic_json(smoke_root / "smoke.json", {"state": "passed", "input_batch": list(images.shape), "loss": float(losses["loss"]), "backbone": model.backbone_name, "feature_names": FEATURE_NAMES, "updated_utc": datetime.now(timezone.utc).isoformat()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lambda-manual", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.6)
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
    source_fidelity = {"method": "MTFM", "dataset": "SIPaKMeD", "paper_doi": "10.1109/JBHI.2022.3180989", "implementation_status": "paper-guided adaptation; CE-Net manual-feature inputs and author weights are unavailable", "paper_reported_sipakmed_accuracy": 0.9867, "paper_reported_score_is_not_rerun_result": True, "recoverable_mechanisms": ["dual ResNet-50 branches", "manual-feature auxiliary fitting task", "five-class and two-class classification heads", "one-way layerwise manual-to-classification fusion", "similarity-aware label smoothing"], "explicit_adaptations": ["deterministic image-only proxy cell/nucleus masks", "six bounded proxy features matching paper feature families", "binary normal labels fixed to 0,1,4 and abnormal labels 2,3", "test-only width-0.25 ResNet-18 smoke path; training uses ResNet-50", "pretrained=False because author weights are unavailable", "200 epochs per fold fixed by user-approved speed compromise"], "proxy_feature_names": FEATURE_NAMES, "run_config": json_safe_args(args) | {"device_resolved": str(device)}}
    atomic_json(run_root / "source_fidelity.json", source_fidelity)
    atomic_json(run_root / "status.json", {"state": "initialized", "method": "MTFM", "dataset": "SIPaKMeD", "configured_folds": [int(item) for item in args.folds.split(",") if item.strip()], "updated_utc": datetime.now(timezone.utc).isoformat()})
    feature_table = build_feature_table(manifest)
    np.save(run_root / "proxy_features.npy", np.stack([feature_table[str(path)] for path in manifest["image_path"].tolist()]))
    run_smoke(manifest, device, args, run_root / "smoke")
    if args.smoke_only:
        atomic_json(run_root / "status.json", {"state": "smoke_complete", "method": "MTFM", "updated_utc": datetime.now(timezone.utc).isoformat()})
        return
    results = []
    for fold in [int(item) for item in args.folds.split(",") if item.strip()]:
        results.append(train_fold(fold, manifest, feature_table, run_root, device, args))
        write_summary(run_root, results)
    summary = write_summary(run_root, results)
    atomic_json(run_root / "status.json", {"state": "completed", "method": "MTFM", "dataset": "SIPaKMeD", "completed_folds": len(results), "summary": summary, "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
