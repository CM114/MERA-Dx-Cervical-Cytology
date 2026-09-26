#!/usr/bin/env python3
"""Grouped five-fold CerCan-Net feature-ensemble adaptation on SIPaKMeD."""

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
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.sipakmed.paper_baselines_v1.cercan import CONFIG, CerCanNet


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
    if set(manifest["label"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly labels 0 through 4")
    if set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly outer folds 0 through 4")
    missing_files = [path for path in manifest["image_path"].tolist() if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest


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


class CerCanTrainingDataset(Dataset):
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
        CerCanTrainingDataset(frame, augment),
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
def evaluate_model(model: CerCanNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        predictions.extend(model(images.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        truths.extend(labels.tolist())
    return classification_metrics(truths, predictions)


@torch.no_grad()
def extract_features(model: CerCanNet, frame: pd.DataFrame, device: torch.device, batch_size: int, workers: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(frame, augment=False, batch_size=batch_size, workers=workers, shuffle=False, seed=seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for images, batch_labels in loader:
        features.append(model.forward_features(images.to(device, non_blocking=True)).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features, axis=0).astype(np.float32), np.concatenate(labels, axis=0).astype(np.int64)


class TopKFeatureSelector:
    def __init__(self, k: int = 400):
        self.k = int(k)
        self.indices_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TopKFeatureSelector":
        scores, _ = f_classif(features, labels)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-scores, kind="stable")
        self.indices_ = order[: min(self.k, features.shape[1])]
        self.scores_ = scores
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.indices_ is None:
            raise RuntimeError("feature selector must be fitted before transform")
        return np.asarray(features[:, self.indices_], dtype=np.float32)


def save_checkpoint(path: Path, model: CerCanNet, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict) -> None:
    torch.save({"method": "CerCan-Net", "epoch": epoch, "selection_metrics": metrics, "model_config": CONFIG, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, path)


def train_fold(fold: int, manifest: pd.DataFrame, run_root: Path, device: torch.device, args: argparse.Namespace) -> dict:
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

    fit_loader = make_loader(fit, augment=True, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    model = CerCanNet(num_classes=5, width_multiplier=args.width_multiplier).to(device)
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
        select_metrics = evaluate_model(model, select_loader, device)
        record = {"fold": fold, "epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), "fit_loss": loss_sum / max(seen, 1), "fit_samples_seen": seen, "select": select_metrics, "seconds": time.time() - started}
        with epoch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = (select_metrics["macro_f1"], select_metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_record = record
            save_checkpoint(fold_root / "best.pt", model, optimizer, epoch, select_metrics)
        save_checkpoint(fold_root / "latest.pt", model, optimizer, epoch, select_metrics)
        atomic_json(status_path, {"state": "training", "method": "CerCan-Net", "fold": fold, "epoch": epoch, "total_epochs": total_epochs, "latest_select_macro_f1": select_metrics["macro_f1"], "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"fold={fold} epoch={epoch}/{total_epochs} loss={record['fit_loss']:.4f} select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}", flush=True)

    if best_record is None:
        raise RuntimeError(f"no checkpoint was selected for fold {fold}")
    checkpoint = torch.load(fold_root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    x_fit, y_fit = extract_features(model, fit, device, args.eval_batch_size, args.workers, fold_seed)
    x_select, y_select = extract_features(model, select, device, args.eval_batch_size, args.workers, fold_seed)
    x_heldout, y_heldout = extract_features(model, heldout, device, args.eval_batch_size, args.workers, fold_seed)
    selector = TopKFeatureSelector(k=args.selected_features).fit(x_fit, y_fit)
    z_fit, z_select, z_heldout = selector.transform(x_fit), selector.transform(x_select), selector.transform(x_heldout)
    candidate_cs = [float(item) for item in args.svm_cs.split(",") if item.strip()]
    best_svm = None
    best_svm_record = None
    for c_value in candidate_cs:
        svm = LinearSVC(C=c_value, dual=False, max_iter=args.svm_max_iter, random_state=fold_seed)
        svm.fit(z_fit, y_fit)
        select_metrics = classification_metrics(y_select, svm.predict(z_select))
        score = (select_metrics["macro_f1"], select_metrics["accuracy"])
        if best_svm_record is None or score > (best_svm_record["metrics"]["macro_f1"], best_svm_record["metrics"]["accuracy"]):
            best_svm = svm
            best_svm_record = {"C": c_value, "metrics": select_metrics}
    if best_svm is None or best_svm_record is None:
        raise RuntimeError("no SVM candidate was fitted")
    heldout_predictions = best_svm.predict(z_heldout)
    heldout_metrics = classification_metrics(y_heldout, heldout_predictions)
    np.savez_compressed(fold_root / "features.npz", fit=x_fit, select=x_select, heldout=x_heldout)
    np.savez_compressed(fold_root / "svm.npz", coef=best_svm.coef_, intercept=best_svm.intercept_, classes=best_svm.classes_)
    atomic_json(fold_root / "feature_selection.json", {"method": "f_classif top-k", "fit_only": True, "selected_feature_count": int(len(selector.indices_)), "selected_indices": selector.indices_.tolist(), "score_max": float(np.max(selector.scores_))})
    result = {"method": "CerCan-Net", "dataset": "SIPaKMeD", "fold": fold, "seed": fold_seed, "n_fit": len(fit), "n_select": len(select), "n_heldout": len(heldout), "fit_views": len(AUGMENTATION_VARIANTS), "best_backbone_epoch": int(best_record["epoch"]), "selected_feature_count": int(len(selector.indices_)), "selected_classifier": best_svm_record, "heldout_metrics": heldout_metrics, "heldout_evaluation_count": 1, "class_names": EXPECTED_CLASS_NAMES, "predictions_recorded": len(y_heldout) == len(heldout_predictions) == len(heldout)}
    atomic_json(fold_root / "result.json", result)
    atomic_json(status_path, {"state": "fold_complete", "method": "CerCan-Net", "fold": fold, "result": heldout_metrics, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} macro_f1={heldout_metrics['macro_f1']:.4f}", flush=True)
    del model, optimizer, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = [{"fold": result["fold"], **result["heldout_metrics"]} for result in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary = {"method": "CerCan-Net", "dataset": "SIPaKMeD", "completed_folds": len(results), "metrics_mean": {name: float(table[name].mean()) for name in metric_names}, "metrics_sample_sd": {name: float(table[name].std(ddof=1)) for name in metric_names} if len(table) > 1 else {}, "folds": rows}
    atomic_json(run_root / "summary.json", summary)
    return summary


def run_smoke(manifest: pd.DataFrame, device: torch.device, args: argparse.Namespace, smoke_root: Path) -> None:
    row_ids = [int(manifest.index[manifest.label == label][0]) for label in range(5)]
    loader = make_loader(manifest.loc[row_ids].copy(), augment=False, batch_size=5, workers=0, shuffle=False, seed=args.seed)
    images, labels = next(iter(loader))
    model = CerCanNet(num_classes=5, width_multiplier=args.width_multiplier).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("CerCan-Net smoke produced non-finite loss")
    loss.backward()
    optimizer.step()
    smoke_root.mkdir(parents=True, exist_ok=False)
    atomic_json(smoke_root / "smoke.json", {"state": "passed", "input_batch": list(images.shape), "loss": float(loss.detach()), "backbones": CONFIG["backbones"], "feature_levels": CONFIG["feature_levels"], "updated_utc": datetime.now(timezone.utc).isoformat()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-features", type=int, default=400)
    parser.add_argument("--svm-cs", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--width-multiplier", type=float, default=0.5)
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
    source_fidelity = {"method": "CerCan-Net", "dataset": "SIPaKMeD", "paper_doi": "10.1016/j.eswa.2023.120624", "implementation_status": "paper-guided PyTorch adaptation; author code/full text unavailable in current workspace", "paper_reported_sipakmed_accuracy": 0.977, "paper_reported_score_is_not_rerun_result": True, "recoverable_mechanisms": ["MobileNet/DarkNet-19/ResNet-18 lightweight CNN ensemble", "last-three-deep-level feature extraction", "corresponding-level cross-CNN fusion", "DWT reduction and feature selection"], "explicit_adaptations": ["custom width-0.5 MobileNetV1 and DarkNet19 implementations plus width-0.5 ResNet18", "single-level Haar low-pass used as deterministic DWT reduction", "top-k f_classif selector with k=400 fitted on outer-fit rows only", "LinearSVC classifier with C chosen from inner selection rows only", "six deterministic rotation/flip views used for fit-side backbone training only", "pretrained=False because author/ImageNet weights were not locally available and the actual initialization is recorded"], "run_config": json_safe_args(args) | {"device_resolved": str(device), "model_config": CONFIG}}
    atomic_json(run_root / "source_fidelity.json", source_fidelity)
    atomic_json(run_root / "status.json", {"state": "initialized", "method": "CerCan-Net", "dataset": "SIPaKMeD", "configured_folds": [int(item) for item in args.folds.split(",") if item.strip()], "updated_utc": datetime.now(timezone.utc).isoformat()})
    run_smoke(manifest, device, args, run_root / "smoke")
    if args.smoke_only:
        atomic_json(run_root / "status.json", {"state": "smoke_complete", "method": "CerCan-Net", "updated_utc": datetime.now(timezone.utc).isoformat()})
        return
    results = []
    for fold in [int(item) for item in args.folds.split(",") if item.strip()]:
        results.append(train_fold(fold, manifest, run_root, device, args))
        write_summary(run_root, results)
    summary = write_summary(run_root, results)
    atomic_json(run_root / "status.json", {"state": "completed", "method": "CerCan-Net", "dataset": "SIPaKMeD", "completed_folds": len(results), "summary": summary, "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
