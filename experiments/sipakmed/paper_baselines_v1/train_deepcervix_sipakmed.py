#!/usr/bin/env python3
"""Grouped five-fold PyTorch reproduction of DeepCervix-HDFF on SIPaKMeD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.sipakmed.paper_baselines_v1.deepcervix_hdff import (
    ARCHITECTURES,
    DeepCervixBranch,
    FeatureFusionHead,
    concatenate_embeddings,
    make_internal_split,
)


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "class_name", "label", "group_id", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks required columns: {sorted(missing)}")
    if len(manifest) != 4049:
        raise ValueError(f"expected the verified 4049-image SIPaKMeD manifest, found {len(manifest)}")
    if tuple(
        manifest.sort_values("label").drop_duplicates("label")["class_name"].tolist()
    ) != EXPECTED_CLASS_NAMES:
        raise ValueError("SIPaKMeD class-id mapping differs from the verified five-class mapping")
    if set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly outer folds 0 through 4")
    missing_files = [p for p in manifest["image_path"].tolist() if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest


class SIPaKMeDDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy()
        if self.augment:
            from scipy.ndimage import rotate

            if random.random() < 0.5:
                array = array[:, ::-1]
            if random.random() < 0.5:
                array = array[::-1, :]
            angle = random.uniform(-5.0, 5.0)
            if abs(angle) > 1e-3:
                array = rotate(array, angle, axes=(1, 0), reshape=False, order=1, mode="nearest")
            array *= random.uniform(0.5, 1.3)
            channel_shift = np.random.uniform(-20.0, 20.0, size=(1, 1, 3)).astype(np.float32)
            array = np.clip(array + channel_shift, 0.0, 255.0)
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor, int(row.label)


def make_loader(
    frame: pd.DataFrame,
    *,
    augment: bool,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SIPaKMeDDataset(frame, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=drop_last,
        generator=generator,
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
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[int], list[int]]:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits, _ = model(images)
        truths.extend(labels.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(truths, predictions), truths, predictions


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metric: dict, architecture: str) -> None:
    torch.save(
        {
            "architecture": architecture,
            "epoch": epoch,
            "selection_metrics": metric,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_branch(
    architecture: str,
    fit_loader: DataLoader,
    select_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    status_path: Path,
) -> dict:
    branch_dir = output_dir / "branches" / architecture
    branch_dir.mkdir(parents=True, exist_ok=False)
    model = DeepCervixBranch(architecture, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    if architecture == "resnet50":
        stage2_lr = 1e-5
    elif architecture == "vgg19":
        stage2_lr = 1e-4
    else:
        # VGG16 is inferred from the companion VGG19 recipe; Xception's
        # training cell is absent from the author repository.
        stage2_lr = 1e-4
    stages = ((args.epochs_stage1, 1e-3), (args.epochs_stage2, stage2_lr))
    global_epoch = 0
    best_score = (-1.0, -1.0)
    best_record: dict | None = None
    epoch_log = branch_dir / "epochs.jsonl"

    for stage_number, (stage_epochs, learning_rate) in enumerate(stages, start=1):
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            started = time.time()
            model.train()
            total_loss = 0.0
            seen = 0
            for batch_index, (images, labels) in enumerate(fit_loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                seen += len(labels)
                if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                    break
            select_metrics, _, _ = evaluate_classifier(model, select_loader, device)
            score = (select_metrics["macro_f1"], select_metrics["accuracy"])
            record = {
                "architecture": architecture,
                "epoch": global_epoch,
                "stage": stage_number,
                "stage_epoch": stage_epoch,
                "learning_rate": learning_rate,
                "fit_loss": total_loss / max(seen, 1),
                "select": select_metrics,
                "seconds": time.time() - started,
            }
            with epoch_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if score > best_score:
                best_score = score
                best_record = record
                save_checkpoint(branch_dir / "best.pt", model, global_epoch, select_metrics, architecture)
            save_checkpoint(branch_dir / "latest.pt", model, global_epoch, select_metrics, architecture)
            status = {
                "state": "training_branch",
                "fold": int(output_dir.name.split("_")[-1]),
                "architecture": architecture,
                "epoch": global_epoch,
                "total_epochs": args.epochs_stage1 + args.epochs_stage2,
                "latest_select_macro_f1": select_metrics["macro_f1"],
                "elapsed_seconds": time.time() - started,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(status_path, status)
            print(
                f"fold={status['fold']} branch={architecture} epoch={global_epoch}/"
                f"{args.epochs_stage1 + args.epochs_stage2} loss={record['fit_loss']:.4f} "
                f"select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}",
                flush=True,
            )
            if args.max_epochs_per_stage and stage_epoch >= args.max_epochs_per_stage:
                break

    if best_record is None:
        raise RuntimeError(f"no checkpoint was selected for {architecture}")
    checkpoint = torch.load(branch_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint
    return {
        "model": model,
        "best_epoch": int(best_record["epoch"]),
        "best_selection": best_record["select"],
        "stage2_learning_rate": stage2_lr,
    }


@torch.no_grad()
def extract_embeddings(model: DeepCervixBranch, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        _, embedding = model(images)
        features.append(embedding.cpu())
        labels_all.append(labels.cpu())
    return torch.cat(features), torch.cat(labels_all)


def train_fusion_head(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    select_features: torch.Tensor,
    select_labels: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[FeatureFusionHead, dict]:
    model = FeatureFusionHead(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(fit_features, fit_labels),
        batch_size=16,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_metrics: dict = {}
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (features, labels) in enumerate(loader, start=1):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
        model.eval()
        with torch.no_grad():
            select_logits = model(select_features.to(device))
            pred = select_logits.argmax(1).cpu().tolist()
        metrics = classification_metrics(select_labels.tolist(), pred)
        score = (metrics["macro_f1"], metrics["accuracy"])
        print(
            f"fusion epoch={epoch}/{args.head_epochs} loss={loss_sum/max(seen,1):.4f} "
            f"select_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                {"epoch": epoch, "selection_metrics": metrics, "model_state_dict": model.state_dict()},
                output_dir / "fusion_head_best.pt",
            )
    saved = torch.load(output_dir / "fusion_head_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, {"best_epoch": best_epoch, "best_selection": best_metrics}


@torch.no_grad()
def predict_fusion(
    branch_models: list[DeepCervixBranch],
    head: FeatureFusionHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    truths: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        branch_embeddings = [model(images)[1] for model in branch_models]
        logits = head(concatenate_embeddings(branch_embeddings))
        truths.extend(labels.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return truths, predictions


def run_fold(fold: int, manifest: pd.DataFrame, run_root: Path, device: torch.device, args: argparse.Namespace) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=False)
    outer_train = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select = make_internal_split(outer_train, fold_seed)
    for name, frame in (("fit", fit), ("select", select), ("heldout", heldout)):
        if frame.empty or set(frame.label.astype(int)) != set(range(5)):
            raise RuntimeError(f"fold {fold} {name} split is empty or missing a class")
    groups = {name: set(frame.group_id) for name, frame in (("fit", fit), ("select", select), ("heldout", heldout))}
    if groups["fit"] & groups["select"] or groups["fit"] & groups["heldout"] or groups["select"] & groups["heldout"]:
        raise RuntimeError(f"fold {fold} contains source-group leakage")
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
            "groups": {name: len(values) for name, values in groups.items()},
            "pairwise_group_intersections": 0,
            "classes": EXPECTED_CLASS_NAMES,
        },
    )
    fit_loader = make_loader(fit, augment=True, batch_size=32, workers=args.workers, shuffle=True, drop_last=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, batch_size=64, workers=args.workers, shuffle=False, seed=fold_seed)
    status_path = run_root / "status.json"
    branch_models: list[DeepCervixBranch] = []
    branch_records: dict = {}
    for architecture in ARCHITECTURES:
        record = train_branch(architecture, fit_loader, select_loader, fold_root, device, args, fold_seed, status_path)
        branch_models.append(record.pop("model"))
        branch_records[architecture] = record
        torch.cuda.empty_cache()

    embedding_loaders = {
        name: make_loader(frame, augment=False, batch_size=64, workers=args.workers, shuffle=False, seed=fold_seed)
        for name, frame in (("fit", fit), ("select", select), ("heldout", heldout))
    }
    split_embeddings: dict[str, torch.Tensor] = {}
    split_labels: dict[str, torch.Tensor] = {}
    for name, loader in embedding_loaders.items():
        branch_features = []
        labels_ref = None
        for model in branch_models:
            features, labels = extract_embeddings(model, loader, device)
            branch_features.append(features)
            if labels_ref is None:
                labels_ref = labels
            elif not torch.equal(labels_ref, labels):
                raise RuntimeError(f"branch embedding label order mismatch in {name} split")
        split_embeddings[name] = torch.cat(branch_features, dim=1)
        split_labels[name] = labels_ref
    if split_embeddings["fit"].shape[1] != 4096:
        raise RuntimeError("four branch embeddings did not concatenate to 4096 features")

    head, fusion_info = train_fusion_head(
        split_embeddings["fit"],
        split_labels["fit"],
        split_embeddings["select"],
        split_labels["select"],
        fold_root,
        device,
        args,
    )
    head.eval()
    heldout_logits = head(split_embeddings["heldout"].to(device))
    heldout_predictions = heldout_logits.argmax(1).cpu().tolist()
    metrics = classification_metrics(split_labels["heldout"].tolist(), heldout_predictions)
    result = {
        "method": "DeepCervix-HDFF",
        "dataset": "SIPaKMeD",
        "fold": fold,
        "seed": fold_seed,
        "n_fit": len(fit),
        "n_select": len(select),
        "n_heldout": len(heldout),
        "branch_models": branch_records,
        "fusion_head": fusion_info,
        "heldout_metrics": metrics,
        "class_names": EXPECTED_CLASS_NAMES,
        "heldout_evaluation_count": 1,
    }
    atomic_json(fold_root / "result.json", result)
    print(
        f"FOLD COMPLETE fold={fold} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    atomic_json(status_path, {"state": "fold_complete", "fold": fold, "result": metrics, "updated_utc": datetime.now(timezone.utc).isoformat()})
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = []
    for result in results:
        row = {"fold": result["fold"], **result["heldout_metrics"]}
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    summary = {
        "method": "DeepCervix-HDFF",
        "dataset": "SIPaKMeD",
        "completed_folds": len(results),
        "metrics_mean": {column: float(table[column].mean()) for column in ("accuracy", "macro_f1", "macro_precision", "macro_recall")},
        "metrics_sample_sd": {column: float(table[column].std(ddof=1)) for column in ("accuracy", "macro_f1", "macro_precision", "macro_recall")} if len(table) > 1 else {},
        "folds": rows,
    }
    atomic_json(run_root / "summary.json", summary)
    return summary


def run_smoke(manifest: pd.DataFrame, device: torch.device, args: argparse.Namespace, smoke_root: Path) -> None:
    row_ids = []
    for label in range(5):
        row_ids.append(int(manifest.index[manifest.label == label][0]))
    batch_frame = manifest.loc[row_ids].copy()
    images, labels = next(iter(make_loader(batch_frame, augment=False, batch_size=5, workers=0, shuffle=False)))
    images, labels = images.to(device), labels.to(device)
    embeddings = []
    for architecture in ARCHITECTURES:
        model = DeepCervixBranch(architecture, pretrained=False).to(device).train()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        logits, embedding = model(images)
        nn.CrossEntropyLoss()(logits, labels).backward()
        optimizer.step()
        embeddings.append(embedding.detach())
        del model, optimizer
        torch.cuda.empty_cache()
    fused = concatenate_embeddings(embeddings)
    head = FeatureFusionHead().to(device).train()
    head_optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    head_optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(head(fused), labels).backward()
    head_optimizer.step()
    atomic_json(
        smoke_root / "smoke.json",
        {
            "state": "passed",
            "input_batch": list(images.shape),
            "branch_embedding_shape": list(embeddings[0].shape),
            "fused_embedding_shape": list(fused.shape),
            "num_classes": 5,
            "heldout_scored": False,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"SMOKE PASSED: 4 branch backward steps; fused={tuple(fused.shape)}; heldout not evaluated", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=50)
    parser.add_argument("--head-epochs", type=int, default=200)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0, help="debug only; 0 uses all batches")
    parser.add_argument("--max-epochs-per-stage", type=int, default=0, help="debug only; 0 uses configured stage epochs")
    parser.add_argument("--smoke", action="store_true", help="run forward/backward plumbing only; does not score held-out data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.smoke:
        smoke_root = args.results_root / "_smoke_deepcervix_hdff"
        smoke_root.mkdir(parents=True, exist_ok=True)
        manifest = validate_manifest(args.manifest)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_smoke(manifest, device, args, smoke_root)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the full reproduction on CPU")
    if args.run_name is None:
        args.run_name = datetime.now(timezone.utc).strftime("deepcervix_hdff_%Y%m%dT%H%M%SZ")
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = validate_manifest(args.manifest)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 maps server GPU 1 to local cuda:0.
    seed_everything(args.seed)
    source = {
        "paper_doi": "10.1016/j.compbiomed.2021.104649",
        "author_repository": "https://github.com/Mamunur-20/DeepCervix",
        "author_repository_commit": "f161ea1dfdeb524d8765593c44135e1cf5a10240",
        "implementation_status": "paper-guided PyTorch reimplementation; not an exact-code rerun",
        "author_architecture": "VGG16 + VGG19 + Xception + ResNet50; four 1024-D branch embeddings; 4096-D fusion with Dropout(0.5), BatchNorm and five-way classifier",
        "author_recipe": "224x224 RGB; batch 32 branch training; rotation ±5°, horizontal/vertical flips, brightness [0.5,1.3], channel shift ±20; 50 epochs at Adam 1e-3 then 50 at backbone-specific lower LR; fusion head batch 16, Adam 1e-3, 200 epochs",
        "split_deviation": "author notebook uses cell-level train/validation/test partitions; this rerun uses the user's fixed source-group-disjoint five-fold SIPaKMeD manifest and an inner group-disjoint selection split",
        "code_adaptations": [
            "PyTorch/timm implementation; canonical timm legacy_xception is used because the checked-in five-class author notebook loads an external Xception model without its construction/training cell.",
            "VGG16 stage-2 LR 1e-4 is inferred from the companion VGG19 recipe; Xception stage-2 LR 1e-4 is an explicit adaptation because its recipe is absent.",
            "Use validation macro-F1 (accuracy tie-break) to select branch and fusion checkpoints; author notebook reports fixed-epoch models.",
            "Input pixel range and augmentation follow the checked-in ImageDataGenerator defaults as closely as possible; geometric rotation uses SciPy nearest-edge interpolation.",
        ],
        "run_config": {
            "manifest": str(args.manifest),
            "outer_folds": [0, 1, 2, 3, 4],
            "requested_folds": [int(value) for value in args.folds.split(",")],
            "seed": args.seed,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "head_epochs": args.head_epochs,
            "workers": args.workers,
            "image_size": IMAGE_SIZE,
            "batch_size_branch": 32,
            "batch_size_fusion": 16,
            "cuda_visible_device": "1",
        },
    }
    atomic_json(run_root / "source_fidelity.json", source)
    atomic_json(run_root / "status.json", {"state": "starting", "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"DEVICE={torch.cuda.get_device_name(device)}; MANIFEST={args.manifest}; rows={len(manifest)}", flush=True)
    results: list[dict] = []
    requested_folds = [int(value) for value in args.folds.split(",")]
    for fold in requested_folds:
        results.append(run_fold(fold, manifest, run_root, device, args))
        summary = write_summary(run_root, results)
        print(f"SUMMARY completed={summary['completed_folds']} folds mean={summary['metrics_mean']}", flush=True)
    atomic_json(run_root / "status.json", {"state": "complete", "completed_folds": len(results), "updated_utc": datetime.now(timezone.utc).isoformat()})
    print("ALL REQUESTED FOLDS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
