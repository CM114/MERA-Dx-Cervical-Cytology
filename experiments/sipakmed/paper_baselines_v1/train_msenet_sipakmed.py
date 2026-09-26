#!/usr/bin/env python3
"""Grouped five-fold MSENet adaptation on SIPaKMeD."""

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
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiments.sipakmed.paper_baselines_v1.msenet import MSENET_CONFIG, MSENetBase, enhance_probabilities


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "external_data/SIPAKMED/manifests_v2/manifest.csv"
DEFAULT_RESULTS = ROOT / "results/sipakmed_paper_arch_baselines_v1"
EXPECTED_CLASS_NAMES = ("Superficial-Intermediate", "Parabasal", "Koilocytotic", "Dyskeratotic", "Metaplastic")


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


def validate_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "class_name", "label", "group_id", "fold"}
    missing = required - set(manifest.columns)
    if missing or len(manifest) != 4049:
        raise ValueError(f"invalid verified SIPaKMeD manifest: missing={sorted(missing)} rows={len(manifest)}")
    class_rows = manifest.sort_values("label").drop_duplicates("label")
    if tuple(class_rows["class_name"].tolist()) != EXPECTED_CLASS_NAMES:
        raise ValueError("SIPaKMeD class mapping differs from the verified mapping")
    if set(manifest["label"].astype(int)) != set(range(5)) or set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain labels and outer folds 0 through 4")
    missing_files = [path for path in manifest["image_path"].tolist() if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images: {missing_files[:3]}")
    return manifest.reset_index(drop=True)


def make_internal_split(outer_train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, selected = next(splitter.split(outer_train["image_path"], outer_train["label"], outer_train["group_id"]))
    select = outer_train.iloc[selected].copy()
    fit = outer_train.drop(outer_train.index[selected]).copy()
    if set(fit["group_id"]) & set(select["group_id"]):
        raise RuntimeError("internal fit/select split has group leakage")
    return fit.reset_index(drop=True), select.reset_index(drop=True)


class MSENetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool, image_size: int = 256):
        self.frame = frame.reset_index(drop=True)
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomAffine(degrees=40, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=11),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row.label)


def make_loader(frame: pd.DataFrame, *, augment: bool, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        MSENetDataset(frame, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, return_probabilities: bool = False):
    model.eval()
    truths: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in loader:
        probabilities.append(model(images.to(device, non_blocking=True)).softmax(1).cpu().numpy())
        truths.extend(labels.tolist())
    probs = np.concatenate(probabilities, axis=0)
    predictions = probs.argmax(1)
    result = metrics(truths, predictions)
    if return_probabilities:
        return result, np.asarray(truths, dtype=np.int64), probs
    return result


def train_one_backbone(
    name: str,
    fit: pd.DataFrame,
    select: pd.DataFrame,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    seed_everything(seed)
    model = MSENetBase(name, num_classes=5, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    fit_loader = make_loader(fit, augment=True, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=seed)
    select_loader = make_loader(select, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=seed)
    best_score = (-1.0, -1.0)
    best_record = None
    log_path = output / "epochs.jsonl"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_index, (images, labels) in enumerate(fit_loader, start=1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
        scheduler.step()
        select_metrics = evaluate(model, select_loader, device)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss": loss_sum / max(count, 1),
            "select": select_metrics,
            "seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if (select_metrics["macro_f1"], select_metrics["accuracy"]) > best_score:
            best_score = (select_metrics["macro_f1"], select_metrics["accuracy"])
            best_record = record
            torch.save({"method": "MSENet", "backbone": name, "epoch": epoch, "model_state_dict": model.state_dict(), "selection_metrics": select_metrics, "config": json_safe(vars(args))}, output / "best.pt")
        print(f"backbone={name} epoch={epoch}/{args.epochs} loss={record['loss']:.4f} select_macro_f1={select_metrics['macro_f1']:.4f}", flush=True)
    if best_record is None:
        raise RuntimeError(f"no checkpoint selected for {name}")
    return best_record


def load_model(name: str, checkpoint: Path, device: torch.device, pretrained: bool) -> nn.Module:
    model = MSENetBase(name, num_classes=5, pretrained=pretrained).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    # LazyLinear is materialized by the checkpoint state when loading.
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def train_fold(fold: int, manifest: pd.DataFrame, run_root: Path, device: torch.device, args: argparse.Namespace) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    fold_root.mkdir(parents=False, exist_ok=False)
    outer_train = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select = make_internal_split(outer_train, fold_seed)
    frames = {"fit": fit, "select": select, "heldout": heldout}
    groups = {key: set(value.group_id.astype(str)) for key, value in frames.items()}
    intersections = {f"{a}_x_{b}": len(groups[a] & groups[b]) for a, b in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))}
    if any(intersections.values()):
        raise RuntimeError(f"fold {fold} has group leakage: {intersections}")
    split_dir = fold_root / "split"
    split_dir.mkdir()
    for key, value in frames.items():
        value.to_csv(split_dir / f"{key}.csv", index=False)
    atomic_json(fold_root / "split_audit.json", {"fold": fold, "counts": {key: len(value) for key, value in frames.items()}, "groups": {key: len(value) for key, value in groups.items()}, "pairwise_group_intersections": intersections, "fit_views": "online affine + flip"})

    model_records = {}
    for name in MSENET_CONFIG["backbones"]:
        model_dir = fold_root / name
        model_dir.mkdir()
        model_records[name] = train_one_backbone(name, fit, select, model_dir, device, args, fold_seed)

    fit_loader = make_loader(fit, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    heldout_loader = make_loader(heldout, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    fit_labels = fit["label"].to_numpy(dtype=np.int64)
    heldout_labels = heldout["label"].to_numpy(dtype=np.int64)
    enhanced = []
    calibration = {}
    individual = {}
    for name in MSENET_CONFIG["backbones"]:
        model = load_model(name, fold_root / name / "best.pt", device, args.pretrained)
        _, fit_truths, fit_probs = evaluate(model, fit_loader, device, return_probabilities=True)
        held_metrics, held_truths, held_probs = evaluate(model, heldout_loader, device, return_probabilities=True)
        changed, calibration[name] = enhance_probabilities(held_probs, fit_probs, fit_truths, MSENET_CONFIG["temperature"])
        enhanced.append(changed)
        individual[name] = held_metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble_probs = np.sum(np.stack(enhanced, axis=0), axis=0)
    ensemble_pred = ensemble_probs.argmax(1)
    result = {
        "method": "MSENet",
        "dataset": "SIPaKMeD",
        "fold": fold,
        "seed": fold_seed,
        "n_fit": len(fit),
        "n_select": len(select),
        "n_heldout": len(heldout),
        "individual_heldout_metrics": individual,
        "ensemble_heldout_metrics": metrics(heldout_labels, ensemble_pred),
        "calibration": calibration,
        "heldout_evaluation_count": 1,
        "class_names": EXPECTED_CLASS_NAMES,
        "predictions_recorded": True,
    }
    atomic_json(fold_root / "result.json", result)
    atomic_json(run_root / "status.json", {"state": "fold_complete", "method": "MSENet", "fold": fold, "result": result["ensemble_heldout_metrics"], "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={result['ensemble_heldout_metrics']['accuracy']:.4f} macro_f1={result['ensemble_heldout_metrics']['macro_f1']:.4f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    args = parser.parse_args()
    manifest = validate_manifest(args.manifest)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    atomic_json(run_root / "status.json", {"state": "starting", "method": "MSENet", "folds": folds, "updated_utc": datetime.now(timezone.utc).isoformat()})
    results = []
    for fold in folds:
        atomic_json(run_root / "status.json", {"state": "training", "method": "MSENet", "fold": fold, "phase": "base_models", "updated_utc": datetime.now(timezone.utc).isoformat()})
        results.append(train_fold(fold, manifest, run_root, device, args))
    frame = pd.DataFrame([row["ensemble_heldout_metrics"] for row in results])
    summary = {
        "method": "MSENet",
        "dataset": "SIPaKMeD",
        "run_name": args.run_name,
        "folds": folds,
        "fold_metrics": [row["ensemble_heldout_metrics"] for row in results],
        "mean_accuracy": float(frame["accuracy"].mean()),
        "mean_macro_f1": float(frame["macro_f1"].mean()),
        "mean_macro_precision": float(frame["macro_precision"].mean()),
        "mean_macro_recall": float(frame["macro_recall"].mean()),
        "sample_sd_accuracy": float(frame["accuracy"].std(ddof=1)),
        "sample_sd_macro_f1": float(frame["macro_f1"].std(ddof=1)),
    }
    atomic_json(run_root / "summary.json", summary)
    pd.DataFrame([{"fold": fold, **row["ensemble_heldout_metrics"]} for fold, row in zip(folds, results)]).to_csv(run_root / "fold_metrics.csv", index=False)
    atomic_json(run_root / "source_fidelity.json", {
        "method": "MSENet",
        "dataset": "SIPaKMeD",
        "paper_doi": "10.1016/j.engappai.2023.106336",
        "official_repository": "https://github.com/rishavpramanik/msenet",
        "implementation_status": "paper-faithful PyTorch adaptation with group-disjoint outer folds and inner selection",
        "paper_reported_sipakmed_accuracy": 0.9721,
        "paper_reported_score_is_not_rerun_result": True,
        "recoverable_mechanisms": ["Xception", "InceptionV3", "VGG16", "three dense layers 512-256-128", "class-wise mean/std probability enhancement", "temperature softmax T=2", "sum aggregation"],
        "explicit_adaptations": ["outer folds and inner fit/select split use the verified source-group-disjoint manifest", "pretrained=False unless --pretrained is explicitly supplied because weights are not cached", "heldout is evaluated once per backbone and ensemble", "online augmentation follows the official rotation/shift/shear/zoom/flip configuration"],
        "run_config": json_safe(vars(args)) | {"device_resolved": str(device), "model_config": MSENET_CONFIG},
    })
    atomic_json(run_root / "status.json", {"state": "completed", "method": "MSENet", "completed_folds": len(results), "summary": summary, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
