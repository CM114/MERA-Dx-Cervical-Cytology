#!/usr/bin/env python3
"""Grouped five-fold PyTorch reproduction of A2SDNet121 on SIPaKMeD.

The paper expands the training set six-fold with rotations and flips.  This
runner materializes those six deterministic views on the fly, so augmented
copies never cross the source-group split boundaries.
"""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.sipakmed.paper_baselines_v1.a2sdnet121 import A2SDNet121, make_internal_split


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
    class_rows = manifest.sort_values("label").drop_duplicates("label")
    if tuple(class_rows["class_name"].tolist()) != EXPECTED_CLASS_NAMES:
        raise ValueError("SIPaKMeD class-id mapping differs from the verified five-class mapping")
    if set(manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("manifest must contain exactly outer folds 0 through 4")
    missing_files = [p for p in manifest["image_path"].tolist() if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest references missing images; first: {missing_files[:3]}")
    return manifest


class A2SDTrainingDataset(Dataset):
    """SIPaKMeD images with six paper-guided geometric views per source row."""

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
        A2SDTrainingDataset(frame, augment),
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


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, metric: dict) -> None:
    torch.save(
        {
            "method": "A2SDNet121",
            "epoch": epoch,
            "selection_metrics": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def restore_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device | None = None,
) -> tuple[dict, int]:
    """Restore a fold checkpoint and return it with the next epoch to run.

    Checkpoints from the original runner were written immediately before the
    StepLR update.  Advance the scheduler when necessary so a resumed run has
    the same learning-rate boundary as an uninterrupted run.
    """
    load_device = device if device is not None else next(model.parameters()).device
    checkpoint = torch.load(path, map_location=load_device, weights_only=False)
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"checkpoint {path} lacks required keys: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    checkpoint_epoch = int(checkpoint["epoch"])
    if scheduler.last_epoch < checkpoint_epoch:
        scheduler.step()
    return checkpoint, checkpoint_epoch + 1


def should_resume_fold(run_root: Path, fold: int, resume_existing: bool) -> bool:
    """Resume only folds that already have a directory on disk."""
    return bool(resume_existing and (run_root / f"fold_{fold}").is_dir())


def train_fold(
    fold: int,
    manifest: pd.DataFrame,
    run_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    *,
    resume: bool = False,
) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    fold_root = run_root / f"fold_{fold}"
    if resume:
        if not fold_root.is_dir():
            raise FileNotFoundError(f"cannot resume fold {fold}; missing {fold_root}")
        split_dir = fold_root / "split"
        split_paths = {name: split_dir / f"{name}.csv" for name in ("fit", "select", "heldout")}
        missing = [str(path) for path in split_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cannot resume fold {fold}; missing split files: {missing}")
        fit = pd.read_csv(split_paths["fit"])
        select = pd.read_csv(split_paths["select"])
        heldout = pd.read_csv(split_paths["heldout"])
    else:
        fold_root.mkdir(parents=True, exist_ok=False)
        outer_train = manifest.loc[manifest.fold != fold].copy()
        heldout = manifest.loc[manifest.fold == fold].copy()
        fit, select = make_internal_split(outer_train, fold_seed)
        split_dir = fold_root / "split"
        split_dir.mkdir()
        fit.to_csv(split_dir / "fit.csv", index=False)
        select.to_csv(split_dir / "select.csv", index=False)
        heldout.to_csv(split_dir / "heldout.csv", index=False)

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
    if not resume:
        atomic_json(
            fold_root / "split_audit.json",
            {
                "fold": fold,
                "counts": {name: len(frame) for name, frame in (("fit", fit), ("select", select), ("heldout", heldout))},
                "augmented_fit_samples_per_epoch": len(A2SDTrainingDataset(fit, augment=True)),
                "groups": {name: len(values) for name, values in groups.items()},
                "pairwise_group_intersections": intersections,
                "classes": EXPECTED_CLASS_NAMES,
            },
        )
    fit_loader = make_loader(fit, augment=True, batch_size=args.batch_size, workers=args.workers, shuffle=True, seed=fold_seed)
    select_loader = make_loader(select, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)
    heldout_loader = make_loader(heldout, augment=False, batch_size=args.eval_batch_size, workers=args.workers, shuffle=False, seed=fold_seed)

    model = A2SDNet121(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total_epochs = min(args.epochs, args.max_epochs) if args.max_epochs else args.epochs
    best_score = (-1.0, -1.0)
    best_record: dict | None = None
    epoch_log = fold_root / "epochs.jsonl"
    status_path = run_root / "status.json"
    resumed_from_epoch: int | None = None

    if resume:
        checkpoint, start_epoch = restore_training_checkpoint(
            fold_root / "latest.pt", model, optimizer, scheduler, device
        )
        resumed_from_epoch = int(checkpoint["epoch"])
        if resumed_from_epoch >= total_epochs:
            start_epoch = total_epochs + 1
        if epoch_log.is_file():
            records = [json.loads(line) for line in epoch_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if records:
                best_record = max(records, key=lambda item: (item["select"]["macro_f1"], item["select"]["accuracy"]))
                best_score = (best_record["select"]["macro_f1"], best_record["select"]["accuracy"])
        if best_record is None:
            best_record = {
                "fold": fold,
                "epoch": int(checkpoint["epoch"]),
                "select": checkpoint.get("selection_metrics", {"macro_f1": -1.0, "accuracy": -1.0}),
            }
            best_score = (best_record["select"]["macro_f1"], best_record["select"]["accuracy"])
    else:
        start_epoch = 1

    for epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (images, labels) in enumerate(fit_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
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
        scheduler.step()
        if score > best_score:
            best_score = score
            best_record = record
            save_checkpoint(fold_root / "best.pt", model, optimizer, scheduler, epoch, select_metrics)
        save_checkpoint(fold_root / "latest.pt", model, optimizer, scheduler, epoch, select_metrics)
        status_payload = {
            "state": "training",
            "method": "A2SDNet121",
            "fold": fold,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "configured_epochs": args.epochs,
            "latest_select_macro_f1": select_metrics["macro_f1"],
            "latest_select_accuracy": select_metrics["accuracy"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        if resumed_from_epoch is not None:
            status_payload["resumed_from_epoch"] = resumed_from_epoch
        atomic_json(status_path, status_payload)
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
        "method": "A2SDNet121",
        "dataset": "SIPaKMeD",
        "fold": fold,
        "seed": fold_seed,
        "n_fit": len(fit),
        "n_select": len(select),
        "n_heldout": len(heldout),
        "augmented_fit_samples_per_epoch": len(A2SDTrainingDataset(fit, augment=True)),
        "best_epoch": int(best_record["epoch"]),
        "best_selection": best_record["select"],
        "heldout_metrics": heldout_metrics,
        "heldout_evaluation_count": 1,
        "class_names": EXPECTED_CLASS_NAMES,
        "predictions_recorded": len(truths) == len(predictions) == len(heldout),
        "resumed_from_epoch": resumed_from_epoch,
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
            "method": "A2SDNet121",
            "fold": fold,
            "result": heldout_metrics,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del model, optimizer, scheduler, checkpoint
    torch.cuda.empty_cache()
    return result


def write_summary(run_root: Path, results: list[dict]) -> dict:
    rows = [{"fold": result["fold"], **result["heldout_metrics"]} for result in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(run_root / "fold_metrics.csv", index=False)
    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary = {
        "method": "A2SDNet121",
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
    model = A2SDNet121(num_classes=5).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images.to(device))
    nn.CrossEntropyLoss()(logits, labels.to(device)).backward()
    optimizer.step()
    atomic_json(
        smoke_root / "smoke.json",
        {
            "state": "passed",
            "input_batch": list(images.shape),
            "output_shape": list(logits.shape),
            "num_classes": 5,
            "augmentation_variants": AUGMENTATION_VARIANTS,
            "heldout_scored": False,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"SMOKE PASSED: input={tuple(images.shape)} output={tuple(logits.shape)}; heldout not evaluated", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-epochs", type=int, default=0, help="debug only; cap configured epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0, help="debug only; 0 uses all batches")
    parser.add_argument("--smoke", action="store_true", help="run forward/backward plumbing only")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume incomplete folds in an existing run directory from latest.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    manifest = validate_manifest(args.manifest)
    if args.smoke:
        smoke_root = args.results_root / "_smoke_a2sdnet121"
        smoke_root.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_smoke(manifest, device, args, smoke_root)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the full reproduction on CPU")
    if args.run_name is None and args.resume_existing:
        raise ValueError("--resume-existing requires --run-name")
    if args.run_name is None:
        args.run_name = datetime.now(timezone.utc).strftime("a2sdnet121_%Y%m%dT%H%M%SZ")
    run_root = args.results_root / args.run_name
    if args.resume_existing:
        if not run_root.is_dir():
            raise FileNotFoundError(f"cannot resume missing run directory: {run_root}")
        if not (run_root / "source_fidelity.json").is_file():
            raise FileNotFoundError(f"cannot resume without source_fidelity.json: {run_root}")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 maps server GPU 1 to local cuda:0.
    seed_everything(args.seed)
    requested_folds = [int(value) for value in args.folds.split(",")]
    source = {
        "paper_doi": "10.1038/s41598-025-87953-1",
        "official_article": "https://www.nature.com/articles/s41598-025-87953-1",
        "implementation_status": "paper-guided PyTorch reimplementation; no author training repository was available",
        "architecture": "DenseNet121-derived A2SDNet121 with 3x3 stride-1 stem, 2x2 stride-2 max-pool, ADB depths (6,12,16,24), dilation rates (1,2,3) in each block's last three layers, SE after each ADB, three transitions, global average pooling and five-class head",
        "paper_recipe": "224x224 input; six-fold expansion by rotations/flips; SGD lr=1e-4; StepLR gamma=0.1 every 30 epochs; batch size 8; 300 epochs",
        "adaptations": [
            "The paper's six-fold expansion is represented by six deterministic on-the-fly views: identity, 90/180/270-degree rotations, horizontal flip and vertical flip.",
            "The paper does not specify SGD momentum, so PyTorch's default momentum=0.0 is used and recorded.",
            "The paper reports a single split; this rerun uses the user's fixed source-group-disjoint five-fold manifest and an inner group-disjoint selection split.",
            "SE reduction ratio 16, DenseNet bottleneck width 4 and channel compression 0.5 follow the standard DenseNet/SE construction because the article does not state alternatives explicitly.",
            "Images are converted to RGB, resized to 224x224 and scaled to [0,1] without ImageNet normalization, matching the paper's unspecified preprocessing as closely as possible.",
        ],
        "run_config": {
            "manifest": str(args.manifest),
            "outer_folds": [0, 1, 2, 3, 4],
            "requested_folds": requested_folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "workers": args.workers,
            "image_size": IMAGE_SIZE,
            "cuda_visible_device": "1",
        },
    }
    if not args.resume_existing:
        atomic_json(run_root / "source_fidelity.json", source)
        atomic_json(run_root / "status.json", {"state": "starting", "method": "A2SDNet121", "updated_utc": datetime.now(timezone.utc).isoformat()})
    else:
        atomic_json(
            run_root / "status.json",
            {
                "state": "resuming",
                "method": "A2SDNet121",
                "run_name": args.run_name,
                "requested_folds": requested_folds,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"DEVICE={torch.cuda.get_device_name(device)}; MANIFEST={args.manifest}; rows={len(manifest)}", flush=True)
    results: list[dict] = []
    for fold in requested_folds:
        result_path = run_root / f"fold_{fold}" / "result.json"
        if args.resume_existing and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            print(f"FOLD ALREADY COMPLETE fold={fold}; loaded {result_path}", flush=True)
            continue
        resume_fold = should_resume_fold(run_root, fold, args.resume_existing)
        results.append(train_fold(fold, manifest, run_root, device, args, resume=resume_fold))
        summary = write_summary(run_root, results)
        print(f"SUMMARY completed={summary['completed_folds']} folds mean={summary['metrics_mean']}", flush=True)
    atomic_json(
        run_root / "status.json",
        {"state": "complete", "method": "A2SDNet121", "completed_folds": len(results), "updated_utc": datetime.now(timezone.utc).isoformat()},
    )
    print("ALL REQUESTED FOLDS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
