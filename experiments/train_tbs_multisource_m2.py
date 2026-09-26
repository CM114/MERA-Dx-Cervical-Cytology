"""Train M2 on normalClass plus xudata while evaluating target dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms  # noqa: E402
from experiments.tbs.losses import compute_stage1_loss  # noqa: E402
from experiments.tbs.losses import validate_label_smoothing  # noqa: E402
from experiments.tbs.metrics import compute_stage1_metrics, save_evaluation_artifacts  # noqa: E402
from experiments.tbs.models import build_stage1_model  # noqa: E402
from experiments.tbs.multisource import (  # noqa: E402
    build_source_schedule,
    manifest_has_verified_case_identity,
    summarize_source_csv,
    validate_tbs5_manifest,
)
from experiments.tbs.optimization import build_discriminative_parameter_groups  # noqa: E402
from experiments.tbs.sampling import build_case_class_balanced_weights  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_train_csv", type=Path, required=True)
    parser.add_argument("--target_dev_csv", type=Path, required=True)
    parser.add_argument("--supplement_train_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--variant", choices=("m2",), default="m2")
    parser.add_argument("--target_fraction", type=float, default=0.7)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--input_mode", choices=("crop", "letterbox"), default="letterbox")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_strategy", choices=("uniform", "case_class_balanced"), default="case_class_balanced")
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_fraction < 1.0:
        parser.error("target_fraction must be strictly between 0 and 1")
    try:
        args.label_smoothing = validate_label_smoothing(args.label_smoothing)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs and batch_size must be positive; num_workers must be nonnegative")
    return args


def _worker_seed(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, args, shuffle, case_balanced=False):
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if shuffle and case_balanced:
        sampler = WeightedRandomSampler(
            build_case_class_balanced_weights(dataset.records),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _train_batch(model, batch, optimizer, scaler, device, amp, label_smoothing):
    images = batch["image"].to(device, non_blocking=True)
    diagnosis = batch["diagnosis_label"].to(device, non_blocking=True)
    screen = batch["screen_label"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp):
        output = model(images)
        losses = compute_stage1_loss(
            output,
            diagnosis,
            screen,
            "m2",
            lambda_screen=0.0,
            label_smoothing=label_smoothing,
        )
    if not torch.isfinite(losses["loss"]).all():
        raise FloatingPointError("nonfinite multi-source training loss")
    scaler.scale(losses["loss"]).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(losses["loss"].detach().cpu()), int(images.shape[0])


@torch.no_grad()
def evaluate_target(model, loader, device, amp):
    model.eval()
    labels, probabilities, paths = [], [], []
    maturity_labels, maturity_names = [], []
    losses = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        diagnosis = batch["diagnosis_label"].to(device, non_blocking=True)
        screen = batch["screen_label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)
            batch_losses = compute_stage1_loss(
                output,
                diagnosis,
                screen,
                "m2",
                lambda_screen=0.0,
                label_smoothing=0.0,
            )
        losses.append((float(batch_losses["loss"].cpu()), int(images.shape[0])))
        labels.append(diagnosis.cpu().numpy())
        probabilities.append(output["diagnosis_probs"].cpu().numpy())
        paths.extend(batch["image_path"])
        maturity_labels.extend(batch["maturity_label"].cpu().numpy().tolist())
        maturity_names.extend(batch["maturity_name"])
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    metrics = compute_stage1_metrics(y_true, y_prob)
    metrics["val_loss"] = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    return {"metrics": metrics, "y_true": y_true, "y_prob": y_prob, "paths": paths, "maturity_labels": maturity_labels, "maturity_names": maturity_names}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-source training")
    for path in (args.target_train_csv, args.target_dev_csv, args.supplement_train_csv):
        validate_tbs5_manifest(path)
    if not manifest_has_verified_case_identity(args.target_train_csv, source="target"):
        raise ValueError("target train manifest must contain non-empty case_key values")
    if not manifest_has_verified_case_identity(args.target_dev_csv, source="target"):
        raise ValueError("target dev manifest must contain non-empty case_key values")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    seed_everything(args.seed)

    train_transform, eval_transform = build_transforms(args.img_size, args.input_mode)
    target_train = XUDataTBS5Dataset(args.target_train_csv, train_transform)
    supplement_train = XUDataTBS5Dataset(args.supplement_train_csv, train_transform)
    target_dev = XUDataTBS5Dataset(args.target_dev_csv, eval_transform)
    if args.sampling_strategy == "case_class_balanced" and not all(str(r.get("case_key", "")).strip() for r in target_train.records):
        raise ValueError("target train manifest needs non-empty case_key for case-balanced sampling")

    target_loader = _loader(target_train, args, True, args.sampling_strategy == "case_class_balanced")
    supplement_loader = _loader(supplement_train, args, True, False)
    dev_loader = _loader(target_dev, args, False, False)
    # One target-loader pass defines an epoch; supplement batches fill the
    # remaining source fraction without repeatedly replaying the target set.
    steps_per_epoch = max(
        len(target_loader),
        int(math.ceil(len(target_loader) / args.target_fraction)),
    )
    schedule = build_source_schedule(steps_per_epoch, args.target_fraction)

    source_summary = {
        "schema_version": "xudata-replacement-multisource-m2-v1",
        "target_fraction_requested": args.target_fraction,
        "target_train": summarize_source_csv(args.target_train_csv, "target"),
        "target_dev": summarize_source_csv(args.target_dev_csv, "target_dev"),
        "supplement_train": summarize_source_csv(args.supplement_train_csv, "supplement"),
        "schedule_steps": len(schedule),
        "schedule_target_steps": schedule.count("target"),
        "schedule_supplement_steps": schedule.count("supplement"),
        "target_loader_batches": len(target_loader),
        "supplement_loader_batches": len(supplement_loader),
        "steps_per_epoch": steps_per_epoch,
        "evaluation_policy": "target_dev_only; supplement is never evaluated or case-aggregated",
    }
    (args.out_dir / "source_mix_summary.json").write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    resolved["target_train_sha256"] = file_sha256(args.target_train_csv)
    resolved["target_dev_sha256"] = file_sha256(args.target_dev_csv)
    resolved["supplement_train_sha256"] = file_sha256(args.supplement_train_csv)
    (args.out_dir / "args.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    try:
        import timm

        timm_version = timm.__version__
    except Exception:
        timm_version = "unavailable"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "target_train_manifest_sha256": resolved["target_train_sha256"],
        "target_dev_manifest_sha256": resolved["target_dev_sha256"],
        "supplement_train_manifest_sha256": resolved["supplement_train_sha256"],
        "evaluation_policy": "target_dev_only",
    }
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    model = build_stage1_model("m2", model_name=args.model_name, pretrained=args.pretrained).cuda()
    optimizer = torch.optim.AdamW(build_discriminative_parameter_groups(model, args.lr, args.backbone_lr_multiplier), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals = {"target": [0.0, 0], "supplement": [0.0, 0]}
        target_iter = iter(target_loader)
        supplement_iter = iter(supplement_loader)
        for source in schedule:
            if source == "target":
                try:
                    batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    batch = next(target_iter)
            else:
                try:
                    batch = next(supplement_iter)
                except StopIteration:
                    supplement_iter = iter(supplement_loader)
                    batch = next(supplement_iter)
            value, count = _train_batch(
                model,
                batch,
                optimizer,
                scaler,
                torch.device("cuda:0"),
                not args.no_amp,
                args.label_smoothing,
            )
            train_totals[source][0] += value * count
            train_totals[source][1] += count
        evaluation = evaluate_target(model, dev_loader, torch.device("cuda:0"), not args.no_amp)
        metrics = evaluation["metrics"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "target_steps": schedule.count("target"),
            "supplement_steps": schedule.count("supplement"),
            "train_target_loss": train_totals["target"][0]
            / max(train_totals["target"][1], 1),
            "train_supplement_loss": train_totals["supplement"][0]
            / max(train_totals["supplement"][1], 1),
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / "metrics.csv", index=False)
        print(f"Epoch {epoch:03d}/{args.epochs} | target_loss={row['train_target_loss']:.4f} | supplement_loss={row['train_supplement_loss']:.4f} | val_loss={metrics['val_loss']:.4f} | macro_f1={metrics['macro_f1']:.4f} | balanced_acc={metrics['balanced_accuracy']:.4f} | macro_auc={metrics['macro_auc']:.4f}")
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"epoch": epoch, "variant": "m2", "model_name": args.model_name, "model_state": model.state_dict(), "metrics": metrics, "args": resolved}, args.out_dir / "best_model.pth")
            (args.out_dir / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2) + "\n", encoding="utf-8")
            save_evaluation_artifacts(args.out_dir, "dev", evaluation["y_true"], evaluation["y_prob"], evaluation["paths"], evaluation["maturity_labels"], evaluation["maturity_names"])
    torch.save(
        {
            "epoch": args.epochs,
            "variant": "m2",
            "model_name": args.model_name,
            "model_state": model.state_dict(),
            "args": resolved,
        },
        args.out_dir / "last_model.pth",
    )
    print(f"\nBest target-dev Macro F1: {best:.6f}")
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
