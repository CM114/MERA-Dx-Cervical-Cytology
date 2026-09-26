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
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms  # noqa: E402
from experiments.tbs.losses import compute_m3_loss  # noqa: E402
from experiments.tbs.metrics import (  # noqa: E402
    compute_semantic_metrics,
    compute_stage1_metrics,
    save_evaluation_artifacts,
)
from experiments.tbs.models import build_tbs_semantic_model  # noqa: E402


LOSS_NAMES = (
    "loss",
    "diagnosis_loss",
    "screen_loss",
    "morph_loss",
    "evidence_loss",
    "decorr_loss",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")
    return config


def _validated_nonnegative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, required=True)
    known, _ = pre_parser.parse_known_args()
    config = _load_config(known.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--experiment_name", default=config.get("experiment_name"))
    parser.add_argument("--model_name", default=config.get("model_name", "caformer_s18"))
    parser.add_argument("--train_csv", type=Path, default=Path(config["train_csv"]))
    parser.add_argument("--dev_csv", type=Path, default=Path(config["dev_csv"]))
    parser.add_argument("--out_dir", type=Path, default=Path(config["out_dir"]))
    parser.add_argument("--img_size", type=int, default=int(config.get("img_size", 224)))
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 30)))
    parser.add_argument("--batch_size", type=int, default=int(config.get("batch_size", 64)))
    parser.add_argument("--lr", type=float, default=float(config.get("lr", 1e-4)))
    parser.add_argument(
        "--weight_decay", type=float, default=float(config.get("weight_decay", 1e-4))
    )
    parser.add_argument("--num_workers", type=int, default=int(config.get("num_workers", 8)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 42)))
    parser.add_argument(
        "--semantic_dim", type=int, default=int(config.get("semantic_dim", 128))
    )
    parser.add_argument(
        "--lambda_morph", type=float, default=float(config.get("lambda_morph", 0.3))
    )
    parser.add_argument(
        "--lambda_evidence",
        type=float,
        default=float(config.get("lambda_evidence", 0.3)),
    )
    parser.add_argument(
        "--lambda_decorr", type=float, default=float(config.get("lambda_decorr", 0.01))
    )

    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=bool(config.get("pretrained", True)))

    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=bool(config.get("amp", True)))

    args = parser.parse_args()
    if args.semantic_dim <= 0:
        parser.error("semantic_dim must be positive")
    try:
        args.lambda_morph = _validated_nonnegative(
            "lambda_morph", args.lambda_morph
        )
        args.lambda_evidence = _validated_nonnegative(
            "lambda_evidence", args.lambda_evidence
        )
        args.lambda_decorr = _validated_nonnegative(
            "lambda_decorr", args.lambda_decorr
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _environment_payload(args):
    try:
        import timm

        timm_version = timm.__version__
    except Exception:
        timm_version = "unavailable"
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "timm": timm_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "train_manifest_sha256": file_sha256(args.train_csv),
        "dev_manifest_sha256": file_sha256(args.dev_csv),
    }
    schema_path = args.train_csv.parent / "label_schema_tbs5.json"
    if schema_path.is_file():
        payload["label_schema_sha256"] = file_sha256(schema_path)
    if torch.cuda.is_available():
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch_arch_list": torch.cuda.get_arch_list(),
            }
        )
    return payload


def _make_loader(dataset, args, shuffle):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _batch_targets(batch, device):
    return {
        "diagnosis_labels": batch["diagnosis_label"].to(
            device, non_blocking=True
        ),
        "screen_labels": batch["screen_label"].to(device, non_blocking=True),
        "morph_labels": batch["morph_label"].to(device, non_blocking=True),
        "evidence_labels": batch["evidence_label"].to(
            device, non_blocking=True
        ),
        "semantic_mask": batch["semantic_mask"].to(
            device, non_blocking=True
        ).bool(),
    }


def _compute_losses(output, targets, args):
    return compute_m3_loss(
        output,
        targets["diagnosis_labels"],
        targets["screen_labels"],
        targets["morph_labels"],
        targets["evidence_labels"],
        targets["semantic_mask"],
        lambda_morph=args.lambda_morph,
        lambda_evidence=args.lambda_evidence,
        lambda_decorr=args.lambda_decorr,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    semantic_count = 0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _batch_targets(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            losses = _compute_losses(output, targets, args)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        abnormal_count = int(targets["semantic_mask"].sum().item())
        sample_count += batch_size
        semantic_count += abnormal_count
        batch_count += 1
        for name in ("loss", "diagnosis_loss", "screen_loss"):
            totals[name] += float(losses[name].detach().item()) * batch_size
        for name in ("morph_loss", "evidence_loss"):
            totals[name] += float(losses[name].detach().item()) * abnormal_count
        totals["decorr_loss"] += float(losses["decorr_loss"].detach().item())

    return {
        "loss": totals["loss"] / sample_count,
        "diagnosis_loss": totals["diagnosis_loss"] / sample_count,
        "screen_loss": totals["screen_loss"] / sample_count,
        "morph_loss": totals["morph_loss"] / semantic_count,
        "evidence_loss": totals["evidence_loss"] / semantic_count,
        "decorr_loss": totals["decorr_loss"] / batch_count,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    semantic_count = 0
    batch_count = 0
    labels = []
    probabilities = []
    morph_labels = []
    morph_high_probabilities = []
    evidence_labels = []
    evidence_definitive_probabilities = []
    semantic_masks = []
    image_paths = []
    maturity_labels = []
    maturity_names = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _batch_targets(batch, device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            losses = _compute_losses(output, targets, args)

        batch_size = images.shape[0]
        abnormal_count = int(targets["semantic_mask"].sum().item())
        sample_count += batch_size
        semantic_count += abnormal_count
        batch_count += 1
        for name in ("loss", "diagnosis_loss", "screen_loss"):
            totals[name] += float(losses[name].detach().item()) * batch_size
        for name in ("morph_loss", "evidence_loss"):
            totals[name] += float(losses[name].detach().item()) * abnormal_count
        totals["decorr_loss"] += float(losses["decorr_loss"].detach().item())

        labels.append(targets["diagnosis_labels"].cpu().numpy())
        probabilities.append(output["diagnosis_probs"].cpu().numpy())
        morph_labels.append(targets["morph_labels"].cpu().numpy())
        morph_high_probabilities.append(output["morph_probs"][:, 1].cpu().numpy())
        evidence_labels.append(targets["evidence_labels"].cpu().numpy())
        evidence_definitive_probabilities.append(
            output["evidence_probs"][:, 1].cpu().numpy()
        )
        semantic_masks.append(targets["semantic_mask"].cpu().numpy())
        image_paths.extend(batch["image_path"])
        maturity_labels.extend(batch["maturity_label"].cpu().numpy().tolist())
        maturity_names.extend(batch["maturity_name"])

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    morph_true = np.concatenate(morph_labels)
    morph_high_prob = np.concatenate(morph_high_probabilities)
    evidence_true = np.concatenate(evidence_labels)
    evidence_definitive_prob = np.concatenate(
        evidence_definitive_probabilities
    )
    semantic_mask = np.concatenate(semantic_masks).astype(bool)

    metrics = compute_stage1_metrics(y_true, y_prob)
    metrics.update(
        compute_semantic_metrics(
            morph_true,
            morph_high_prob,
            evidence_true,
            evidence_definitive_prob,
            semantic_mask,
            y_prob,
        )
    )
    metrics.update(
        {
            "val_loss": totals["loss"] / sample_count,
            "val_diagnosis_loss": totals["diagnosis_loss"] / sample_count,
            "val_screen_loss": totals["screen_loss"] / sample_count,
            "val_morph_loss": totals["morph_loss"] / semantic_count,
            "val_evidence_loss": totals["evidence_loss"] / semantic_count,
            "val_decorr_loss": totals["decorr_loss"] / batch_count,
        }
    )
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "morph_true": morph_true,
        "morph_high_prob": morph_high_prob,
        "evidence_true": evidence_true,
        "evidence_definitive_prob": evidence_definitive_prob,
        "semantic_mask": semantic_mask,
        "image_paths": image_paths,
        "maturity_labels": maturity_labels,
        "maturity_names": maturity_names,
    }


def _print_dataset_summary(name, csv_path):
    frame = pd.read_csv(csv_path)
    print(f"\n{name}: {len(frame)} images")
    print(
        frame.groupby(["diagnosis_label", "diagnosis_name"], sort=True)
        .size()
        .to_string()
    )
    abnormal = frame.loc[frame["semantic_mask"].astype(int) == 1]
    print("\nTBS semantic combinations (abnormal only):")
    print(
        abnormal.groupby(["morph_label", "evidence_label"], sort=True)
        .size()
        .to_string()
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M3 training")

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (args.out_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_args, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    environment = _environment_payload(args)
    with (args.out_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(environment, indent=2, ensure_ascii=False))

    _print_dataset_summary("train", args.train_csv)
    _print_dataset_summary("dev", args.dev_csv)

    train_transform, eval_transform = build_transforms(args.img_size)
    train_dataset = XUDataTBS5Dataset(args.train_csv, train_transform)
    dev_dataset = XUDataTBS5Dataset(args.dev_csv, eval_transform)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    dev_loader = _make_loader(dev_dataset, args, shuffle=False)

    device = torch.device("cuda:0")
    model = build_tbs_semantic_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        semantic_dim=args.semantic_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = []
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        evaluation = evaluate(model, dev_loader, device, args)
        dev_metrics = evaluation["metrics"]
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "lr": learning_rate,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **dev_metrics,
        }
        history.append(epoch_metrics)
        pd.DataFrame(history).to_csv(args.out_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={dev_metrics['val_loss']:.4f} | "
            f"macro_f1={dev_metrics['macro_f1']:.4f} | "
            f"macro_auc={dev_metrics['macro_auc']:.4f} | "
            f"morph_acc={dev_metrics['morph_accuracy']:.4f} | "
            f"evidence_acc={dev_metrics['evidence_accuracy']:.4f} | "
            f"screen_sens={dev_metrics['screen_sensitivity']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            checkpoint = {
                "epoch": epoch,
                "variant": "m3",
                "model_name": args.model_name,
                "model_state": model.state_dict(),
                "metrics": dev_metrics,
                "args": resolved_args,
            }
            torch.save(checkpoint, args.out_dir / "best_model.pth")
            best_payload = {"epoch": epoch, **dev_metrics}
            with (args.out_dir / "best_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(best_payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            save_evaluation_artifacts(
                args.out_dir,
                "dev",
                evaluation["y_true"],
                evaluation["y_prob"],
                evaluation["image_paths"],
                evaluation["maturity_labels"],
                evaluation["maturity_names"],
                morph_labels=evaluation["morph_true"],
                morph_high_prob=evaluation["morph_high_prob"],
                evidence_labels=evaluation["evidence_true"],
                evidence_definitive_prob=evaluation[
                    "evidence_definitive_prob"
                ],
                semantic_mask=evaluation["semantic_mask"],
            )

    torch.save(
        {
            "epoch": args.epochs,
            "variant": "m3",
            "model_name": args.model_name,
            "model_state": model.state_dict(),
            "args": resolved_args,
        },
        args.out_dir / "last_model.pth",
    )
    print(f"\nBest dev Macro F1: {best_macro_f1:.6f}")
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
