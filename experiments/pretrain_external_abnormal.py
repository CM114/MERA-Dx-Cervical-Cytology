"""Train a four-class abnormal-only auxiliary backbone."""

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
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.dataset import build_transforms  # noqa: E402
from experiments.tbs.external_dataset import ExternalAbnormalDataset  # noqa: E402
from experiments.tbs.models import build_external_abnormal_model  # noqa: E402


LABEL_NAMES = ("ASC-US", "LSIL", "ASC-H", "HSIL")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _prepare_output(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing pretraining output: {out_dir}")
    else:
        out_dir.mkdir(parents=True)
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--input_mode", choices=("crop", "letterbox"), default="letterbox")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser


def _environment(args):
    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = "unavailable"
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm_version,
        "cuda_available": torch.cuda.is_available(),
        "train_manifest_sha256": _sha256(args.train_csv),
        "dev_manifest_sha256": _sha256(args.dev_csv),
        "raw_data_modified": False,
        "target_heldout_opened": False,
        "target_calibration_opened": False,
        "target_sealed_test_opened": False,
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
    return payload


def _run_epoch(model, loader, optimizer, scaler, device, amp_enabled, train):
    model.train(train)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    y_true = []
    y_pred = []
    image_paths = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)["logits"]
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
            image_paths.extend(batch["image_path"])
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(4)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "image_paths": image_paths,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for external abnormal pretraining")
    out_dir = _prepare_output(args.out_dir)
    seed_everything(args.seed)
    train_transform, eval_transform = build_transforms(args.img_size, input_mode=args.input_mode)
    train_dataset = ExternalAbnormalDataset(args.train_csv, train_transform)
    dev_dataset = ExternalAbnormalDataset(args.dev_csv, eval_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, worker_init_fn=_seed_worker,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, worker_init_fn=_seed_worker,
    )
    model = build_external_abnormal_model(args.model_name, pretrained=args.pretrained).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    resolved_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(resolved_args, indent=2) + "\n", encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(_environment(args), indent=2) + "\n", encoding="utf-8")
    (out_dir / "safety.json").write_text(json.dumps({
        "schema_version": "external-abnormal-pretraining-safety-v1",
        "raw_data_modified": False,
        "target_heldout_opened": False,
        "target_calibration_opened": False,
        "target_sealed_test_opened": False,
        "purpose": "four-class auxiliary backbone pretraining only",
    }, indent=2) + "\n", encoding="utf-8")

    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, scaler, torch.device("cuda:0"), not args.no_amp, True)
        dev_metrics = _run_epoch(model, dev_loader, optimizer, scaler, torch.device("cuda:0"), not args.no_amp, False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "dev_loss": dev_metrics["loss"],
            "macro_f1": dev_metrics["macro_f1"],
            "balanced_accuracy": dev_metrics["balanced_accuracy"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={row['train_loss']:.4f} | "
            f"dev_loss={row['dev_loss']:.4f} | macro_f1={row['macro_f1']:.4f} | "
            f"balanced_acc={row['balanced_accuracy']:.4f}"
        )
        if row["macro_f1"] > best:
            best = row["macro_f1"]
            torch.save({"model_state": model.state_dict(), "model_name": args.model_name, "epoch": epoch, "args": resolved_args}, out_dir / "best_model.pth")
            (out_dir / "best_metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            prediction_frame = pd.DataFrame({
                "image_path": dev_metrics["image_paths"],
                "true_label": dev_metrics["y_true"],
                "pred_label": dev_metrics["y_pred"],
            })
            prediction_frame.to_csv(out_dir / "dev_predictions.csv", index=False)
    torch.save({"model_state": model.state_dict(), "model_name": args.model_name, "epoch": args.epochs, "args": resolved_args}, out_dir / "last_model.pth")
    print(f"Best external dev Macro F1: {best:.6f}")
    print(f"Results: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
