"""Train M2-Residual-OOF on difficult boundary samples from OOF analysis."""

import argparse
import json
import sys
import time
import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.tbs.m2_residual_model import build_m2_residual, residual_loss
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M2-Residual-OOF")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--difficult_csv", type=Path, required=True,
                   help="difficult_boundary_samples.csv from OOF generation")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_delta", type=float, default=0.01)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    for n, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(t in str(path).lower() for t in ("test", "calibration")):
            p.error(f"{n} must not contain 'test' or 'calibration'")
    return args


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device,
                difficult_set, lambda_kl, lambda_delta):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"].to(device)
        paths = batch["image_path"]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            corrected, m0_logits, delta_low, delta_high, _ = model(images, return_corrections=True)

            # Identify masks based on OOF difficulty labels
            low_mask = torch.tensor(
                [Path(p).stem in difficult_set.get("low", set()) for p in paths],
                device=device,
            )
            high_mask = torch.tensor(
                [Path(p).stem in difficult_set.get("high", set()) for p in paths],
                device=device,
            )
            diff_mask = low_mask | high_mask
            easy_mask = ~diff_mask

            loss = residual_loss(corrected, m0_logits, delta_low, delta_high, labels,
                                 low_mask, high_mask, easy_mask, lambda_kl, lambda_delta)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"]
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            corrected = model(images)
        probs = torch.softmax(corrected.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return compute_stage1_metrics(labels, probs), probs, labels


def main(argv=None):
    args = parse_args(argv)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load difficult samples
    diff_df = pd.read_csv(args.difficult_csv)
    difficult_set = {
        "low": set(Path(p).stem for p in diff_df[diff_df["difficulty_set"] == "low"]["image_path"]),
        "high": set(Path(p).stem for p in diff_df[diff_df["difficulty_set"] == "high"]["image_path"]),
    }
    print(f"Difficult samples: low={len(difficult_set['low'])} high={len(difficult_set['high'])}", flush=True)

    # Build model
    print("Building M2-Residual from M0 checkpoint...", flush=True)
    model = build_m2_residual(args.m0_checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_params:,}", flush=True)

    # Data
    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = XUDataTBS5Dataset(args.train_csv, eval_tf)
    dev_ds = XUDataTBS5Dataset(args.dev_csv, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, amp_enabled, device,
                                 difficult_set, args.lambda_kl, args.lambda_delta)
        dev, dev_probs, dev_labels = evaluate(model, dev_loader, device, amp_enabled)
        scheduler.step()
        dt = time.time() - t0

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "dev_macro_f1": float(dev["macro_f1"]),
            "dev_accuracy": float(dev["accuracy"]),
            "dev_macro_auc": float(dev["macro_auc"]),
            "dev_nll": float(dev["nll"]), "dev_ece": float(dev["ece"]),
        }
        rows.append(row)
        print(f"Epoch {epoch:2d}/{args.epochs}: loss={train_loss:.4f} "
              f"f1={row['dev_macro_f1']:.4f} auc={row['dev_macro_auc']:.4f} ({dt:.1f}s)", flush=True)

        if row["dev_macro_f1"] > best_f1:
            best_f1 = row["dev_macro_f1"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "metrics": {k: float(v) for k, v in dev.items()},
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }, out_dir / "best_model.pth")

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # Final predictions
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            images = batch["image"].to(device)
            labels = batch["diagnosis_label"]
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                corrected = model(images)
            all_probs.append(torch.softmax(corrected.float(), dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    pdf = pd.DataFrame(probs, columns=[f"prob_{n}" for n in ["Normal","ASC-US","LSIL","ASC-H","HSIL"]])
    pdf["pred_label"] = probs.argmax(axis=1)
    pdf["true_label"] = labels
    pdf.to_csv(out_dir / "dev_predictions.csv", index=False)

    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(labels, probs.argmax(axis=1)):
        cm[t, p] += 1
    pd.DataFrame(cm, index=["N","A-US","L-H","A-H","H-H"],
                 columns=["N","A-US","L-H","A-H","H-H"]).to_csv(out_dir / "confusion_matrix.csv")

    best = rows[int(np.argmax([r["dev_macro_f1"] for r in rows]))]
    (out_dir / "best_metrics.json").write_text(json.dumps(dict(best), indent=2))
    (out_dir / "args.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    (out_dir / "completed.json").write_text(json.dumps({
        "status": "completed", "best_macro_f1": best_f1,
        "calibration_used": False, "test_used": False,
    }, indent=2))

    print(f"\nBest dev Macro F1: {best_f1:.6f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
