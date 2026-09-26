"""Train M2-Specialist: frozen M0 gate + zero-init adapter + conditional expert.

Abnormal samples only.  No screening loss.
"""

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
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.tbs.m2_specialist_model import (
    build_m2_specialist,
    specialist_loss,
    combine_probs,
)
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M2-Specialist")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_adapter", type=float, default=1e-4)
    p.add_argument("--lr_cond_head", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    for name, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(t in str(path).lower() for t in ("test", "calibration")):
            p.error(f"{name} must not contain 'test' or 'calibration'")
    return args


def param_groups(model, lr_adapter, lr_cond_head, wd):
    groups = {"adapter": [], "cond_head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adapter" in n:
            groups["adapter"].append(p)
        elif "conditional_head" in n:
            groups["cond_head"].append(p)
    result = [
        {"params": groups["adapter"], "lr": lr_adapter, "weight_decay": wd, "name": "adapter"},
        {"params": groups["cond_head"], "lr": lr_cond_head, "weight_decay": wd, "name": "cond_head"},
    ]
    for g in result:
        print(f"  {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.1e}", flush=True)
    return result


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device):
    model.train()
    total_loss = 0.0
    total_abn = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            _, q_logits, a0 = model(images)
            loss = specialist_loss(q_logits, labels, a0)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n_abn = int((labels != 0).sum())
        total_loss += loss.item() * n_abn
        total_abn += n_abn
    return total_loss / max(total_abn, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"]
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            gate_logits, q_logits, a0 = model(images)
            probs = combine_probs(q_logits, a0)
        all_probs.append(probs.cpu().numpy())
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

    print("=== M2-Specialist ===", flush=True)
    print("Building from M0 checkpoint...", flush=True)
    model = build_m2_specialist(args.m0_checkpoint, device)

    # Verify zero-init
    adapter_up_weight = model.adapter.up.weight
    assert torch.allclose(adapter_up_weight, torch.zeros_like(adapter_up_weight)), \
        "Adapter up projection is not zero-initialised"
    print("  Adapter W2 zero-init verified", flush=True)

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = XUDataTBS5Dataset(args.train_csv, eval_tf)
    dev_ds = XUDataTBS5Dataset(args.dev_csv, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pg = param_groups(model, args.lr_adapter, args.lr_cond_head, args.weight_decay)
    optimizer = torch.optim.AdamW(pg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_abn_f1 = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, amp_enabled, device)
        dev, dev_probs, dev_labels = evaluate(model, dev_loader, device, amp_enabled)
        scheduler.step()
        dt = time.time() - t0

        # Abnormal 4-class F1
        abn_mask = dev_labels != 0
        from sklearn.metrics import f1_score
        abn_f1 = f1_score(dev_labels[abn_mask], dev_probs[abn_mask].argmax(axis=1),
                          labels=[1, 2, 3, 4], average="macro", zero_division=0)

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "dev_macro_f1": float(dev["macro_f1"]),
            "dev_abnormal_f1": float(abn_f1),
            "dev_accuracy": float(dev["accuracy"]),
            "dev_macro_auc": float(dev["macro_auc"]),
            "dev_nll": float(dev["nll"]), "dev_ece": float(dev["ece"]),
        }
        rows.append(row)
        print(f"Epoch {epoch:2d}/{args.epochs}: loss={train_loss:.4f} "
              f"5F1={row['dev_macro_f1']:.4f} abnF1={row['dev_abnormal_f1']:.4f} "
              f"auc={row['dev_macro_auc']:.4f} ({dt:.1f}s)", flush=True)

        if abn_f1 > best_abn_f1:
            best_abn_f1 = abn_f1
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "metrics": {k: float(v) for k, v in dev.items()},
                "abnormal_f1": float(abn_f1),
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
                _, q_logits, a0 = model(images)
                probs = combine_probs(q_logits, a0)
            all_probs.append(probs.cpu().numpy())
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

    best = rows[int(np.argmax([r["dev_abnormal_f1"] for r in rows]))]
    (out_dir / "best_metrics.json").write_text(json.dumps(dict(best), indent=2))
    (out_dir / "args.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    (out_dir / "completed.json").write_text(json.dumps({
        "status": "completed", "best_abnormal_f1": best_abn_f1,
        "calibration_used": False, "test_used": False,
    }, indent=2))

    print(f"\nBest abnormal F1: {best_abn_f1:.6f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
