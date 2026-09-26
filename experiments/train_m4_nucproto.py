"""Train M4-NucProto with P0-P4 variant support."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from experiments.tbs.m4_nucproto_model import build_m4_nucproto
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M4-NucProto")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--mask_dir_train", type=Path, default=None)
    p.add_argument("--mask_dir_dev", type=Path, default=None)
    p.add_argument("--variant", type=str, default="P3",
                   choices=["P0","P1","P2","P3","P4"],
                   help="P0=M0 baseline P1=fixed centroids P2=learnable global P3=+nucleus P4=+shifted")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
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


class ProtoDataset:
    def __init__(self, manifest_csv, mask_dir, transform, variant):
        self.base = XUDataTBS5Dataset(manifest_csv, transform)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.variant = variant

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        mask_tensor = torch.zeros(224, 224)
        mask_valid = True
        if self.mask_dir and self.variant in ("P3", "P4"):
            sid = Path(item["image_path"]).stem
            mp = self.mask_dir / f"{sid}.png"
            try:
                with Image.open(mp) as m:
                    mi = m.convert("L").resize((224, 224), Image.BILINEAR)
                mask_arr = np.asarray(mi, dtype=np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_arr)
                # P4: shift nucleus mask by 28px to break spatial alignment
                if self.variant == "P4":
                    mask_arr_shifted = np.roll(mask_arr, shift=28, axis=(0, 1))
                    mask_tensor = torch.from_numpy(mask_arr_shifted.copy())
                area = mask_arr.sum()
                mask_valid = (area > 10) and (area < mask_arr.size * 0.95)
            except Exception:
                mask_valid = False
        return {**item, "mask": mask_tensor, "mask_valid": mask_valid, "sample_id": Path(item["image_path"]).stem}


def collate_fn(batch):
    return (
        torch.stack([b["image"] for b in batch]),
        torch.tensor([b["diagnosis_label"] for b in batch], dtype=torch.long),
        torch.stack([b["mask"] for b in batch]),
        torch.tensor([b["mask_valid"] for b in batch], dtype=torch.bool),
    )


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device):
    model.train()
    total_loss, n = 0.0, 0
    for images, labels, masks, valid in loader:
        images, labels = images.to(device), labels.to(device)
        masks, valid = masks.to(device), valid.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _ = model(images, masks, valid)
            loss = F.cross_entropy(logits, labels)
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
    probs, labels = [], []
    for images, lbls, masks, valid in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _ = model(images, masks.to(device), valid.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(lbls.cpu().numpy())
    return compute_stage1_metrics(np.concatenate(labels), np.concatenate(probs)), np.concatenate(probs), np.concatenate(labels)


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

    print(f"=== M4-NucProto {args.variant} ===", flush=True)
    model = build_m4_nucproto(args.m0_checkpoint, device, args.variant)

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = ProtoDataset(args.train_csv, args.mask_dir_train, eval_tf, args.variant)
    dev_ds = ProtoDataset(args.dev_csv, args.mask_dir_dev, eval_tf, args.variant)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Init prototypes from train centroids
    if args.variant != "P0":
        model.init_prototypes_from_train(train_loader, device, amp_enabled)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_trainable:,}", flush=True)

    if args.variant == "P0":
        # P0: no trainable parameters — just evaluate once
        print("  No trainable params — evaluating M0 baseline directly", flush=True)
        dev, probs, labels = evaluate(model, dev_loader, device, amp_enabled)
        best_f1 = float(dev["macro_f1"])
        rows = [{"epoch": 0, "train_loss": 0.0, "dev_macro_f1": best_f1,
                 "dev_accuracy": float(dev["accuracy"]), "dev_macro_auc": float(dev["macro_auc"])}]
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        best_f1 = -1.0
        rows = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scaler, amp_enabled, device)
            dev, probs, labels = evaluate(model, dev_loader, device, amp_enabled)
            scheduler.step()
            dt = time.time() - t0
            row = {"epoch": epoch, "train_loss": train_loss, "dev_macro_f1": float(dev["macro_f1"]),
                   "dev_accuracy": float(dev["accuracy"]), "dev_macro_auc": float(dev["macro_auc"])}
            rows.append(row)
            print(f"Epoch {epoch:2d}: loss={train_loss:.4f} f1={row['dev_macro_f1']:.4f} auc={row['dev_macro_auc']:.4f} ({dt:.1f}s)", flush=True)
            if row["dev_macro_f1"] > best_f1:
                best_f1 = row["dev_macro_f1"]
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "metrics": {k: float(v) for k, v in dev.items()}}, out_dir / "best_model.pth")

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # Final predictions
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, lbls, masks, valid in dev_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, _ = model(images.to(device), masks.to(device), valid.to(device))
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_labels.append(lbls.numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    best_f1 = float(rows[-1]["dev_macro_f1"])
    pdf = pd.DataFrame(probs, columns=["prob_N","prob_AU","prob_L","prob_AH","prob_HH"])
    pdf["pred_label"] = probs.argmax(axis=1)
    pdf["true_label"] = labels_arr
    pdf.to_csv(out_dir / "dev_predictions.csv", index=False)

    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(labels_arr, probs.argmax(axis=1)):
        cm[t, p] += 1
    pd.DataFrame(cm, index=["N","A-U","L","A-H","H"], columns=["N","A-U","L","A-H","H"]).to_csv(out_dir / "confusion_matrix.csv")

    best = rows[int(np.argmax([r["dev_macro_f1"] for r in rows]))]
    (out_dir / "best_metrics.json").write_text(json.dumps(dict(best), indent=2))
    (out_dir / "completed.json").write_text(json.dumps({"status": "completed", "best_f1": best_f1, "variant": args.variant,
                                                         "calibration_used": False, "test_used": False}, indent=2))
    print(f"\n[{args.variant}] Best F1: {best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
