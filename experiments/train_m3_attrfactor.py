"""Train M3-AttrFactor with spatial nucleus pooling + weak morphology attributes."""

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

from experiments.tbs.m3_attrfactor_model import build_attrfactor
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


MORPH_ATTRS = ["nucleus_area_px", "nucleus_area_frac", "boundary_gradient_mean", "nucleus_r_mean"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M3-AttrFactor")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--mask_dir_train", type=Path, required=True)
    p.add_argument("--mask_dir_dev", type=Path, required=True)
    p.add_argument("--train_features", type=Path, default=None,
                   help="train_nucleus_features.csv for morphology weak supervision")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=5e-6)
    p.add_argument("--lr_heads", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_morph", type=float, default=0.1)
    p.add_argument("--lambda_preserve", type=float, default=0.1)
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


def _fit_morph_scaler(features_csv):
    """Fit per-attribute mean/std from train features CSV. Returns (mean, std)."""
    feats = pd.read_csv(features_csv)
    raw = feats[MORPH_ATTRS].values.astype(np.float64)
    raw[np.isnan(raw)] = 0.0
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


class AttrDataset:
    def __init__(self, manifest_csv, mask_dir, transform, features_csv=None, morph_mean=None, morph_std=None):
        self.base = XUDataTBS5Dataset(manifest_csv, transform)
        self.mask_dir = Path(mask_dir)
        self.morph_targets = {}
        if features_csv is not None and morph_mean is not None and morph_std is not None:
            feats = pd.read_csv(features_csv)
            for _, row in feats.iterrows():
                sid = row["sample_id"]
                vals = []
                for j, attr in enumerate(MORPH_ATTRS):
                    v = row.get(attr, 0.0)
                    if pd.isna(v):
                        v = 0.0
                    vals.append((float(v) - float(morph_mean[j])) / float(morph_std[j]))
                self.morph_targets[sid] = np.array(vals, dtype=np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        sid = Path(item["image_path"]).stem
        mask_path = self.mask_dir / f"{sid}.png"
        try:
            with Image.open(mask_path) as m:
                mask_img = m.convert("L").resize((224, 224), Image.BILINEAR)
            mask_tensor = torch.from_numpy(np.asarray(mask_img, dtype=np.float32) / 255.0)
        except Exception:
            mask_tensor = torch.zeros(224, 224)
        morph_target = self.morph_targets.get(sid, np.zeros(len(MORPH_ATTRS), dtype=np.float32))
        return {
            **item,
            "mask": mask_tensor,
            "morph_target": torch.from_numpy(morph_target),
            "sample_id": sid,
        }


def collate_fn(batch):
    return (
        torch.stack([b["image"] for b in batch]),
        torch.tensor([b["diagnosis_label"] for b in batch], dtype=torch.long),
        torch.stack([b["mask"] for b in batch]),
        torch.stack([b["morph_target"] for b in batch]),
    )


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device,
                lambda_morph, lambda_preserve):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels, masks, morph_targets in loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        morph_targets = morph_targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            probs, morph_pred, m, e_low, e_high, m0_probs = model(images, masks)
            # Diagnostic CE
            log_probs = torch.log(probs.clamp(1e-12, 1.0))
            ce = F.nll_loss(log_probs, labels)

            loss = ce

            # Morphology weak supervision
            if lambda_morph > 0 and morph_pred is not None:
                morph_loss = F.mse_loss(morph_pred, morph_targets)
                loss = loss + lambda_morph * morph_loss

            # KL preservation to M0
            if lambda_preserve > 0:
                kl = (m0_probs * (torch.log(m0_probs + 1e-12) - torch.log(probs + 1e-12))).sum(dim=1).mean()
                loss = loss + lambda_preserve * kl

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
    for images, labels, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            probs, _, _, _, _, _ = model(images, masks)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return compute_stage1_metrics(labels, probs), probs, labels


def build_param_groups(model, lr_backbone, lr_heads, wd):
    bb, heads = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in n:
            bb.append(p)
        else:
            heads.append(p)
    groups = [
        {"params": bb, "lr": lr_backbone, "weight_decay": wd, "name": "backbone"},
        {"params": heads, "lr": lr_heads, "weight_decay": wd, "name": "heads"},
    ]
    for g in groups:
        print(f"  {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.1e}", flush=True)
    return groups


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

    print("=== M3-AttrFactor ===", flush=True)
    model = build_attrfactor(args.m0_checkpoint, device)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable: {total_trainable:,}", flush=True)

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    morph_mean, morph_std = None, None
    if args.train_features is not None:
        morph_mean, morph_std = _fit_morph_scaler(args.train_features)
        print(f"  Morph mean: {morph_mean}", flush=True)
        print(f"  Morph std:  {morph_std}", flush=True)
    train_ds = AttrDataset(args.train_csv, args.mask_dir_train, eval_tf,
                           args.train_features, morph_mean, morph_std)
    dev_ds = AttrDataset(args.dev_csv, args.mask_dir_dev, eval_tf)  # no morph targets needed for dev

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    pg = build_param_groups(model, args.lr_backbone, args.lr_heads, args.weight_decay)
    optimizer = torch.optim.AdamW(pg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, amp_enabled, device,
                                 args.lambda_morph, args.lambda_preserve)
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
            }, out_dir / "best_model.pth")

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # Final predictions
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, masks, _ in dev_loader:
            images = images.to(device)
            masks = masks.to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                probs, _, _, _, _, _ = model(images, masks)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    pdf = pd.DataFrame(probs, columns=[f"prob_{n}" for n in ["N","A-US","L-H","A-H","H-H"]])
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
    (out_dir / "completed.json").write_text(json.dumps({
        "status": "completed", "best_macro_f1": best_f1,
        "calibration_used": False, "test_used": False,
    }, indent=2))

    print(f"\nBest dev F1: {best_f1:.6f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
