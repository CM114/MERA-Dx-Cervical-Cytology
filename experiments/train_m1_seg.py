"""Train M1-Seg-Joint with gradient-connectivity pre-flight and C0/C1 control.

Usage:
  C0 (control): --lambda_seg 0
  C1 (aux):     --lambda_seg 0.05
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
from PIL import Image

from experiments.tbs.m1_seg_model import (
    build_m1_seg_joint,
    seg_loss_weighted,
    compute_quality_weights,
)
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M1-Seg-Joint")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--mask_dir_train", type=Path, required=True)
    p.add_argument("--mask_dir_dev", type=Path, required=True)
    p.add_argument("--train_features", type=Path, default=None,
                   help="train_nucleus_features.csv for quality weights")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_stage2", type=float, default=2e-6)
    p.add_argument("--lr_stage34", type=float, default=5e-6)
    p.add_argument("--lr_cls_head", type=float, default=1e-5)
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_seg", type=float, default=0.05,
                   help="0.0 for C0 control, 0.05 for C1 joint")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seg_size", type=int, default=56)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_gradient_test", action="store_true",
                   help="Skip pre-flight gradient connectivity check")
    args = p.parse_args(argv)
    for name, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(t in str(path).lower() for t in ("test", "calibration")):
            p.error(f"{name} must not contain 'test' or 'calibration'")
    return args


class SegDataset:
    def __init__(self, manifest_csv, mask_dir, transform, seg_size=56, features_csv=None):
        self.base = XUDataTBS5Dataset(manifest_csv, transform)
        self.mask_dir = Path(mask_dir)
        self.seg_size = seg_size
        self.quality_map = {}
        if features_csv is not None:
            feats = pd.read_csv(features_csv)
            for _, row in feats.iterrows():
                sid = row["sample_id"]
                q = 1.0
                if not row.get("nonempty", True):
                    q = 0.0
                elif row.get("foreground_fraction", 0.05) > 0.90:
                    q = 0.0
                elif row.get("edge_contact", False):
                    q = 0.5
                elif int(row.get("component_count", 1)) > 3:
                    q = 0.5
                self.quality_map[sid] = q

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        sid = Path(item["image_path"]).stem
        mask_path = self.mask_dir / f"{sid}.png"
        try:
            with Image.open(mask_path) as m:
                mask_img = m.convert("L").resize((self.seg_size, self.seg_size), Image.BILINEAR)
            mask_tensor = torch.from_numpy(np.asarray(mask_img, dtype=np.float32) / 255.0)
        except Exception:
            mask_tensor = torch.zeros(self.seg_size, self.seg_size)
        return {
            **item,
            "mask": mask_tensor,
            "quality": self.quality_map.get(sid, 1.0),
            "nonempty": bool(mask_tensor.sum() > 10),
            "edge_contact": False,
            "component_count": 1,
            "sample_id": sid,
        }


def collate_fn(batch):
    return (
        torch.stack([b["image"] for b in batch]),
        torch.tensor([b["diagnosis_label"] for b in batch], dtype=torch.long),
        torch.stack([b["mask"] for b in batch]),
        torch.tensor([b["quality"] for b in batch], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Gradient connectivity test
# ---------------------------------------------------------------------------


def gradient_connectivity_test(model, device):
    """Verify seg loss gradients reach stage2, not stage3/4/classifier."""
    import math
    print("=== Gradient connectivity test ===", flush=True)
    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    dummy_mask = torch.rand(2, 56, 56, device=device)
    model.train()

    # Test 1: seg loss only
    model.zero_grad(set_to_none=True)
    _, mask_pred = model(dummy_img)
    seg_loss = seg_loss_weighted(mask_pred, dummy_mask)
    seg_loss.backward()

    results = {}
    for scope, prefix in [("stage2", "stages.2"), ("stage3", "stages.3"),
                           ("stage4", "stages.4"), ("cls_head", "diagnosis_head"),
                           ("decoder", "seg_decoder")]:
        grad_norm = 0.0
        for n, p in model.named_parameters():
            if prefix in n and p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        results[f"seg→{scope}"] = math.sqrt(grad_norm) if grad_norm > 0 else 0.0

    # Test 2: cls loss only
    model.zero_grad(set_to_none=True)
    logits, _ = model(dummy_img)
    cls_loss = nn.functional.cross_entropy(logits, torch.randint(0, 5, (2,), device=device))
    cls_loss.backward()

    for scope, prefix in [("stage2", "stages.2"), ("stage3", "stages.3"),
                           ("stage4", "stages.4"), ("cls_head", "diagnosis_head"),
                           ("decoder", "seg_decoder")]:
        grad_norm = 0.0
        for n, p in model.named_parameters():
            if prefix in n and p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        results[f"cls→{scope}"] = math.sqrt(grad_norm) if grad_norm > 0 else 0.0

    # Check stage2 feature connectivity
    assert model._stage2_feat is not None, "stage2 feature not captured"
    assert model._stage2_feat.requires_grad, "stage2 feature detached from graph"
    assert model._stage2_feat.grad_fn is not None, "stage2 feature has no grad_fn"

    print(f"  seg→stage2: {results['seg→stage2']:.6f}", flush=True)
    print(f"  seg→stage3: {results['seg→stage3']:.6f} (should be ~0)", flush=True)
    print(f"  seg→stage4: {results['seg→stage4']:.6f} (should be ~0)", flush=True)
    print(f"  seg→cls_head: {results['seg→cls_head']:.6f} (should be ~0)", flush=True)
    print(f"  seg→decoder: {results['seg→decoder']:.6f}", flush=True)
    print(f"  cls→stage2: {results['cls→stage2']:.6f}", flush=True)
    print(f"  cls→stage3: {results['cls→stage3']:.6f}", flush=True)
    print(f"  cls→stage4: {results['cls→stage4']:.6f}", flush=True)
    print(f"  cls→cls_head: {results['cls→cls_head']:.6f}", flush=True)
    print(f"  cls→decoder: {results['cls→decoder']:.6f} (should be ~0)", flush=True)

    # Hard gates
    assert results["seg→stage2"] > 0, "SEG GRADIENT NOT REACHING STAGE2"
    assert results["seg→stage3"] < 1e-6 or results["seg→stage3"] < results["seg→stage2"] * 0.01, \
        "Seg gradient leaking into stage3"
    assert results["cls→decoder"] < 1e-6 or results["cls→decoder"] < results["cls→cls_head"] * 0.01, \
        "Cls gradient leaking into decoder"

    print("  PASSED", flush=True)
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_param_groups(model, lr_stage2, lr_stage34, lr_cls_head, lr_decoder, wd):
    groups = {"stage2": [], "stage34": [], "cls_head": [], "decoder": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "diagnosis_head" in n:
            groups["cls_head"].append(p)
        elif "seg_decoder" in n:
            groups["decoder"].append(p)
        elif "stages.2" in n:
            groups["stage2"].append(p)
        elif "stages.3" in n or "stages.4" in n:
            groups["stage34"].append(p)
        else:
            groups["stage34"].append(p)  # catch-all for head/other params

    param_groups = [
        {"params": groups["stage2"], "lr": lr_stage2, "weight_decay": wd, "name": "stage2"},
        {"params": groups["stage34"], "lr": lr_stage34, "weight_decay": wd, "name": "stage34"},
        {"params": groups["cls_head"], "lr": lr_cls_head, "weight_decay": wd, "name": "cls_head"},
        {"params": groups["decoder"], "lr": lr_decoder, "weight_decay": wd, "name": "decoder"},
    ]
    # Remove empty groups
    param_groups = [g for g in param_groups if g["params"]]
    for g in param_groups:
        print(f"  {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.1e}", flush=True)
    return param_groups


def train_epoch(model, loader, optimizer, scaler, amp_enabled, lambda_seg, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels, masks, quality in loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        quality = quality.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, mask_pred = model(images)
            cls_loss = nn.functional.cross_entropy(logits, labels)
            loss = cls_loss
            if lambda_seg > 0 and mask_pred is not None:
                seg_loss = seg_loss_weighted(mask_pred, masks, quality)
                loss = loss + lambda_seg * seg_loss

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
    for images, lbls, masks, _ in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _ = model(images)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(lbls)
    return compute_stage1_metrics(
        torch.cat(labels).numpy(), torch.cat(probs).numpy()
    )


def main(argv=None):
    import math
    import random

    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "C1_joint" if args.lambda_seg > 0 else "C0_control"
    print(f"=== M1-Seg-Joint {tag} (λ_seg={args.lambda_seg}) ===", flush=True)

    # Build model
    print("Building model from M0 checkpoint...", flush=True)
    model = build_m1_seg_joint(args.m0_checkpoint, device, freeze_stem_stage1=True)

    # Gradient connectivity test
    if not args.skip_gradient_test:
        grad_results = gradient_connectivity_test(model, device)
        (out_dir / "gradient_test.json").write_text(json.dumps(grad_results, indent=2))
    else:
        print("  Skipping gradient test", flush=True)

    # Data
    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = SegDataset(args.train_csv, args.mask_dir_train, eval_tf,
                          args.seg_size, args.train_features)
    dev_ds = SegDataset(args.dev_csv, args.mask_dir_dev, eval_tf, args.seg_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Optimiser with grouped LRs
    param_groups = build_param_groups(
        model, args.lr_stage2, args.lr_stage34, args.lr_cls_head, args.lr_decoder, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, amp_enabled,
                                 args.lambda_seg, device)
        dev = evaluate(model, dev_loader, device, amp_enabled)
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
              f"f1={row['dev_macro_f1']:.4f} auc={row['dev_macro_auc']:.4f} ({dt:.1f}s)",
              flush=True)

        if row["dev_macro_f1"] > best_f1:
            best_f1 = row["dev_macro_f1"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "metrics": {k: float(v) for k, v in dev.items()},
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }, out_dir / "best_model.pth")

    # Save artifacts
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # Final predictions
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, lbls, _, _ in dev_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, _ = model(images)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_labels.append(lbls.numpy())
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
    (out_dir / "best_metrics.json").write_text(json.dumps({
        "best_epoch": best["epoch"], "best_macro_f1": best["dev_macro_f1"],
        "best_accuracy": best["dev_accuracy"], "best_macro_auc": best["dev_macro_auc"],
    }, indent=2))
    (out_dir / "args.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(
        {"torch": torch.__version__, "cuda": torch.cuda.is_available()}, indent=2))
    (out_dir / "completed.json").write_text(json.dumps({
        "status": "completed", "best_macro_f1": best_f1,
        "calibration_used": False, "test_used": False, "variant": tag,
    }, indent=2))

    print(f"\n[{tag}] Best dev Macro F1: {best_f1:.6f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
