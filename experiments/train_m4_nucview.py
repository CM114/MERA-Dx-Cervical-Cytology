"""Train M4-NucView with V0-V4 variant support and ROI extraction."""

import argparse, json, sys, time, datetime
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T

from experiments.tbs.m4_nucview_model import (
    build_m4_nucview,
    extract_nucleus_roi, extract_center_roi, extract_random_roi,
)
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_ROI_TF = T.Compose([T.Resize((128, 128)), T.ToTensor(), _NORM])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train M4-NucView")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--mask_dir_train", type=Path, required=True)
    p.add_argument("--mask_dir_dev", type=Path, required=True)
    p.add_argument("--variant", type=str, default="V1", choices=["V0","V1","V2","V3","V4"])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_kl", type=float, default=0.5)
    p.add_argument("--lambda_delta", type=float, default=0.01)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--roi_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    for n, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(t in str(path).lower() for t in ("test","calibration")):
            p.error(f"{n} must not contain those words")
    return args


class NucViewDataset:
    def __init__(self, manifest_csv, mask_dir, transform, variant, roi_size=128, rng=None):
        self.base = XUDataTBS5Dataset(manifest_csv, transform)
        self.mask_dir = Path(mask_dir)
        self.variant = variant
        self.roi_size = roi_size
        self.rng = rng or np.random.RandomState(42)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        sid = Path(item["image_path"]).stem
        mask_path = self.mask_dir / f"{sid}.png"
        full_img = Image.open(item["image_path"]).convert("RGB")

        core_roi = torch.zeros(3, self.roi_size, self.roi_size)
        ctx_roi = torch.zeros(3, self.roi_size, self.roi_size)

        if self.variant != "V0":
            try:
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=bool)
            except Exception:
                mask = np.zeros((full_img.height, full_img.width), dtype=bool)

            if self.variant == "V1":
                c = extract_nucleus_roi(full_img, mask, 1.15, self.roi_size)
                x = extract_nucleus_roi(full_img, mask, 1.5, self.roi_size)
            elif self.variant == "V2":
                c = extract_center_roi(full_img, self.roi_size)
                x = extract_center_roi(full_img, self.roi_size)
            elif self.variant == "V3":
                c = extract_random_roi(full_img, mask, self.roi_size, self.rng)
                x = extract_random_roi(full_img, mask, self.roi_size, self.rng)
            elif self.variant == "V4":
                c = extract_nucleus_roi(full_img, mask, 1.15, self.roi_size)
                x = extract_nucleus_roi(full_img, mask, 1.5, self.roi_size)
                # Swap core with another sample in the same batch (handled in collate)

            if c is not None:
                core_roi = _ROI_TF(c)
            if x is not None:
                ctx_roi = _ROI_TF(x)

        return {
            "image": item["image"],
            "label": torch.tensor(item["diagnosis_label"], dtype=torch.long),
            "core_roi": core_roi,
            "ctx_roi": ctx_roi,
            "sample_id": sid,
            "diagnosis_name": item["diagnosis_name"],
        }


def collate_v4(batch):
    """V4: swap core_roi within same diagnosis class."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    core = torch.stack([b["core_roi"] for b in batch])
    ctx = torch.stack([b["ctx_roi"] for b in batch])

    # Shuffle core_roi within same diagnosis class
    diag_groups = {}
    for i, b in enumerate(batch):
        dn = b["diagnosis_name"]
        diag_groups.setdefault(dn, []).append(i)
    new_core = core.clone()
    for dn, indices in diag_groups.items():
        if len(indices) > 1:
            perm = torch.randperm(len(indices))
            for j, orig_idx in enumerate(indices):
                new_core[orig_idx] = core[indices[perm[j]]]
    return images, labels, new_core, ctx


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    core = torch.stack([b["core_roi"] for b in batch])
    ctx = torch.stack([b["ctx_roi"] for b in batch])
    return images, labels, core, ctx


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device, lambda_kl, lambda_delta):
    model.train()
    total_loss, n = 0.0, 0
    gates, deltas = [], []
    for images, labels, core, ctx in loader:
        images, labels = images.to(device), labels.to(device)
        core, ctx = core.to(device), ctx.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            z, info = model(images, core, ctx)
            ce = F.cross_entropy(z, labels)
            loss = ce
            if lambda_delta > 0:
                loss = loss + lambda_delta * info.get("delta_norm", 0.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        gates.append(info.get("gate", 0.0))
        deltas.append(info.get("delta_norm", 0.0))
    return total_loss / max(n, 1), np.mean(gates), np.mean(deltas)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    probs, labels = [], []
    for images, lbls, core, ctx in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            z, _ = model(images, core.to(device), ctx.to(device))
        probs.append(torch.softmax(z.float(), dim=1).cpu().numpy())
        labels.append(lbls.numpy())
    return compute_stage1_metrics(np.concatenate(labels), np.concatenate(probs)), np.concatenate(probs), np.concatenate(labels)


def main(argv=None):
    args = parse_args(argv)
    import random; random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== M4-NucView {args.variant} ===", flush=True)
    model = build_m4_nucview(args.m0_checkpoint, device, args.variant)

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    rng = np.random.RandomState(args.seed)
    train_ds = NucViewDataset(args.train_csv, args.mask_dir_train, eval_tf, args.variant, args.roi_size, rng)
    dev_ds = NucViewDataset(args.dev_csv, args.mask_dir_dev, eval_tf, args.variant, args.roi_size, rng)

    cf = collate_v4 if args.variant == "V4" else collate_fn
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=cf)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    if args.variant == "V0":
        dev, probs, labels = evaluate(model, dev_loader, device, amp_enabled)
        best_f1 = float(dev["macro_f1"])
        rows = [{"epoch": 0, "train_loss": 0, "dev_macro_f1": best_f1, "dev_macro_auc": float(dev["macro_auc"]), "gate": 0, "delta": 0}]
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,}", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        best_f1 = -1.0; rows = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, gate_val, delta_val = train_epoch(model, train_loader, optimizer, scaler, amp_enabled, device, args.lambda_kl, args.lambda_delta)
            dev, probs, labels = evaluate(model, dev_loader, device, amp_enabled)
            scheduler.step(); dt = time.time() - t0
            row = {"epoch": epoch, "train_loss": train_loss, "dev_macro_f1": float(dev["macro_f1"]),
                   "dev_macro_auc": float(dev["macro_auc"]), "gate": gate_val, "delta": delta_val}
            rows.append(row)
            print(f"Epoch {epoch:2d}: loss={train_loss:.4f} f1={row['dev_macro_f1']:.4f} auc={row['dev_macro_auc']:.4f} gate={gate_val:.3f} Δ={delta_val:.4f} ({dt:.1f}s)", flush=True)
            if row["dev_macro_f1"] > best_f1:
                best_f1 = row["dev_macro_f1"]
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": {k: float(v) for k, v in dev.items()}}, out_dir / "best_model.pth")

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # Final predictions
    model.eval(); all_probs, all_labels = [], []
    with torch.no_grad():
        for images, lbls, core, ctx in dev_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                z, _ = model(images.to(device), core.to(device), ctx.to(device))
            all_probs.append(torch.softmax(z.float(), dim=1).cpu().numpy()); all_labels.append(lbls.numpy())
    probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)
    pdf = pd.DataFrame(probs, columns=["prob_N","prob_AU","prob_L","prob_AH","prob_HH"])
    pdf["pred_label"] = probs.argmax(axis=1); pdf["true_label"] = labels
    pdf.to_csv(out_dir / "dev_predictions.csv", index=False)

    cm = np.zeros((5,5), dtype=int)
    for t, p in zip(labels, probs.argmax(axis=1)): cm[t,p] += 1
    pd.DataFrame(cm, index=["N","AU","L","AH","H"], columns=["N","AU","L","AH","H"]).to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "best_metrics.json").write_text(json.dumps(dict(rows[-1]), indent=2))
    (out_dir / "completed.json").write_text(json.dumps({"status":"completed","best_f1":best_f1,"variant":args.variant,"calibration_used":False,"test_used":False}, indent=2))
    print(f"\n[{args.variant}] Best F1: {best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
