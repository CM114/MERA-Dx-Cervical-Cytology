"""Train new M3 dual-view model (frozen M0 backbone + nucleus features).

Single-variable experiment: only difference from M0 is nucleus morphology features.
"""

import argparse
import json
import math
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

from experiments.tbs.nucleus_dataset import DualViewDataset, fit_scaler
from experiments.tbs.nucleus_model import build_dualview_model, DualViewModel
from experiments.tbs.dataset import build_transforms
from experiments.tbs.metrics import compute_stage1_metrics


FEATURE_COLUMNS = [
    "nucleus_area_px", "nucleus_area_frac", "nucleus_perimeter",
    "equivalent_diameter", "solidity", "eccentricity", "circularity",
    "component_count", "largest_component_area_px", "largest_component_area_frac",
    "edge_contact", "boundary_gradient_mean",
    "nucleus_r_mean", "nucleus_r_std", "nucleus_g_mean", "nucleus_g_std",
    "nucleus_b_mean", "nucleus_b_std",
]

OWNER_FILENAME = ".m3_dualview_owner.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train M3 dual-view nucleus model")
    parser.add_argument("--m0_checkpoint", type=Path, required=True,
                        help="Path to frozen M0 best_model.pth")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--train_features", type=Path, required=True,
                        help="train_nucleus_features.csv from feature extraction")
    parser.add_argument("--dev_features", type=Path, required=True,
                        help="dev_nucleus_features.csv from feature extraction")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (42, 7, or 2026)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224,
                        help="Letterbox size (must match M0)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision")
    parser.add_argument("--device", default="cuda:1",
                        help="Torch device")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for name, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(token in str(path).lower() for token in ("test", "calibration")):
            parser.error(f"{name} must not contain 'test' or 'calibration'")

    return args


def collate_fn(batch):
    """Collate (image_tensor, nucleus_vec, label, sample_id) tuples."""
    images = torch.stack([b[0] for b in batch])
    nucleus = torch.from_numpy(np.stack([b[1] for b in batch]))
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    sample_ids = [b[3] for b in batch]
    return images, nucleus, labels, sample_ids


def train_epoch(model, loader, optimizer, scaler, amp_enabled, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, nucleus, labels, _ in loader:
        images = images.to(device)
        nucleus = nucleus.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _, _ = model(images, nucleus)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    all_logits = []
    all_labels = []
    all_probs = []
    for images, nucleus, labels, _ in loader:
        images = images.to(device)
        nucleus = nucleus.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _, _ = model(images, nucleus)
        probs = torch.softmax(logits.float(), dim=1)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    metrics = compute_stage1_metrics(labels.numpy(), probs.numpy())
    return metrics, probs.numpy(), labels.numpy()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_out_dir(out_dir, overwrite):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {out_dir}")
        existing = [p for p in out_dir.iterdir() if not p.name.startswith(".")]
        if existing and not overwrite:
            raise FileExistsError(f"Output directory non-empty; use --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    owner = out_dir / OWNER_FILENAME
    owner.write_text(json.dumps({"owner": "m3_dualview", "created": str(datetime.datetime.now())}, indent=2))
    return out_dir


def main(argv=None):
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU", flush=True)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    amp_enabled = args.amp and torch.cuda.is_available()

    # --- Build model ---------------------------------------------------------
    print("Building model from frozen M0 checkpoint...", flush=True)
    model = build_dualview_model(args.m0_checkpoint, device)
    # Verify backbone is frozen
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    if backbone_trainable > 0:
        raise RuntimeError(f"Backbone has {backbone_trainable} trainable parameters; must be 0")
    print(f"  Backbone trainable params: {backbone_trainable}", flush=True)

    # --- Fit scaler on train features ----------------------------------------
    print("Fitting StandardScaler on train nucleus features...", flush=True)
    scaler = fit_scaler(args.train_features, FEATURE_COLUMNS)
    # Save scaler immediately for reproducibility
    import pickle
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # --- Datasets -------------------------------------------------------------
    _, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = DualViewDataset(
        args.train_csv, args.train_features, eval_transform, scaler, FEATURE_COLUMNS,
    )
    dev_ds = DualViewDataset(
        args.dev_csv, args.dev_features, eval_transform, scaler, FEATURE_COLUMNS,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # --- Optimiser (only MLP + classifier, backbone is frozen) ----------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # --- Training loop --------------------------------------------------------
    best_macro_f1 = -1.0
    metrics_rows = []
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler_amp, amp_enabled, device)
        dev_metrics, dev_probs, dev_labels = evaluate(model, dev_loader, device, amp_enabled)
        scheduler.step()
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(dev_metrics.get("nll", math.nan)),
            "accuracy": float(dev_metrics["accuracy"]),
            "macro_f1": float(dev_metrics["macro_f1"]),
            "balanced_accuracy": float(dev_metrics["balanced_accuracy"]),
            "macro_auc": float(dev_metrics["macro_auc"]),
            "nll": float(dev_metrics["nll"]),
            "ece": float(dev_metrics["ece"]),
            "brier": float(dev_metrics["brier"]),
        }
        metrics_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f} "
            f"dev_f1={row['macro_f1']:.4f} dev_auc={row['macro_auc']:.4f} "
            f"nll={row['nll']:.4f} ({dt:.1f}s)",
            flush=True,
        )

        if row["macro_f1"] > best_macro_f1:
            best_macro_f1 = row["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": {k: float(v) for k, v in dev_metrics.items()},
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "nucleus_dims": {
                        "cell_dim": model.cell_dim,
                        "nucleus_dim": model.nucleus_dim,
                        "nucleus_emb_dim": model.nucleus_emb_dim,
                    },
                },
                out_dir / "best_model.pth",
            )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    # --- Final dev predictions ------------------------------------------------
    _, best_probs, best_labels = evaluate(model, dev_loader, device, amp_enabled)
    pred_df = pd.DataFrame(best_probs, columns=["prob_normal", "prob_ascus", "prob_lsil", "prob_asch", "prob_hsil"])
    pred_df["pred_label"] = best_probs.argmax(axis=1)
    pred_df["true_label"] = best_labels
    pred_df.to_csv(out_dir / "dev_predictions.csv", index=False)

    # --- Best metrics ---------------------------------------------------------
    best_metrics = {
        "best_epoch": best_epoch,
        "best_macro_f1": float(best_macro_f1),
        "best_accuracy": float(metrics_rows[best_epoch - 1]["accuracy"]),
        "best_balanced_accuracy": float(metrics_rows[best_epoch - 1]["balanced_accuracy"]),
        "best_macro_auc": float(metrics_rows[best_epoch - 1]["macro_auc"]),
        "best_nll": float(metrics_rows[best_epoch - 1]["nll"]),
        "best_ece": float(metrics_rows[best_epoch - 1]["ece"]),
    }
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2))

    # --- Confusion matrix -----------------------------------------------------
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(best_labels, best_probs.argmax(axis=1)):
        cm[t, p] += 1
    cm_df = pd.DataFrame(
        cm,
        index=["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"],
        columns=["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # --- Args + environment ---------------------------------------------------
    args_json = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(args_json, indent=2))
    env_json = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "environment.json").write_text(json.dumps(env_json, indent=2))

    # --- Completed ------------------------------------------------------------
    completed = {
        "schema_version": "m3-dualview-v1",
        "status": "completed",
        "best_epoch": best_epoch,
        "best_macro_f1": float(best_macro_f1),
        "train_samples": len(train_ds),
        "dev_samples": len(dev_ds),
        "calibration_used": False,
        "test_used": False,
    }
    (out_dir / "completed.json").write_text(json.dumps(completed, indent=2))

    print(f"\nBest dev Macro F1: {best_macro_f1:.6f} (epoch {best_epoch})", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
