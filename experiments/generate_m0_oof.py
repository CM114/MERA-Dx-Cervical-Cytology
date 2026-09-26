"""Generate 5-fold OOF predictions for M0 on the training set.

Each fold: fine-tune M0 from its checkpoint for 3 epochs on 4 folds,
predict on the held-out fold.  Identifies difficult boundary samples
for M2-Residual-OOF training.
"""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.tbs.models import build_stage1_model
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="M0 5-fold OOF generation")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def split_folds(n_samples, n_folds, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        folds.append(indices[start:end])
    return folds


def train_fold(model, train_indices, dataset, device, amp_enabled, epochs, lr, batch_size, num_workers):
    """Fine-tune M0 on a subset for a few epochs."""
    train_subset = Subset(dataset, train_indices)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["diagnosis_label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(images)
                loss = nn.functional.cross_entropy(output["diagnosis_logits"], labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
        scheduler.step()
    return model


@torch.no_grad()
def predict_fold(model, indices, dataset, device, amp_enabled):
    """Generate predictions for a held-out fold."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_probs, all_logits, all_labels, all_paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"]
        paths = batch["image_path"]
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
        probs = output["diagnosis_probs"].cpu().numpy()
        logits = output["diagnosis_logits"].cpu().numpy()
        all_probs.append(probs)
        all_logits.append(logits)
        all_labels.append(labels.numpy())
        all_paths.extend(paths)
    return (
        np.concatenate(all_probs),
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_paths,
    )


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

    # Load full dataset
    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    full_ds = XUDataTBS5Dataset(args.train_csv, eval_tf)
    n = len(full_ds)
    print(f"Train samples: {n}, folds: {args.n_folds}", flush=True)

    folds = split_folds(n, args.n_folds, args.seed)

    all_oof_probs = np.zeros((n, 5), dtype=np.float64)
    all_oof_logits = np.zeros((n, 5), dtype=np.float64)
    all_labels = np.zeros(n, dtype=np.int64)
    all_paths = [""] * n

    for fold_idx, test_indices in enumerate(folds):
        train_indices = np.setdiff1d(np.arange(n), test_indices)
        print(f"\nFold {fold_idx + 1}/{args.n_folds}: train={len(train_indices)} test={len(test_indices)}", flush=True)

        # Build fresh model from checkpoint
        model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
        ckpt = torch.load(args.m0_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=True)
        model.to(device)

        t0 = time.time()
        model = train_fold(model, train_indices, full_ds, device, amp_enabled,
                           args.finetune_epochs, args.lr, args.batch_size, args.num_workers)
        probs, logits, labels, paths = predict_fold(model, test_indices, full_ds, device, amp_enabled)
        dt = time.time() - t0

        all_oof_probs[test_indices] = probs
        all_oof_logits[test_indices] = logits
        all_labels[test_indices] = labels
        for j, idx in enumerate(test_indices):
            all_paths[idx] = paths[j]

        # Quick metrics
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)
        print(f"  Fold OOF Macro F1: {f1:.4f} ({dt:.1f}s)", flush=True)

    # Save OOF predictions
    df = pd.DataFrame(all_oof_probs, columns=["prob_N","prob_AU","prob_L","prob_AH","prob_HH"])
    df["oof_pred"] = all_oof_probs.argmax(axis=1)
    df["true_label"] = all_labels
    df["image_path"] = all_paths

    # Compute margins
    for i in range(n):
        tc = all_labels[i]
        df.at[i, "true_class_prob"] = all_oof_probs[i, tc]
        wrong = np.delete(all_oof_probs[i], tc)
        df.at[i, "margin"] = all_oof_probs[i, tc] - wrong.max()

        # Top-2 check for boundary pairs
        sorted_idx = np.argsort(-all_oof_probs[i])
        top2 = sorted(sorted_idx[:2])
        df.at[i, "top2_is_low_pair"] = (top2 == [1, 2])
        df.at[i, "top2_is_high_pair"] = (top2 == [3, 4])
        if df.at[i, "top2_is_low_pair"]:
            df.at[i, "pair_margin"] = abs(all_oof_probs[i, 1] - all_oof_probs[i, 2])
        elif df.at[i, "top2_is_high_pair"]:
            df.at[i, "pair_margin"] = abs(all_oof_probs[i, 3] - all_oof_probs[i, 4])
        else:
            df.at[i, "pair_margin"] = float("nan")

    df.to_csv(out_dir / "train_oof_predictions.csv", index=False)

    # Identify difficult boundary samples
    low_mask = (
        df["true_label"].isin([1, 2])
        & (
            (df["oof_pred"] != df["true_label"])
            | ((df["top2_is_low_pair"]) & (df["pair_margin"] < 0.3))
        )
    )
    high_mask = (
        df["true_label"].isin([3, 4])
        & (
            (df["oof_pred"] != df["true_label"])
            | ((df["top2_is_high_pair"]) & (df["pair_margin"] < 0.3))
        )
    )

    low_diff = df[low_mask].copy()
    high_diff = df[high_mask].copy()
    low_diff["difficulty_set"] = "low"
    high_diff["difficulty_set"] = "high"
    difficult = pd.concat([low_diff, high_diff], ignore_index=True)

    difficult_path = out_dir / "difficult_boundary_samples.csv"
    difficult.to_csv(difficult_path, index=False)

    # Summary
    summary = {
        "train_samples": n,
        "n_folds": args.n_folds,
        "oof_macro_f1": float(f1_score(all_labels, all_oof_probs.argmax(axis=1), average="macro", zero_division=0)),
        "low_difficult": int(low_mask.sum()),
        "high_difficult": int(high_mask.sum()),
        "total_difficult": int(low_mask.sum() + high_mask.sum()),
        "low_oof_errors": int((df["true_label"].isin([1, 2]) & (df["oof_pred"] != df["true_label"])).sum()),
        "high_oof_errors": int((df["true_label"].isin([3, 4]) & (df["oof_pred"] != df["true_label"])).sum()),
    }
    print(f"\nOOF F1: {summary['oof_macro_f1']:.4f}", flush=True)
    print(f"Low boundary difficult: {summary['low_difficult']}", flush=True)
    print(f"High boundary difficult: {summary['high_difficult']}", flush=True)
    (out_dir / "oof_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
