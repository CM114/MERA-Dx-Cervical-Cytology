"""OOF independence audit with proxy group-aware evaluation.

Phase 1: Fixed OOF — each fold trained from ImageNet init (not M0 ckpt).
Phase 2: Proxy group construction from filename patterns + perceptual hash.
Phase 3: StratifiedGroupKFold OOF vs random OOF comparison.
Phase 4: Train-vs-dev domain classifier.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from experiments.tbs.models import build_stage1_model
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OOF independence audit")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase 2: Proxy group construction
# ---------------------------------------------------------------------------


def build_proxy_groups(manifest_df):
    """Build proxy source groups from filename heuristics.

    Returns array of group IDs (int), same length as manifest.
    """
    paths = manifest_df["image_path"].tolist()
    groups = np.full(len(paths), -1, dtype=int)
    next_gid = 0

    # Rule 1: dated prefix (e.g. "2020-11-24 - 201703151")
    import re
    dated = re.compile(r'^(\d{4}-\d{2}-\d{2}\s*[-_]\s*\d+)')
    # Rule 2: source prefix from audit (format: source:<id>)
    # Rule 3: 3-segment numeric prefix (e.g. "0_11166_139")
    numeric3 = re.compile(r'^(\d+_\d+_\d+)(?:_|\.)')
    # Rule 4: 4-segment numeric (e.g. position/index format)
    numeric4 = re.compile(r'^(\d+_\d+_\d+_\d+)(?:_|\.)')

    group_map = {}
    for i, p in enumerate(paths):
        stem = Path(p).stem
        keys = []
        m = dated.match(stem)
        if m:
            keys.append(("dated", m.group(1)[:16]))  # first 16 chars
        m = numeric4.match(stem)
        if m:
            keys.append(("n4", m.group(1)))
        else:
            m = numeric3.match(stem)
            if m:
                keys.append(("n3", m.group(1)))

        if keys:
            key = keys[0]  # use highest-priority match
            if key not in group_map:
                group_map[key] = next_gid
                next_gid += 1
            groups[i] = group_map[key]

    # Assign singletons unique IDs
    for i in range(len(groups)):
        if groups[i] == -1:
            groups[i] = next_gid
            next_gid += 1

    n_groups = len(set(groups))
    n_singletons = sum(1 for g in groups if list(groups).count(g) == 1)
    print(f"  Proxy groups: {n_groups} total, {n_singletons} singletons", flush=True)
    return groups


# ---------------------------------------------------------------------------
# Phase 3: Group-aware OOF
# ---------------------------------------------------------------------------


def run_group_aware_oof(full_ds, labels, groups, n_folds, device, amp_enabled, args):
    """Run StratifiedGroupKFold OOF with ImageNet init for each fold."""

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    n = len(full_ds)
    all_oof_probs = np.zeros((n, 5), dtype=np.float64)
    all_oof_labels = np.zeros(n, dtype=np.int64)

    fold_idx = 0
    for train_idx, test_idx in sgkf.split(np.zeros(n), labels, groups):
        fold_idx += 1
        print(f"\n  Fold {fold_idx}/{n_folds}: train={len(train_idx)} test={len(test_idx)}", flush=True)

        # ImageNet init (NOT from M0 checkpoint!)
        model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=True).to(device)

        train_subset = Subset(full_ds, train_idx)
        loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        t0 = time.time()
        model.train()
        for epoch in range(args.train_epochs):
            total_loss = 0.0
            for batch in loader:
                images = batch["image"].to(device)
                lbls = batch["diagnosis_label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(images)
                    loss = nn.functional.cross_entropy(output["diagnosis_logits"], lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * images.size(0)
            scheduler.step()

        # Predict held-out fold
        test_subset = Subset(full_ds, test_idx)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images = batch["image"].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(images)
                probs = output["diagnosis_probs"].cpu().numpy()
                start = batch_idx * args.batch_size
                end = start + len(probs)
                all_oof_probs[test_idx[start:end]] = probs
                all_oof_labels[test_idx[start:end]] = batch["diagnosis_label"].numpy()

        dt = time.time() - t0
        fold_f1 = f1_score(all_oof_labels[test_idx], all_oof_probs[test_idx].argmax(axis=1),
                           average="macro", zero_division=0)
        print(f"    Fold OOF F1: {fold_f1:.4f} ({dt:.1f}s)", flush=True)

    return all_oof_probs, all_oof_labels


def count_difficult(probs, labels):
    """Count difficult boundary samples."""
    pred = probs.argmax(axis=1)
    low_mask = np.isin(labels, [1, 2]) & (pred != labels)
    high_mask = np.isin(labels, [3, 4]) & (pred != labels)
    # Also count low-margin correct predictions
    for i in range(len(labels)):
        tc = labels[i]
        wrong = np.delete(probs[i], tc)
        margin = probs[i, tc] - wrong.max()
        if margin < 0.1:
            if labels[i] in [1, 2] and i not in np.where(low_mask)[0]:
                low_mask[i] = True
            elif labels[i] in [3, 4] and i not in np.where(high_mask)[0]:
                high_mask[i] = True
    return int(low_mask.sum()), int(high_mask.sum())


# ---------------------------------------------------------------------------
# Phase 4: Train-vs-dev domain classifier
# ---------------------------------------------------------------------------


def domain_classifier_audit(m0_checkpoint_path, train_csv, dev_csv, device, args):
    """Train logistic regression on frozen M0 features to distinguish train vs dev."""
    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_ds = XUDataTBS5Dataset(train_csv, eval_tf)
    dev_ds = XUDataTBS5Dataset(dev_csv, eval_tf)

    model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=True)
    model.to(device)
    model.eval()

    def extract_features(dataset, label):
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        feats, targets = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    output = model(images)
                feats.append(output["features"].cpu().numpy())
                targets.append(np.full(len(images), label))
        return np.concatenate(feats), np.concatenate(targets)

    X_train, y_train = extract_features(train_ds, 0)
    X_dev, y_dev = extract_features(dev_ds, 1)
    X = np.concatenate([X_train, X_dev])
    y = np.concatenate([y_train, y_dev])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
    clf.fit(X_scaled, y)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_prob)
    acc = clf.score(X_scaled, y)
    print(f"  Domain classifier AUC: {auc:.4f}  Accuracy: {acc:.4f}", flush=True)
    return {"domain_auc": float(auc), "domain_accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    full_ds = XUDataTBS5Dataset(args.train_csv, eval_tf)
    train_df = pd.read_csv(args.train_csv)
    labels = train_df["diagnosis_label"].values.astype(np.int64)
    n = len(full_ds)

    # Phase 2: Build proxy groups
    print("=== Phase 2: Proxy group construction ===", flush=True)
    groups = build_proxy_groups(train_df)

    # Phase 3a: Random OOF (from ImageNet init)
    print(f"\n=== Phase 3a: Random OOF ({args.n_folds}-fold, ImageNet init) ===", flush=True)
    random_groups = np.arange(n)  # each sample is its own group = random split
    probs_random, labels_random = run_group_aware_oof(
        full_ds, labels, random_groups, args.n_folds, device, amp_enabled, args
    )
    f1_random = f1_score(labels_random, probs_random.argmax(axis=1), average="macro", zero_division=0)
    low_r, high_r = count_difficult(probs_random, labels_random)
    print(f"\n  Random OOF F1: {f1_random:.4f}  Difficult: low={low_r} high={high_r}", flush=True)

    # Phase 3b: Group-aware OOF
    print(f"\n=== Phase 3b: Group-aware OOF ===", flush=True)
    probs_group, labels_group = run_group_aware_oof(
        full_ds, labels, groups, args.n_folds, device, amp_enabled, args
    )
    f1_group = f1_score(labels_group, probs_group.argmax(axis=1), average="macro", zero_division=0)
    low_g, high_g = count_difficult(probs_group, labels_group)
    print(f"\n  Group-aware OOF F1: {f1_group:.4f}  Difficult: low={low_g} high={high_g}", flush=True)

    # Phase 4: Domain classifier
    print(f"\n=== Phase 4: Train-vs-dev domain classifier ===", flush=True)
    # Find M0 checkpoint for feature extraction
    m0_ckpt = Path("results/tbs5/stage1/m0_caformer_letterbox_clean_v2_seed42/best_model.pth")
    domain_result = domain_classifier_audit(m0_ckpt, args.train_csv, args.dev_csv, device, args)

    # Summary
    summary = {
        "random_oof_f1": float(f1_random),
        "random_oof_difficult_low": low_r,
        "random_oof_difficult_high": high_r,
        "group_aware_oof_f1": float(f1_group),
        "group_aware_oof_difficult_low": low_g,
        "group_aware_oof_difficult_high": high_g,
        "oof_gap": float(f1_random - f1_group),
        **domain_result,
        "dev_macro_f1_reference": 0.7845,
        "train_oof_to_dev_gap_random": float(f1_random - 0.7845),
        "train_oof_to_dev_gap_group": float(f1_group - 0.7845),
    }
    print(f"\n=== Summary ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
