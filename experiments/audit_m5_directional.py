"""M5 directional geometry audit: diagnose ASC-US→LSIL vs LSIL→ASC-US asymmetry.

Steps:
  1. Directional geometry audit — per-sample distances to pair centroids
  2. Class-conditional percentile normalisation
  3. Prediction-direction stratified scoring
  4. Seed42 re-evaluation with fixed formula
"""

import sys, json, math, argparse
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="M5 directional geometry audit")
    p.add_argument("--features_train", type=Path, required=True)
    p.add_argument("--features_dev", type=Path, required=True)
    p.add_argument("--labels_train", type=Path, required=True)
    p.add_argument("--labels_dev", type=Path, required=True)
    p.add_argument("--dev_preds_csv", type=Path, required=True)
    p.add_argument("--m7_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    return p.parse_args(argv)


def _load_labels(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        col = "diagnosis_label" if "diagnosis_label" in df.columns else "true_label"
        return df[col].values.astype(np.int64)
    return np.load(path, allow_pickle=True).astype(np.int64)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    ft = np.load(args.features_train)
    fd = np.load(args.features_dev)
    lt = _load_labels(args.labels_train)
    ld = _load_labels(args.labels_dev)
    preds = pd.read_csv(args.dev_preds_csv)
    m7 = pd.read_csv(args.m7_csv)

    fn_dev = fd / (np.linalg.norm(fd, axis=1, keepdims=True) + 1e-12)
    fn_train = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-12)

    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    y_pred = preds[prob_cols].values.argmax(axis=1) if prob_cols else preds["pred_label"].values
    y_true = ld
    m7_rej = m7["is_rejected"].values

    # ---- Step 1: Directional geometry ----
    print("=== Step 1: Directional geometry audit ===", flush=True)
    centroids = {}
    for c in range(5):
        mask = lt == c
        centroids[c] = fn_train[mask].mean(axis=0)

    # Per-sample distances to ASC-US(1) and LSIL(2) centroids
    d1 = np.linalg.norm(fn_dev - centroids[1], axis=1)
    d2 = np.linalg.norm(fn_dev - centroids[2], axis=1)

    # For true ASC-US and true LSIL samples
    for true_c, name in [(1, "ASC-US"), (2, "LSIL")]:
        mask = y_true == true_c
        print(f"\n  True {name} ({mask.sum()} samples):", flush=True)
        print(f"    Mean d_to_self: {d1[mask].mean():.4f}" if true_c == 1 else f"    Mean d_to_self: {d2[mask].mean():.4f}", flush=True)
        other_d = d2[mask] if true_c == 1 else d1[mask]
        print(f"    Mean d_to_other: {other_d.mean():.4f}", flush=True)
        ratio = (d1[mask] / (d2[mask] + 1e-12)) if true_c == 1 else (d2[mask] / (d1[mask] + 1e-12))
        print(f"    Mean distance ratio (self/other): {ratio.mean():.4f}", flush=True)
        # For errors specifically
        err_mask = mask & (y_pred != true_c)
        if err_mask.sum() > 0:
            err_ratio = (d1[err_mask] / (d2[err_mask] + 1e-12)) if true_c == 1 else (d2[err_mask] / (d1[err_mask] + 1e-12))
            print(f"    Error samples ({err_mask.sum()}): mean ratio={err_ratio.mean():.4f}", flush=True)

    # ---- Step 2: Class-conditional percentile normalisation ----
    print("\n=== Step 2: Class-conditional percentile normalisation ===", flush=True)
    # Instead of global rank, rank within each predicted class
    m5_cc = np.zeros(len(fd))
    for pred_c in range(5):
        pmask = y_pred == pred_c
        if pmask.sum() == 0:
            continue
        # Score: 1 / (d_to_self + epsilon), higher = more typical
        d_self = np.linalg.norm(fn_dev[pmask] - centroids[pred_c], axis=1)
        typicality = 1.0 / (d_self + 0.01)
        # Rank within this predicted class
        ranks = np.searchsorted(np.sort(typicality), typicality).astype(float)
        m5_cc[pmask] = 1.0 - ranks / (pmask.sum() + 1)  # 0 = most typical, 1 = least typical
    print(f"  Class-conditional percentile: mean={m5_cc.mean():.4f} std={m5_cc.std():.4f}", flush=True)

    # ---- Step 3: Prediction-direction stratified scoring ----
    print("\n=== Step 3: Prediction-direction stratified ===", flush=True)
    # Only score samples whose top-2 is the boundary pair
    # Use distance ratio to pair centroids
    ratio_12 = d1 / (d2 + 1e-12)  # <1 = closer to ASC-US, >1 = closer to LSIL
    # Boundary ambiguity: how close to the decision boundary
    boundary_score = 1.0 - np.abs(d1 - d2) / (d1 + d2 + 1e-12)

    # For samples predicted as ASC-US: risk of being true LSIL
    au_pred_mask = y_pred == 1
    au_risk = np.zeros(len(fd))
    if au_pred_mask.any():
        # High risk = high boundary_score AND ratio_12 > 1 (closer to LSIL centroid despite pred=ASC-US)
        au_risk[au_pred_mask] = boundary_score[au_pred_mask] * np.clip(ratio_12[au_pred_mask], 1.0, 2.0)

    # For samples predicted as LSIL: risk of being true ASC-US
    lsil_pred_mask = y_pred == 2
    lsil_risk = np.zeros(len(fd))
    if lsil_pred_mask.any():
        lsil_risk[lsil_pred_mask] = boundary_score[lsil_pred_mask] * np.clip(1.0 / (ratio_12[lsil_pred_mask] + 1e-12), 1.0, 2.0)

    m5_stratified = au_risk + lsil_risk
    # Rank-normalise the stratified score
    valid = m5_stratified > 0
    if valid.any():
        ranks = np.zeros(len(m5_stratified))
        ranks[valid] = np.searchsorted(np.sort(m5_stratified[valid]), m5_stratified[valid]).astype(float)
        ranks[valid] /= (valid.sum() + 1)
        m5_stratified = ranks
    print(f"  Stratified score: mean={m5_stratified.mean():.4f}", flush=True)

    # ---- Step 4: Combined score (class-conditional + stratified) ----
    m5_final = (m5_cc + m5_stratified) / 2.0
    print(f"  Final M5 score: mean={m5_final.mean():.4f} std={m5_final.std():.4f}", flush=True)

    # ---- Re-evaluate directional performance ----
    print("\n=== Step 4: Directional re-evaluation (M7-accepted subset) ===", flush=True)
    low_mask = np.isin(y_true, [1, 2])
    m7_acc = ~m7_rej
    subset = low_mask & m7_acc

    for true_c, pred_c, name in [(1, 2, "ASC-US→LSIL"), (2, 1, "LSIL→ASC-US")]:
        err_mask = (y_true == true_c) & (y_pred == pred_c) & m7_acc
        n_err = err_mask.sum()
        if n_err == 0:
            print(f"  {name}: 0 errors in M7-accepted", flush=True)
            continue
        s = m5_final[err_mask]
        s_all = m5_final[subset]
        t80 = np.percentile(s_all, 80)
        t90 = np.percentile(s_all, 90)
        extra_80 = (s >= t80).sum()
        extra_90 = (s >= t90).sum()
        print(f"  {name}: {n_err} errors, M5 top-20% catches {extra_80}, top-10% catches {extra_90}",
              flush=True)

    # Conditional AUROC per direction
    print("\n=== Conditional AUROC per direction ===", flush=True)
    for true_c, pred_c, name in [(1, 2, "ASC-US→LSIL"), (2, 1, "LSIL→ASC-US")]:
        # Among M7-accepted low-pair: errors of this direction vs correctly predicted
        err_mask = (y_true == true_c) & (y_pred == pred_c) & m7_acc
        # Correct: predicted as pred_c AND true = pred_c (correct predictions of the predicted class)
        corr_mask = (y_true == pred_c) & (y_pred == pred_c) & m7_acc & low_mask
        combined = err_mask | corr_mask
        if err_mask.sum() > 0 and corr_mask.sum() > 0:
            y_bin = err_mask[combined].astype(int)
            s_bin = m5_final[combined]
            auroc = roc_auc_score(y_bin, s_bin)
            auprc = average_precision_score(y_bin, s_bin)
            print(f"  {name}: {err_mask.sum()} err vs {corr_mask.sum()} corr, AUROC={auroc:.4f} AUPRC={auprc:.4f}",
                  flush=True)

    # ---- M5 precision at thresholds ----
    print("\n=== M5-only precision (low-pair, M7-accepted) ===", flush=True)
    n_subset = subset.sum()
    n_errors = (subset & (y_pred != y_true)).sum()
    for pct_name, pct in [("top-5%", 95), ("top-10%", 90), ("top-20%", 80)]:
        t = np.percentile(m5_final[subset], pct)
        flagged = subset & (m5_final >= t)
        n_flagged = flagged.sum()
        n_caught = (flagged & (y_pred != y_true)).sum()
        prec = n_caught / max(n_flagged, 1)
        nnr = n_flagged / max(n_caught, 1) if n_caught > 0 else float("inf")
        print(f"  {pct_name}: flagged={n_flagged} caught={n_caught} precision={prec:.3f} NNR={nnr:.1f}",
              flush=True)

    # Save scores
    pd.DataFrame({
        "m5_class_conditional": m5_cc,
        "m5_stratified": m5_stratified,
        "m5_final": m5_final,
    }).to_csv(out_dir / "dev_m5v2_scores.csv", index=False)

    print(f"\nResults: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
