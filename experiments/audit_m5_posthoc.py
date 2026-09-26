"""M5 post-hoc: compute M5-only rejection stats, conditional AUROC, directional breakdown, M7 complementarity.

Reads M5 boundary scores + M7 decisions from previous audit output.
"""

import sys, json, math, argparse
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="M5 post-hoc audit")
    p.add_argument("--m5_csv", type=Path, required=True,
                   help="dev_boundary_scores.csv from M5 audit")
    p.add_argument("--m7_csv", type=Path, required=True,
                   help="M7 dev_decisions.csv")
    p.add_argument("--dev_preds_csv", type=Path, required=True,
                   help="M0 dev_predictions.csv for true labels")
    p.add_argument("--out_dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m5 = pd.read_csv(args.m5_csv)
    m7 = pd.read_csv(args.m7_csv)
    preds = pd.read_csv(args.dev_preds_csv)

    m5_score = m5["m5_boundary_score"].values
    m7_rejected = m7["is_rejected"].values

    col = "true_label" if "true_label" in preds.columns else (
        "diagnosis_label" if "diagnosis_label" in preds.columns else None
    )
    if col is None:
        for c in preds.columns:
            if "label" in c.lower() and "pred" not in c.lower():
                col = c; break
    y = preds[col].values.astype(np.int64)

    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    if prob_cols:
        p = preds[prob_cols].values.argmax(axis=1)
    else:
        p = preds["pred_label"].values

    # ---- Directional analysis ----
    print("=== Directional breakdown (ASC-US <-> LSIL) ===", flush=True)
    for true_c, pred_c, name in [(1, 2, "ASC-US→LSIL"), (2, 1, "LSIL→ASC-US")]:
        mask = (y == true_c) & (p == pred_c)
        n_err = mask.sum()
        m7_caught = (mask & m7_rejected).sum() if n_err > 0 else 0
        # Among M7-accepted errors, top-20% by M5
        m7_accepted = mask & ~m7_rejected
        n_missed = m7_accepted.sum()
        if n_missed > 0:
            s_missed = m5_score[m7_accepted]
            s_all_acc = m5_score[~m7_rejected]
            t20 = np.percentile(s_all_acc, 80) if len(s_all_acc) > 0 else 0
            m5_extra = (s_missed >= t20).sum()
        else:
            m5_extra = 0
        print(f"  {name}: {n_err} errors, M7 caught {m7_caught}, M5 extra {m5_extra}/{n_missed} missed",
              flush=True)

    # Correct ASC-US/LSIL rejected by M5
    for true_c, name in [(1, "ASC-US"), (2, "LSIL")]:
        mask = (y == true_c) & (p == true_c) & ~m7_rejected
        n_correct = mask.sum()
        if n_correct > 0:
            s_correct = m5_score[mask]
            s_all_acc = m5_score[~m7_rejected]
            t20 = np.percentile(s_all_acc, 80)
            false_pos = (s_correct >= t20).sum()
            print(f"  M5 would flag {false_pos}/{n_correct} correct {name} (top-20%)", flush=True)

    # ---- M5-only rejection stats on low pair ----
    print("\n=== M5-only rejection (low pair, M7-accepted subset) ===", flush=True)
    low_mask = np.isin(y, [1, 2])
    m7_acc = ~m7_rejected
    subset = low_mask & m7_acc
    n_subset = subset.sum()
    n_errors = (subset & (p != y)).sum()

    print(f"  M7-accepted low-pair samples: {n_subset}", flush=True)
    print(f"  Errors in this subset: {n_errors}", flush=True)

    for pct_name, pct in [("top-5%", 95), ("top-10%", 90), ("top-20%", 80), ("top-30%", 70)]:
        t = np.percentile(m5_score[subset], pct) if n_subset > 0 else 0
        flagged = subset & (m5_score >= t)
        n_flagged = flagged.sum()
        n_caught = (flagged & (p != y)).sum()
        n_false = n_flagged - n_caught
        prec = n_caught / max(n_flagged, 1)
        nnr = n_flagged / max(n_caught, 1) if n_caught > 0 else float("inf")
        extra_rate = n_flagged / max(len(m7_acc), 1)
        print(f"    {pct_name}: flagged={n_flagged} caught={n_caught} precision={prec:.3f} NNR={nnr:.1f} extra_review={extra_rate:.3%}",
              flush=True)

    # ---- Conditional AUROC on M7-accepted low-pair ----
    print("\n=== Conditional AUROC (M7-accepted low-pair) ===", flush=True)
    if n_errors > 0 and n_errors < n_subset:
        y_sub = (subset & (p != y)).astype(int)[subset]
        s_sub = m5_score[subset]
        auroc = roc_auc_score(y_sub, s_sub)
        auprc = average_precision_score(y_sub, s_sub)
        print(f"  Conditional AUROC={auroc:.4f}  AUPRC={auprc:.4f}", flush=True)

    # ---- M5 vs M7 complementarity ----
    print("\n=== M5 vs M7 complementarity ===", flush=True)
    # Spearman correlation
    from scipy.stats import spearmanr
    # M7 risk proxy: min_risk_value from M7 decisions
    if "min_risk_value" in m7.columns:
        m7_risk = m7["min_risk_value"].values
    else:
        m7_risk = m7_rejected.astype(float)
    corr, pval = spearmanr(m5_score, m7_risk)
    print(f"  Spearman(M5, M7_risk): r={corr:.4f} p={pval:.4f}", flush=True)

    # Jaccard overlap of top-20% flagged
    m5_top20 = m5_score >= np.percentile(m5_score, 80)
    if "min_risk_value" in m7.columns:
        m7_top20 = m7_risk >= np.percentile(m7_risk, 80)
    else:
        m7_top20 = m7_rejected
    jaccard = (m5_top20 & m7_top20).sum() / max((m5_top20 | m7_top20).sum(), 1)
    print(f"  Top-20% Jaccard overlap: {jaccard:.4f}", flush=True)

    # Confidence distribution of M5-only flagged correct low-pair samples
    flagged_correct = subset & (m5_score >= np.percentile(m5_score[subset], 80)) & (p == y)
    if flagged_correct.sum() > 0 and prob_cols:
        conf = preds[prob_cols].values.max(axis=1)[flagged_correct]
        print(f"  M5-flagged correct low samples confidence: mean={conf.mean():.3f} median={np.median(conf):.3f}",
              flush=True)

    # ---- Summary JSON ----
    summary = {
        "low_pair_m7_accepted_errors": int(n_errors),
        "low_pair_m7_accepted_total": int(n_subset),
        "spearman_m5_m7": float(corr),
        "jaccard_top20": float(jaccard),
    }
    (out_dir / "m5_posthoc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
