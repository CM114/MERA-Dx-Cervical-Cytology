"""M5: Frozen-representation boundary uncertainty audit.

Builds core reference library from train (3-seed correct + high local purity),
computes boundary scores for dev, evaluates M5+M7 vs M7 alone.

NO model training. NO parameter updates. Pure diagnostic + routing enhancement.
"""

import sys, json, math, argparse
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, average_precision_score


BOUNDARY_PAIRS = {"low": (1, 2), "high": (3, 4)}  # ASC-US/LSIL, ASC-H/HSIL
CLASS_NAMES = ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="M5 boundary uncertainty audit")
    p.add_argument("--features_train", type=Path, required=True,
                   help="Train features .npy from frozen audit (N_train, 512)")
    p.add_argument("--features_dev", type=Path, required=True,
                   help="Dev features .npy from frozen audit (N_dev, 512)")
    p.add_argument("--labels_train", type=Path, required=True,
                   help="Train labels .npy")
    p.add_argument("--labels_dev", type=Path, required=True,
                   help="Dev labels .npy")
    p.add_argument("--dev_preds_csv", type=Path, required=True,
                   help="M0 dev_predictions.csv for M7 baseline comparison")
    p.add_argument("--m7_decisions_csv", type=Path, default=None,
                   help="M7 dev_decisions.csv for M5+M7 evaluation")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def build_core_reference(features, labels, n_neighbors=20):
    """Identify core reference samples per class: high local purity.

    Returns dict: class_idx -> (reference_features, reference_labels).
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    # L2-normalise
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

    refs = {}
    for c in range(5):
        mask = labels == c
        if mask.sum() < n_neighbors:
            refs[c] = (fn[mask], labels[mask])
            continue
        fc = fn[mask]
        # Local purity: fraction of k-NN with same label
        dist = cdist(fc, fc, metric="euclidean")
        k = min(n_neighbors + 1, len(fc))
        purity = np.array([
            (labels[mask][dist[i].argsort()[1:k]] == c).mean()
            for i in range(len(fc))
        ])
        # Keep top 70% by purity as core
        threshold = np.percentile(purity, 30)
        core_mask = purity >= threshold
        refs[c] = (fc[core_mask], labels[mask][core_mask])
        print(f"  Class {c} ({CLASS_NAMES[c]}): {core_mask.sum()}/{len(fc)} core samples", flush=True)
    return refs


def boundary_scores(features, refs):
    """Compute boundary uncertainty scores for each sample.

    Returns dict with:
      u_pair: adjacent-class probability ambiguity (from features, not predictions)
      u_between: distance-based ambiguity between boundary classes
      u_far: distance to nearest core reference
    """
    features = np.asarray(features, dtype=np.float64)
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    n = len(fn)

    # u_far: min distance to any core reference centroid
    centroids = {}
    for c in range(5):
        if refs[c][0].shape[0] > 0:
            centroids[c] = refs[c][0].mean(axis=0)
    dist_to_centroid = np.full((n, 5), np.inf)
    for c, ct in centroids.items():
        dist_to_centroid[:, c] = np.linalg.norm(fn - ct, axis=1)
    u_far = dist_to_centroid.min(axis=1)

    # u_between: for boundary pairs, compute distance ratio to pair centroids
    u_between = np.full(n, np.nan)
    for pair_name, (a, b) in BOUNDARY_PAIRS.items():
        if a in centroids and b in centroids:
            da = np.linalg.norm(fn - centroids[a], axis=1)
            db = np.linalg.norm(fn - centroids[b], axis=1)
            # High when sample is equidistant from both (ambiguous)
            d_min = np.minimum(da, db)
            d_sum = da + db + 1e-12
            ambiguity = 1.0 - np.abs(da - db) / d_sum  # [0,1], 1=equidistant
            mask = np.isnan(u_between)
            u_between[mask] = ambiguity[mask]

    # u_pair: use distance-based proxy for conditional pair entropy
    # High when the two closest centroids are a boundary pair
    top2_idx = np.argpartition(dist_to_centroid, 1, axis=1)[:, :2]
    u_pair = np.zeros(n)
    for i in range(n):
        t2 = sorted(top2_idx[i])
        if t2 == [1, 2] or t2 == [3, 4]:  # boundary pair
            d0, d1 = dist_to_centroid[i, t2[0]], dist_to_centroid[i, t2[1]]
            u_pair[i] = 1.0 - abs(d0 - d1) / (d0 + d1 + 1e-12)

    return {
        "u_pair": u_pair,
        "u_between": u_between,
        "u_far": u_far,
        "dist_to_centroid": dist_to_centroid,
    }


def rank_normalize(arr):
    """Rank-normalise to [0,1]."""
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isnan(arr)
    valid = arr[~mask]
    if len(valid) == 0:
        return np.zeros_like(arr)
    ranks = np.zeros(len(arr))
    ranks[~mask] = (np.searchsorted(np.sort(valid), valid) + 1) / (len(valid) + 1)
    ranks[mask] = 0.0
    return ranks


def compute_m5_score(scores):
    """Combine boundary scores via rank-normalised equal-weight mean."""
    components = []
    for key in ["u_pair", "u_between", "u_far"]:
        if key in scores and not np.all(np.isnan(scores[key])):
            components.append(rank_normalize(scores[key]))
    if not components:
        return np.zeros(len(next(iter(scores.values()))))
    return np.mean(components, axis=0)


def evaluate_m5_vs_m7(m5_score, labels, dev_preds, m7_rejected=None):
    """Compare M5+M7 vs M7 alone on boundary error detection."""
    results = {}

    for pair_name, (a, b) in BOUNDARY_PAIRS.items():
        # Boundary samples: true label is one of the pair
        pair_mask = np.isin(labels, [a, b])
        if not pair_mask.any():
            continue

        # Error: predicted label is NOT the true label (any class)
        pred_labels = dev_preds["pred_label"].values if "pred_label" in dev_preds.columns \
            else dev_preds[[f"prob_{n}" for n in CLASS_NAMES]].values.argmax(axis=1)
        pair_error = pair_mask & (pred_labels != labels)

        n_pair = pair_mask.sum()
        n_error = pair_error.sum()
        print(f"\n  {pair_name} pair: {n_pair} samples, {n_error} errors ({n_error/n_pair:.3f})", flush=True)

        # M5 score AUROC/AUPRC for detecting boundary errors
        if n_error > 0 and n_error < n_pair:
            y = pair_error[pair_mask].astype(int)
            s = m5_score[pair_mask]
            if not np.all(np.isnan(s)):
                s[np.isnan(s)] = 0.0
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)
                print(f"    M5 boundary score AUROC={auroc:.4f}  AUPRC={auprc:.4f}", flush=True)
                results[f"{pair_name}_auroc"] = float(auroc)
                results[f"{pair_name}_auprc"] = float(auprc)

        # M7 rejection rate on boundary errors
        if m7_rejected is not None:
            m7_caught = m7_rejected[pair_mask] & pair_error[pair_mask]
            m7_recall = m7_caught.sum() / max(n_error, 1)
            print(f"    M7 alone catches {m7_caught.sum()}/{n_error} boundary errors (recall={m7_recall:.3f})", flush=True)
            results[f"{pair_name}_m7_recall"] = float(m7_recall)

            # M5+M7: top k by M5 score among M7 non-rejected
            m7_not_rejected = ~m7_rejected[pair_mask]
            missed_errors = pair_error[pair_mask] & m7_not_rejected
            n_missed = missed_errors.sum()
            if n_missed > 0:
                # Can M5 rank these missed errors higher?
                s_missed = m5_score[pair_mask][missed_errors]
                s_all_not_rejected = m5_score[pair_mask][m7_not_rejected]
                if len(s_all_not_rejected) > 0:
                    threshold = np.percentile(s_all_not_rejected, 80)
                    m5_top20 = s_missed >= threshold
                    print(f"    M5+M7: {n_missed} errors missed by M7, {m5_top20.sum()} in M5 top-20%",
                          flush=True)
                    results[f"{pair_name}_m5_plus_m7_extra"] = int(m5_top20.sum())

    return results


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen features and labels
    print("Loading frozen M0 features...", flush=True)
    ft = np.load(args.features_train)
    fd = np.load(args.features_dev)

    def _load_labels(path):
        path = Path(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            col = "diagnosis_label" if "diagnosis_label" in df.columns else "true_label"
            return df[col].values.astype(np.int64)
        return np.load(path, allow_pickle=True).astype(np.int64)

    lt = _load_labels(args.labels_train)
    ld = _load_labels(args.labels_dev)
    print(f"  Train: {ft.shape}, Dev: {fd.shape}", flush=True)

    dev_preds = pd.read_csv(args.dev_preds_csv)
    m7_rejected = None
    if args.m7_decisions_csv:
        m7_df = pd.read_csv(args.m7_decisions_csv)
        m7_rejected = m7_df["is_rejected"].values

    # Build core reference library from train
    print("\nBuilding core reference library...", flush=True)
    refs = build_core_reference(ft, lt)

    # Compute boundary scores for dev
    print("\nComputing boundary scores...", flush=True)
    scores = boundary_scores(fd, refs)
    m5_score = compute_m5_score(scores)
    print(f"  M5 score: mean={m5_score.mean():.4f} std={m5_score.std():.4f}", flush=True)

    # Save scores
    score_df = pd.DataFrame({
        "u_pair": scores["u_pair"],
        "u_between": scores["u_between"],
        "u_far": scores["u_far"],
        "m5_boundary_score": m5_score,
    })
    score_df.to_csv(out_dir / "dev_boundary_scores.csv", index=False)

    # Evaluate
    print("\n=== M5 vs M7 evaluation ===", flush=True)
    eval_results = evaluate_m5_vs_m7(m5_score, ld, dev_preds, m7_rejected)

    # Summary
    summary = {**eval_results,
               "m5_score_mean": float(m5_score.mean()),
               "m5_score_std": float(m5_score.std()),
               "dev_samples": len(fd)}
    (out_dir / "m5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
