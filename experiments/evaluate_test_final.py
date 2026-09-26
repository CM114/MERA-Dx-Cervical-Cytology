"""Final test evaluation: M0 → M6 → M7 → M7-v2 (M5 enhanced).

Single pass. All thresholds frozen. No feedback to method design.
"""

import sys, json, math, argparse, datetime
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

from experiments.tbs.models import build_stage1_model
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.temperature_scaling import apply_temperature, fit_temperature
from experiments.tbs.risk_control import expected_risk, decide as m7_decide, COST_MATRIX


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Final test evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--train_features", type=Path, required=True,
                   help="Train .npy for M5 core reference")
    p.add_argument("--train_labels", type=Path, required=True)
    p.add_argument("--cal_features", type=Path, required=True,
                   help="Calibration .npy for M5 threshold")
    p.add_argument("--cal_labels", type=Path, required=True)
    p.add_argument("--cal_preds_csv", type=Path, required=True,
                   help="M6 calibration_predictions.csv")
    p.add_argument("--cal_m7_csv", type=Path, required=True,
                   help="M7 calibration_decisions.csv")
    p.add_argument("--cal_m5_csv", type=Path, required=True,
                   help="M5 calibration scores CSV")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    return p.parse_args(argv)


def _load_labels(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        for c in ["diagnosis_label", "true_label"]:
            if c in df.columns:
                return df[c].values.astype(np.int64)
    return np.load(path, allow_pickle=True).astype(np.int64)


@torch.no_grad()
def extract_test_features(model, loader, device):
    features, labels, probs, logits_list = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        lbls = batch["diagnosis_label"]
        with torch.cuda.amp.autocast(enabled=True):
            output = model(images)
        features.append(output["features"].cpu().numpy())
        labels.append(lbls.numpy())
        probs.append(output["diagnosis_probs"].cpu().numpy())
        logits_list.append(output["diagnosis_logits"].cpu().numpy())
    return (np.concatenate(features), np.concatenate(labels),
            np.concatenate(probs), np.concatenate(logits_list))


def compute_m5_test(features_test, features_train, labels_train, labels_test,
                    y_pred_test, m7_rejected, cal_m5, cal_m7_rej, cal_labels):
    """M5-Directional scoring with calibration-frozen thresholds."""
    fn_test = features_test / (np.linalg.norm(features_test, axis=1, keepdims=True) + 1e-12)
    fn_train = features_train / (np.linalg.norm(features_train, axis=1, keepdims=True) + 1e-12)

    # Core centroids from train
    centroids = {}
    for c in range(5):
        mask = labels_train == c
        centroids[c] = fn_train[mask].mean(axis=0)

    # Class-conditional percentile
    m5_cc = np.zeros(len(fn_test))
    for pred_c in range(5):
        pmask = y_pred_test == pred_c
        if pmask.sum() == 0:
            continue
        d_self = np.linalg.norm(fn_test[pmask] - centroids[pred_c], axis=1)
        typicality = 1.0 / (d_self + 0.01)
        ranks = np.searchsorted(np.sort(typicality), typicality).astype(float)
        m5_cc[pmask] = 1.0 - ranks / (pmask.sum() + 1)

    # Direction-stratified
    d1 = np.linalg.norm(fn_test - centroids[1], axis=1)
    d2 = np.linalg.norm(fn_test - centroids[2], axis=1)
    ratio_12 = d1 / (d2 + 1e-12)
    boundary_score = 1.0 - np.abs(d1 - d2) / (d1 + d2 + 1e-12)

    au_risk = np.zeros(len(fn_test))
    au_mask = y_pred_test == 1
    if au_mask.any():
        au_risk[au_mask] = boundary_score[au_mask] * np.clip(ratio_12[au_mask], 1.0, 2.0)

    lsil_risk = np.zeros(len(fn_test))
    lsil_mask = y_pred_test == 2
    if lsil_mask.any():
        lsil_risk[lsil_mask] = boundary_score[lsil_mask] * np.clip(1.0/(ratio_12[lsil_mask]+1e-12), 1.0, 2.0)

    m5_strat = au_risk + lsil_risk
    valid = m5_strat > 0
    if valid.any():
        ranks = np.zeros(len(m5_strat))
        ranks[valid] = np.searchsorted(np.sort(m5_strat[valid]), m5_strat[valid]).astype(float)
        ranks[valid] /= (valid.sum() + 1)
        m5_strat = ranks

    m5_score = (m5_cc + m5_strat) / 2.0

    # Thresholds from calibration
    cal_low_mask = np.isin(cal_labels, [1, 2])
    cal_m7_acc = ~cal_m7_rej
    cal_subset = cal_low_mask & cal_m7_acc
    cal_m5_subset = cal_m5["m5_final"].values[cal_subset] if "m5_final" in cal_m5.columns else cal_m5["m5_boundary_score"].values[cal_subset]
    t10 = np.percentile(cal_m5_subset, 90) if cal_subset.sum() > 0 else 1.0
    t20 = np.percentile(cal_m5_subset, 80) if cal_subset.sum() > 0 else 1.0

    # Apply M5 routing
    low_pred_mask = np.isin(y_pred_test, [1, 2])
    m7_acc_test = ~m7_rejected
    m5_eligible = low_pred_mask & m7_acc_test
    m5_flagged_10 = m5_eligible & (m5_score >= t10)
    m5_flagged_20 = m5_eligible & (m5_score >= t20)

    return m5_score, m5_flagged_10, m5_flagged_20, {"t10": float(t10), "t20": float(t20)}


def compute_metrics(y_true, y_pred, m7_rej, m5_flagged, probs):
    """Compute all evaluation metrics."""
    n = len(y_true)
    m7_reviewed = m7_rej
    m7v2_reviewed = m7_rej | m5_flagged

    # Low-pair errors
    low_mask = np.isin(y_true, [1, 2])
    low_pred_mask = np.isin(y_pred, [1, 2])
    low_err = low_mask & (y_pred != y_true)

    # M7 accepted low errors
    m7_acc_low_err = low_err & ~m7_rej
    m7v2_caught = m7_acc_low_err & m5_flagged
    m7_caught = low_err & m7_rej

    # High-grade undercall
    hg_mask = np.isin(y_true, [3, 4])
    hg_uc_m7 = (y_pred != y_true) & hg_mask & ~m7_rej
    hg_uc_m7v2 = (y_pred != y_true) & hg_mask & ~m7v2_reviewed

    # Directional breakdown
    a_to_l = (y_true == 1) & (y_pred == 2)
    l_to_a = (y_true == 2) & (y_pred == 1)

    metrics = {
        "n_samples": n,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "m7_review_rate": float(m7_reviewed.mean()),
        "m7v2_review_rate": float(m7v2_reviewed.mean()),
        "m5_extra_review_rate": float(m5_flagged.mean()),
        "low_pair_errors": int(low_err.sum()),
        "m7_caught_low": int(m7_caught.sum()),
        "m7v2_caught_low": int((m7_rej | m5_flagged)[low_err].sum()),
        "m5_extra_caught_low": int(m7v2_caught.sum()),
        "m5_flagged_correct_low": int((m5_flagged & low_mask & (y_pred == y_true)).sum()),
        "hg_undercall_m7": int(hg_uc_m7.sum()),
        "hg_undercall_m7v2": int(hg_uc_m7v2.sum()),
        "a_to_l_errors": int(a_to_l.sum()),
        "a_to_l_m7_caught": int((a_to_l & m7_rej).sum()),
        "a_to_l_m7v2_caught": int((a_to_l & m7v2_reviewed).sum()),
        "l_to_a_errors": int(l_to_a.sum()),
        "l_to_a_m7_caught": int((l_to_a & m7_rej).sum()),
        "l_to_a_m7v2_caught": int((l_to_a & m7v2_reviewed).sum()),
    }
    return metrics


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"seed{args.seed}"
    print(f"=== TEST FINAL EVALUATION {tag} ===", flush=True)

    # Load model
    model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=True)
    model.to(device)
    model.eval()

    # Data
    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    test_ds = XUDataTBS5Dataset(args.test_csv, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Extract test features + predictions
    print("Extracting test features...", flush=True)
    feats_test, labels_test, probs_raw, logits_raw = extract_test_features(model, test_loader, device)
    y_pred_test = probs_raw.argmax(axis=1)

    # M6: Temperature scaling (use pre-fitted T from calibration)
    cal_preds = pd.read_csv(args.cal_preds_csv)
    # Fit T from calibration if not already done
    cal_logits = cal_preds[[c for c in cal_preds.columns if "logit" in c or "prob_raw" in c]].values
    if cal_logits.shape[1] != 5:
        cal_logits = None
    # M6: Temperature scaling from frozen calibration fit
    t_path = Path(f"results/tbs5/calibration/m6_temperature_scaling/seed{args.seed}/temperature.json")
    t_data = json.loads(t_path.read_text())
    T = t_data["temperature"]
    print(f"  T={T:.4f}", flush=True)
    probs_cal = apply_temperature(logits_raw, T)

    # M7: Risk control (frozen thresholds from calibration)
    risks = expected_risk(probs_cal, COST_MATRIX)
    m7_path = Path(f"results/tbs5/calibration/m7_risk_control/seed{args.seed}/decision_rules.json")
    m7_data = json.loads(m7_path.read_text())
    tau_normal = m7_data["tau_normal"]
    tau_abnormal = m7_data["tau_abnormal"]
    print(f"  tau_normal={tau_normal:.4f} tau_abnormal={tau_abnormal:.4f}", flush=True)

    m7_decisions = m7_decide(risks, labels_test, tau_normal, tau_abnormal)
    m7_rejected = m7_decisions["is_rejected"].values

    # Load frozen M5 calibration data
    cal_labels_arr = _load_labels(args.cal_labels)
    cal_m7 = pd.read_csv(args.cal_m7_csv)
    cal_m7_rej = cal_m7["is_rejected"].values
    cal_m5 = pd.read_csv(args.cal_m5_csv)

    # M5 on test
    ft_train = np.load(args.train_features)
    lt_train = _load_labels(args.train_labels)
    m5_score, m5_flag_10, m5_flag_20, thresholds = compute_m5_test(
        feats_test, ft_train, lt_train, labels_test, y_pred_test,
        m7_rejected, cal_m5, cal_m7_rej, cal_labels_arr,
    )

    # Compute metrics
    metrics_m7 = compute_metrics(labels_test, y_pred_test, m7_rejected, np.zeros(len(labels_test), dtype=bool), probs_raw)
    metrics_m7v2_10 = compute_metrics(labels_test, y_pred_test, m7_rejected, m5_flag_10, probs_raw)
    metrics_m7v2_20 = compute_metrics(labels_test, y_pred_test, m7_rejected, m5_flag_20, probs_raw)

    # Report
    print(f"\n{'Metric':<35s} {'M7':>10s} {'M7-v2(B10)':>10s} {'M7-v2(B20)':>10s}", flush=True)
    print("-" * 70, flush=True)
    for key in ["macro_f1", "m7_review_rate", "low_pair_errors",
                "m7_caught_low", "m5_extra_caught_low", "a_to_l_errors",
                "a_to_l_m7_caught", "a_to_l_m7v2_caught", "l_to_a_errors",
                "l_to_a_m7_caught", "l_to_a_m7v2_caught", "hg_undercall_m7", "hg_undercall_m7v2"]:
        v_m7 = metrics_m7.get(key, 0)
        v_10 = metrics_m7v2_10.get(key, 0)
        v_20 = metrics_m7v2_20.get(key, 0)
        if isinstance(v_m7, float):
            print(f"{key:<35s} {v_m7:>10.4f} {v_10:>10.4f} {v_20:>10.4f}", flush=True)
        else:
            print(f"{key:<35s} {v_m7:>10d} {v_10:>10d} {v_20:>10d}", flush=True)

    # Save
    result = {
        "tag": tag,
        "thresholds": thresholds,
        "tau_normal": tau_normal,
        "tau_abnormal": tau_abnormal,
        "temperature": T,
        "m7": metrics_m7,
        "m7v2_b10": metrics_m7v2_10,
        "m7v2_b20": metrics_m7v2_20,
        "generated_at": str(datetime.datetime.now()),
    }
    (out_dir / f"test_results_{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"\nResults: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
