"""M7-posthoc: Minimum-expected-risk decision with asymmetric cost matrix.

Operates on M6-calibrated probabilities. Does NOT modify model weights.
Thresholds are fit on calibration only; dev is for validation.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed cost matrix C(y, a): rows=true label, cols=predicted action
# Indices: 0=Normal, 1=ASC-US, 2=LSIL, 3=ASC-H, 4=HSIL
# ---------------------------------------------------------------------------

COST_MATRIX = np.array(
    [
        [0, 1, 1, 2, 3],   # Normal
        [2, 0, 1, 2, 3],   # ASC-US
        [3, 1, 0, 2, 3],   # LSIL
        [5, 4, 3, 0, 1],   # ASC-H
        [6, 5, 4, 2, 0],   # HSIL
    ],
    dtype=np.float64,
)

CLASS_NAMES = ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]

# Action constants
ACTION_ACCEPT = "accept"
ACTION_REJECT = "reject"


def expected_risk(probs, cost_matrix=None):
    """Compute expected risk R(a|x) for each action a.

    R(a|x) = Σ_y C(y,a) × P(y|x)

    Parameters
    ----------
    probs : np.ndarray, shape (N, 5)
        Calibrated probabilities (M6 output).
    cost_matrix : np.ndarray, shape (5, 5), optional

    Returns
    -------
    np.ndarray, shape (N, 5) — risk for each action.
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    probs = np.asarray(probs, dtype=np.float64)
    return probs @ cost_matrix  # (N,5) @ (5,5) → (N,5)


def decide(risks, labels, tau_normal=0.5, tau_abnormal=0.8):
    """Apply threshold-based decision rules.

    Parameters
    ----------
    risks : np.ndarray, shape (N, 5)
        Expected risk for each action.
    labels : np.ndarray, shape (N,)
        True labels (for audit, not for decision).
    tau_normal : float
        Risk threshold for accepting Normal.
    tau_abnormal : float
        Risk threshold for accepting any single abnormal class.

    Returns
    -------
    pd.DataFrame with columns: true_label, min_risk_action, min_risk_value,
    accept_normal_r, accept_abnormal_r, action, prediction_set, is_rejected.
    """
    n = len(labels)
    min_action = risks.argmin(axis=1)  # 0-4
    min_risk = risks.min(axis=1)

    rows = []
    for i in range(n):
        action_idx = int(min_action[i])
        risk_val = float(min_risk[i])

        if action_idx == 0:  # Normal
            if risk_val <= tau_normal:
                action = ACTION_ACCEPT
                pred_set = "Normal"
            else:
                action = ACTION_REJECT
                pred_set = "{ASC-US, LSIL}"  # low-grade ambiguity
        elif action_idx in (1, 2):  # ASC-US, LSIL
            if risk_val <= tau_abnormal:
                action = ACTION_ACCEPT
                pred_set = CLASS_NAMES[action_idx]
            else:
                action = ACTION_REJECT
                pred_set = "{ASC-US, LSIL}"
        else:  # ASC-H, HSIL
            if risk_val <= tau_abnormal:
                action = ACTION_ACCEPT
                pred_set = CLASS_NAMES[action_idx]
            else:
                action = ACTION_REJECT
                pred_set = "{ASC-H, HSIL}"

        rows.append(
            {
                "true_label": int(labels[i]),
                "true_name": CLASS_NAMES[int(labels[i])],
                "min_risk_action": CLASS_NAMES[action_idx],
                "min_risk_action_idx": action_idx,
                "min_risk_value": risk_val,
                "action": action,
                "prediction_set": pred_set,
                "is_rejected": action == ACTION_REJECT,
            }
        )

    return pd.DataFrame(rows)


def compute_metrics(decisions):
    """Compute M7 evaluation metrics from decision DataFrame.

    Parameters
    ----------
    decisions : pd.DataFrame
        Output of ``decide()``.

    Returns
    -------
    dict with keys: coverage, rejection_rate, high_grade_undercall_rate,
    normal_specificity, per_class_accept_rate, confusion_like.
    """
    total = len(decisions)
    rejected = decisions["is_rejected"].sum()
    accepted = total - rejected
    coverage = float(accepted / total) if total > 0 else 0.0
    rejection_rate = float(rejected / total) if total > 0 else 0.0

    # High-grade undercall rate: among accepted samples where true is ASC-H/HSIL,
    # how many were predicted as Normal/ASC-US/LSIL?
    accepted_df = decisions[~decisions["is_rejected"]]
    high_grade_mask = accepted_df["true_label"].isin([3, 4])
    high_grade_accepted = accepted_df[high_grade_mask]
    n_high = len(high_grade_accepted)
    if n_high > 0:
        undercall = high_grade_accepted["min_risk_action_idx"].isin([0, 1, 2]).sum()
        high_grade_undercall_rate = float(undercall / n_high)
    else:
        high_grade_undercall_rate = float("nan")

    # Normal specificity: among accepted samples where true is Normal,
    # how many were correctly accepted as Normal?
    normal_accepted = accepted_df[accepted_df["true_label"] == 0]
    n_normal = len(normal_accepted)
    normal_specificity = (
        float((normal_accepted["min_risk_action_idx"] == 0).sum() / n_normal)
        if n_normal > 0
        else float("nan")
    )

    # Per-class accept rate
    per_class_accept = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        subset = decisions[decisions["true_label"] == cls_idx]
        if len(subset) > 0:
            per_class_accept[cls_name] = float((~subset["is_rejected"]).mean())
        else:
            per_class_accept[cls_name] = float("nan")

    return {
        "total": total,
        "accepted": int(accepted),
        "rejected": int(rejected),
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "high_grade_undercall_rate": high_grade_undercall_rate,
        "normal_specificity": normal_specificity,
        "per_class_accept_rate": per_class_accept,
    }


def search_thresholds(risks, labels, tau_normal_range=None, tau_abnormal_range=None,
                      min_normal_specificity=0.90, max_undercall=0.02):
    """Grid search over thresholds on calibration set.

    Prioritises high-grade undercall rate over coverage.

    Parameters
    ----------
    risks : np.ndarray (N, 5)
    labels : np.ndarray (N,)
    tau_normal_range : array-like
    tau_abnormal_range : array-like
    min_normal_specificity : float
    max_undercall : float

    Returns
    -------
    dict with keys: tau_normal, tau_abnormal, metrics, all_results.
    """
    if tau_normal_range is None:
        tau_normal_range = np.linspace(0.1, 2.0, 20)
    if tau_abnormal_range is None:
        tau_abnormal_range = np.linspace(0.1, 3.0, 30)

    all_results = []
    for tn in tau_normal_range:
        for ta in tau_abnormal_range:
            dec = decide(risks, labels, tau_normal=float(tn), tau_abnormal=float(ta))
            m = compute_metrics(dec)
            m["tau_normal"] = float(tn)
            m["tau_abnormal"] = float(ta)
            all_results.append(m)

    df = pd.DataFrame(all_results)

    # Filter by constraints
    feasible = df[
        (df["normal_specificity"] >= min_normal_specificity)
        & (df["high_grade_undercall_rate"] <= max_undercall)
    ]

    if feasible.empty:
        # Relax: keep highest normal specificity, then lowest undercall
        best = df.loc[df["high_grade_undercall_rate"].idxmin()]
        relaxed = True
    else:
        # Choose highest coverage among feasible
        best = feasible.loc[feasible["coverage"].idxmax()]
        relaxed = False

    best_idx = int(best.name) if hasattr(best, "name") else 0

    return {
        "tau_normal": float(best["tau_normal"]),
        "tau_abnormal": float(best["tau_abnormal"]),
        "thresholds_feasible": not relaxed,
        "constraints": {
            "min_normal_specificity": min_normal_specificity,
            "max_undercall": max_undercall,
        },
        "metrics": {
            k: float(best[k]) if not isinstance(best[k], str) and not math.isnan(float(best[k])) else None
            for k in ["coverage", "rejection_rate", "high_grade_undercall_rate", "normal_specificity"]
        },
        "all_results": all_results,
    }


def risk_coverage_curve(risks, labels, n_tau=50):
    """Compute risk-coverage curve by varying tau_abnormal (fixed tau_normal).

    Returns list of (coverage, high_grade_undercall_rate) pairs.
    """
    tau_vals = np.linspace(0.05, 3.0, n_tau)
    curve = []
    for ta in tau_vals:
        dec = decide(risks, labels, tau_normal=0.5, tau_abnormal=float(ta))
        m = compute_metrics(dec)
        curve.append(
            {
                "tau_abnormal": float(ta),
                "coverage": m["coverage"],
                "rejection_rate": m["rejection_rate"],
                "high_grade_undercall_rate": m["high_grade_undercall_rate"],
            }
        )
    return curve
