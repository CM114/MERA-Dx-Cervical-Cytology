"""Single-parameter global Temperature Scaling for post-hoc probability calibration.

Fits a single T > 0 on a sealed calibration split via NLL minimisation.
Does NOT modify model weights or change argmax predictions.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------


def fit_temperature(logits, labels, init_t=1.0):
    """Fit a single temperature scalar T > 0 by minimizing NLL.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C), float32/64
        Raw model logits.
    labels : np.ndarray, shape (N,), int64
        Ground-truth class indices.
    init_t : float
        Initial temperature value.

    Returns
    -------
    dict with keys ``temperature``, ``nll_before``, ``nll_after``,
    ``converged``, ``n_fev``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D")
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same number of samples")

    nll_before = _nll(logits, labels, 1.0)

    result = minimize(
        lambda t: _nll(logits, labels, float(t[0])),
        x0=[init_t],
        bounds=[(1e-6, 50.0)],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )

    t_opt = float(np.clip(result.x[0], 1e-6, 50.0))
    nll_after = _nll(logits, labels, t_opt)

    return {
        "temperature": t_opt,
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
        "converged": bool(result.success),
        "n_fev": int(result.nfev),
    }


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits, return calibrated probabilities.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
    temperature : float

    Returns
    -------
    np.ndarray, shape (N, C), float64
    """
    logits = np.asarray(logits, dtype=np.float64) / temperature
    return softmax(logits, axis=1)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------


def expected_calibration_error(probs, labels, n_bins=15, bin_strategy="equal_width"):
    """Compute expected calibration error (ECE).

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
    labels : np.ndarray, shape (N,)
    n_bins : int
    bin_strategy : str
        ``"equal_width"``: 15 equal-width bins in [0, 1].
        ``"equal_mass"``: adaptive bins with roughly equal sample counts.

    Returns
    -------
    dict with ``ece``, ``bin_count``, ``bin_stats`` (list of per-bin dicts).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidences = probs.max(axis=1)
    accuracies = probs.argmax(axis=1) == labels

    if bin_strategy == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif bin_strategy == "equal_mass":
        bin_edges = np.quantile(confidences, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown bin_strategy: {bin_strategy}")

    total = len(labels)
    ece_sum = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (
            confidences < bin_edges[i + 1] if i < n_bins - 1 else True
        )
        bin_count = int(mask.sum())
        if bin_count == 0:
            bin_stats.append(
                {
                    "bin": i,
                    "lower": float(bin_edges[i]),
                    "upper": float(bin_edges[i + 1]),
                    "count": 0,
                    "accuracy": float("nan"),
                    "confidence": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue

        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece_sum += (bin_count / total) * gap

        bin_stats.append(
            {
                "bin": i,
                "lower": float(bin_edges[i]),
                "upper": float(bin_edges[i + 1]),
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": gap,
            }
        )

    return {"ece": float(ece_sum), "bin_count": n_bins, "bin_strategy": bin_strategy, "bin_stats": bin_stats}


def stratified_ece_screening(probs, labels, n_bins=15):
    """ECE for Normal vs Abnormal (binary calibration)."""
    p_normal = probs[:, 0]
    p_abnormal = 1.0 - p_normal
    is_abnormal = (labels != 0).astype(np.int64)
    binary_probs = np.column_stack([p_normal, p_abnormal])
    return expected_calibration_error(binary_probs, is_abnormal, n_bins, "equal_width")


def stratified_ece_high_grade(probs, labels, n_bins=15):
    """ECE for high-grade group (ASC-H + HSIL) as binary target, unconditional."""
    p_high = probs[:, 3] + probs[:, 4]  # ASC-H + HSIL
    p_other = 1.0 - p_high
    is_high = np.isin(labels, [3, 4]).astype(np.int64)
    binary_probs = np.column_stack([p_other, p_high])
    return expected_calibration_error(binary_probs, is_high, n_bins, "equal_width")


def per_class_ece(probs, labels, n_bins=10):
    """One-vs-rest classwise ECE using equal-mass bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = probs.shape[1]
    result = {}
    for k in range(n_classes):
        p_k = probs[:, k]
        is_k = (labels == k).astype(np.int64)
        binary = np.column_stack([1.0 - p_k, p_k])
        result[f"class_{k}"] = expected_calibration_error(binary, is_k, n_bins, "equal_mass")
    return result


def brier_score(probs, labels):
    """Multi-class Brier score."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def high_group_brier(probs, labels):
    """Brier score for high-grade group (ASC-H + HSIL)."""
    p_high = np.asarray(probs[:, 3] + probs[:, 4], dtype=np.float64)
    is_high = np.isin(np.asarray(labels, dtype=np.int64), [3, 4]).astype(np.float64)
    return float(np.mean((p_high - is_high) ** 2))


def nll(probs, labels):
    """Negative log-likelihood."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.clip(probs, 1e-15, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels])))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(probs, labels, metric_fn, n_bootstrap=1000, ci=95, random_state=0):
    """Bootstrap confidence interval for a metric.

    Parameters
    ----------
    metric_fn : callable(probs, labels) -> float
    """
    rng = np.random.RandomState(random_state)
    n = len(labels)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        values.append(metric_fn(probs[idx], labels[idx]))
    values = np.array(values)
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": float(np.percentile(values, lower_pct)),
        "ci_upper": float(np.percentile(values, upper_pct)),
    }


# ---------------------------------------------------------------------------
# Reliability plot data
# ---------------------------------------------------------------------------


def reliability_data(probs, labels, n_bins=15):
    """Return (bin_confidences, bin_accuracies) for plotting."""
    confidences = probs.max(axis=1)
    accuracies = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = []
    bin_acc = []
    for i in range(n_bins):
        mask = (confidences >= edges[i]) & (
            confidences < edges[i + 1] if i < n_bins - 1 else True
        )
        if mask.sum() == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
        else:
            bin_conf.append(float(confidences[mask].mean()))
            bin_acc.append(float(accuracies[mask].mean()))
    return bin_conf, bin_acc


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _nll(logits, labels, temperature):
    """Compute mean NLL for given temperature."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-12)
    log_probs = log_softmax(scaled, axis=1)
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))
