"""Locked five-class linear probe for frozen TBS5 feature diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.tbs.boundary_audit import PAIR_SPECS, binary_metrics
from experiments.tbs.metrics import compute_stage1_metrics, normalize_probability_rows


EXPECTED_LABELS = np.arange(5, dtype=np.int64)
COMPARISON_METRICS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_macro_f1",
    "high_grade_macro_f1",
    "balanced_accuracy",
    "macro_auc",
    "nll",
    "brier",
    "ece",
    "low_grade_balanced_accuracy",
    "low_grade_roc_auc",
    "high_grade_balanced_accuracy",
    "high_grade_roc_auc",
)
DECISION_THRESHOLDS = {
    "macro_f1": 0.005,
    "abnormal_macro_f1": 0.005,
    "low_grade_macro_f1": 0.0,
    "high_grade_macro_f1": 0.0,
}


def _feature_matrix(values, name):
    array = np.asarray(values)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return array.astype(np.float64, copy=False)


def _five_class_labels(values, expected_length, name, require_all=True):
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must be one-dimensional and match the samples")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite integer labels")
        if not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain integer labels")
    array = array.astype(np.int64, copy=False)
    if not np.isin(array, EXPECTED_LABELS).all():
        raise ValueError(f"{name} must contain labels 0 through 4")
    if require_all and not np.array_equal(np.unique(array), EXPECTED_LABELS):
        raise ValueError(f"{name} must contain all five classes")
    return array


def validate_probe_arrays(train_features, dev_features, train_labels, dev_labels):
    """Validate and normalize the shared M0/PB1 probe inputs."""

    train_x = _feature_matrix(train_features, "train_features")
    dev_x = _feature_matrix(dev_features, "dev_features")
    if train_x.shape[1] != dev_x.shape[1]:
        raise ValueError("train and dev features must have the same dimension")
    train_y = _five_class_labels(train_labels, len(train_x), "train_labels")
    dev_y = _five_class_labels(dev_labels, len(dev_x), "dev_labels")
    return train_x, dev_x, train_y, dev_y


def fit_five_class_probe(train_features, train_labels, dev_features):
    """Fit the preregistered train-only probe and return dev probabilities."""

    train_x = _feature_matrix(train_features, "train_features")
    dev_x = _feature_matrix(dev_features, "dev_features")
    if train_x.shape[1] != dev_x.shape[1]:
        raise ValueError("train and dev features must have the same dimension")
    train_y = _five_class_labels(train_labels, len(train_x), "train_labels")

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
            class_weight=None,
            random_state=0,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    logistic = estimator.named_steps["logisticregression"]
    raw_iterations = np.asarray(logistic.n_iter_, dtype=np.int64).reshape(-1)
    if convergence_messages or np.any(raw_iterations >= logistic.max_iter):
        detail = "; ".join(convergence_messages) or f"n_iter={raw_iterations.tolist()}"
        raise RuntimeError(f"locked linear probe did not converge: {detail}")
    if not np.array_equal(logistic.classes_, EXPECTED_LABELS):
        raise RuntimeError("locked linear probe class order is not 0 through 4")

    # Multinomial lbfgs reports one global count in recent sklearn versions;
    # normalize the diagnostic shape without changing the fitted estimator.
    iterations = (
        np.repeat(raw_iterations, len(EXPECTED_LABELS))
        if raw_iterations.size == 1
        else raw_iterations
    )
    if iterations.shape != (len(EXPECTED_LABELS),):
        raise RuntimeError(f"unexpected probe iteration shape: {iterations.shape}")
    probabilities = normalize_probability_rows(estimator.predict_proba(dev_x))
    return {
        "estimator": estimator,
        "probabilities": probabilities,
        "iterations": iterations,
    }


def _pair_metrics(labels, probabilities, pair_name, pair_labels):
    mask = np.isin(labels, pair_labels)
    pair_true = (labels[mask] == pair_labels[1]).astype(np.int64)
    if set(pair_true.tolist()) != {0, 1}:
        raise ValueError(f"{pair_name} requires both classes")
    pair_probabilities = probabilities[mask][:, pair_labels]
    pair_mass = pair_probabilities.sum(axis=1)
    if np.any(pair_mass <= 0.0):
        raise ValueError(f"{pair_name} probability mass must be positive")
    scores = pair_probabilities[:, 1] / pair_mass
    predictions = (scores >= 0.5).astype(np.int64)
    return binary_metrics(pair_true, scores, predictions)


def compute_probe_metrics(labels, probabilities):
    """Compute locked whole-task, abnormal-class, and pair-restricted metrics."""

    probabilities = normalize_probability_rows(probabilities)
    labels = _five_class_labels(labels, len(probabilities), "labels")
    metrics = compute_stage1_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    abnormal_f1 = [
        f1_score(labels == label, predictions == label, zero_division=0)
        for label in range(1, 5)
    ]
    metrics["abnormal_macro_f1"] = float(np.mean(abnormal_f1))

    for pair_name, spec in PAIR_SPECS.items():
        pair = _pair_metrics(labels, probabilities, pair_name, spec["labels"])
        metrics[f"{pair_name}_macro_f1"] = pair["macro_f1"]
        metrics[f"{pair_name}_balanced_accuracy"] = pair["balanced_accuracy"]
        metrics[f"{pair_name}_roc_auc"] = pair["roc_auc"]
    return metrics


def compare_probe_metrics(m0_metrics, pb1_metrics):
    """Create a fixed PB1-minus-M0 metric comparison table."""

    missing = [
        metric
        for metric in COMPARISON_METRICS
        if metric not in m0_metrics or metric not in pb1_metrics
    ]
    if missing:
        raise ValueError(f"probe metrics are missing comparison fields: {missing}")
    rows = []
    for metric in COMPARISON_METRICS:
        m0_value = float(m0_metrics[metric])
        pb1_value = float(pb1_metrics[metric])
        if not np.isfinite([m0_value, pb1_value]).all():
            raise ValueError(f"comparison metric must be finite: {metric}")
        rows.append(
            {
                "metric": metric,
                "m0": m0_value,
                "pb1": pb1_value,
                "delta": pb1_value - m0_value,
            }
        )
    return pd.DataFrame(rows)


def route_probe_decision(comparison):
    """Apply the locked four-gate diagnostic route; never promote PB1."""

    if not isinstance(comparison, pd.DataFrame):
        raise ValueError("comparison must be a pandas DataFrame")
    required_columns = {"metric", "delta"}
    if not required_columns.issubset(comparison.columns):
        raise ValueError("comparison is missing metric or delta columns")
    if comparison["metric"].duplicated().any():
        raise ValueError("comparison metrics must be unique")
    delta = comparison.set_index("metric")["delta"]
    missing = set(DECISION_THRESHOLDS) - set(delta.index)
    if missing:
        raise ValueError(f"comparison is missing decision metrics: {sorted(missing)}")

    gates = {}
    for metric, threshold in DECISION_THRESHOLDS.items():
        value = float(delta[metric])
        if not np.isfinite(value):
            raise ValueError(f"decision delta must be finite: {metric}")
        gates[metric] = {
            "delta": value,
            "threshold": float(threshold),
            "passed": bool(value >= threshold),
        }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "decision": (
            "HEAD_ALIGNMENT_CANDIDATE"
            if passed
            else "REPRESENTATION_NOT_USEFUL_CLOSE_GLOBAL_PB"
        ),
        "all_gates_passed": bool(passed),
        "gates": gates,
        "pb1_promoted": False,
    }
