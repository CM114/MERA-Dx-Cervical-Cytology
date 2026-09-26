"""Best-checkpoint pairwise and local-geometry dev-set artifacts.

This module deliberately contains no PyTorch dependency.  It reuses the frozen
boundary-audit definitions so M0-PB1 reports use exactly the established
pairwise and geometry metric contracts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.boundary_audit import (
    PAIR_SPECS,
    binary_metrics,
    geometry_metrics,
    select_pair,
)
from experiments.tbs.metrics import normalize_probability_rows


PAIR_COLUMNS = [
    "pair_name",
    "pairwise_macro_f1",
    "pairwise_balanced_accuracy",
    "pairwise_roc_auc",
    "n_dev",
]


def _validate_labels(y_true):
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional integer labels in [0, 4]")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(
        labels.dtype, np.bool_
    ):
        raise ValueError("labels must be integer labels in [0, 4]")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError("labels must be in [0, 4]")
    return labels.astype(np.int64, copy=False)


def _validate_probabilities(y_prob, expected_samples):
    probabilities = np.asarray(y_prob)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("probabilities must have exactly five columns")
    if probabilities.shape[0] != expected_samples:
        raise ValueError("labels and probabilities must contain the same number of samples")
    if np.iscomplexobj(probabilities) or np.issubdtype(
        probabilities.dtype, np.complexfloating
    ):
        raise ValueError("probabilities must not be complex-valued")
    if not np.issubdtype(probabilities.dtype, np.number) or np.issubdtype(
        probabilities.dtype, np.bool_
    ):
        raise ValueError("probabilities must be numeric")
    return normalize_probability_rows(probabilities)


def _validate_features(features, expected_samples):
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if matrix.shape[0] != expected_samples:
        raise ValueError("labels and features must contain the same number of samples")
    if np.iscomplexobj(matrix) or np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("features must not be complex-valued")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.bool_
    ):
        raise ValueError("features must contain only finite numeric values")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain nonfinite values")
    return matrix


def compute_pairboundary_evaluation(y_true, y_prob, features):
    """Compute locked two-boundary metrics from pooled dev-set features.

    ``y_prob`` is normalized using the existing Stage-1 probability contract;
    all metric definitions are delegated to the frozen boundary audit.
    """

    labels = _validate_labels(y_true)
    probabilities = _validate_probabilities(y_prob, len(labels))
    feature_matrix = _validate_features(features, len(labels))

    pair_rows = []
    geometry_rows = []
    for pair_name, spec in PAIR_SPECS.items():
        pair_features, pair_labels, indices = select_pair(
            feature_matrix, labels, spec["labels"]
        )
        pair_probabilities = probabilities[indices][:, spec["labels"]]
        pair_mass = pair_probabilities.sum(axis=1)
        if np.any(pair_mass <= 0.0):
            raise ValueError("pair probability mass must be positive")
        scores = pair_probabilities[:, 1] / pair_mass
        predictions = (scores >= 0.5).astype(np.int64)
        pair_metric = binary_metrics(pair_labels, scores, predictions)
        geometry = geometry_metrics(pair_features, pair_labels, n_neighbors=5)
        pair_rows.append(
            {
                "pair_name": pair_name,
                "pairwise_macro_f1": pair_metric["macro_f1"],
                "pairwise_balanced_accuracy": pair_metric["balanced_accuracy"],
                "pairwise_roc_auc": pair_metric["roc_auc"],
                "n_dev": int(len(indices)),
            }
        )
        geometry_rows.append({"pair_name": pair_name, **geometry["metrics"]})

    pair_frame = (
        pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
        .sort_values("pair_name")
        .reset_index(drop=True)
    )
    geometry_frame = (
        pd.DataFrame(geometry_rows).sort_values("pair_name").reset_index(drop=True)
    )
    summary = {
        "boundary_composite_macro_f1": float(
            pair_frame["pairwise_macro_f1"].mean()
        ),
        "boundary_mean_local_purity": float(
            geometry_frame["mean_local_purity"].mean()
        ),
    }
    return pair_frame, geometry_frame, summary


def save_pairboundary_evaluation(output_dir, y_true, y_prob, features):
    """Write the two locked boundary CSV artifacts and return their summary."""

    pair_frame, geometry_frame, summary = compute_pairboundary_evaluation(
        y_true, y_prob, features
    )
    output_dir = Path(output_dir)
    pair_frame.to_csv(output_dir / "boundary_pair_metrics.csv", index=False)
    geometry_frame.to_csv(output_dir / "boundary_geometry_metrics.csv", index=False)
    return summary
