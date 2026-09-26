"""Deterministic feature-space diagnostics for the two locked TBS5 boundaries.

This module intentionally has no PyTorch or image dependencies.  It operates on
already-extracted frozen features so its statistical contracts can be tested on
any machine with NumPy, pandas, and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, normalize


PAIR_SPECS = {
    "low_grade": {"labels": (1, 2), "names": ("ASC-US", "LSIL")},
    "high_grade": {"labels": (3, 4), "names": ("ASC-H", "HSIL")},
}


def _as_feature_matrix(features, name):
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def _as_binary_labels(labels, expected_length, name):
    array = np.asarray(labels)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must be one-dimensional and match the samples")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    if set(array.tolist()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return array.astype(np.int64, copy=False)


def select_pair(features, labels, pair_labels):
    """Select one two-class boundary and map its second label to binary one."""

    features = _as_feature_matrix(features, "features")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must be one-dimensional and match features")
    if len(pair_labels) != 2 or pair_labels[0] == pair_labels[1]:
        raise ValueError("pair_labels must contain two distinct labels")
    first_label, second_label = pair_labels
    indices = np.flatnonzero(np.isin(labels, pair_labels))
    if indices.size == 0:
        raise ValueError(f"no samples found for pair labels {tuple(pair_labels)}")
    binary_labels = (labels[indices] == second_label).astype(np.int64)
    if set(binary_labels.tolist()) != {0, 1}:
        raise ValueError(f"pair labels {tuple(pair_labels)} must both be present")
    return features[indices], binary_labels, indices


def binary_metrics(y_true, scores, predictions):
    """Return fixed binary metrics after strict shape and finite-value checks."""

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    y_true = _as_binary_labels(y_true, len(y_true), "y_true")
    if predictions.ndim != 1 or len(predictions) != len(y_true):
        raise ValueError("predictions must be one-dimensional and match y_true")
    if not np.isin(predictions, (0, 1)).all():
        raise ValueError("predictions must contain only 0 and 1")
    if scores.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("scores must be one-dimensional and match y_true")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def fit_linear_probe(train_x, train_y, dev_x, dev_y):
    """Fit the locked train-only linear probe and evaluate it once on dev."""

    train_x = _as_feature_matrix(train_x, "train_x")
    dev_x = _as_feature_matrix(dev_x, "dev_x")
    if train_x.shape[1] != dev_x.shape[1]:
        raise ValueError("train_x and dev_x must have the same feature dimension")
    train_y = _as_binary_labels(train_y, len(train_x), "train_y")
    dev_y = _as_binary_labels(dev_y, len(dev_x), "dev_y")

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        ),
    )
    estimator.fit(train_x, train_y)
    scores = estimator.predict_proba(dev_x)[:, 1]
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "estimator": estimator,
        "scores": scores,
        "predictions": predictions,
        "metrics": binary_metrics(dev_y, scores, predictions),
    }


def fit_knn_probe(train_x, train_y, dev_x, dev_y, n_neighbors=5):
    """Fit the locked cosine-distance k-NN probe on train features."""

    train_x = _as_feature_matrix(train_x, "train_x")
    dev_x = _as_feature_matrix(dev_x, "dev_x")
    if train_x.shape[1] != dev_x.shape[1]:
        raise ValueError("train_x and dev_x must have the same feature dimension")
    train_y = _as_binary_labels(train_y, len(train_x), "train_y")
    dev_y = _as_binary_labels(dev_y, len(dev_x), "dev_y")
    if not isinstance(n_neighbors, int) or not 1 <= n_neighbors <= len(train_x):
        raise ValueError("n_neighbors must be between 1 and the train sample count")

    estimator = make_pipeline(
        Normalizer(norm="l2"),
        KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            weights="distance",
            algorithm="brute",
        ),
    )
    estimator.fit(train_x, train_y)
    scores = estimator.predict_proba(dev_x)[:, 1]
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "estimator": estimator,
        "scores": scores,
        "predictions": predictions,
        "metrics": binary_metrics(dev_y, scores, predictions),
    }


def geometry_metrics(features, labels, n_neighbors=5):
    """Measure normalized class geometry and leave-one-out local purity."""

    features = _as_feature_matrix(features, "features")
    labels = _as_binary_labels(labels, len(features), "labels")
    if len(features) < 3:
        raise ValueError("geometry_metrics requires at least three samples")
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")

    normalized = normalize(features, norm="l2")
    centroids = [normalized[labels == label].mean(axis=0) for label in (0, 1)]
    centroid_distance = float(np.linalg.norm(centroids[0] - centroids[1]))
    class_scatter = []
    for label, centroid in enumerate(centroids):
        deviations = normalized[labels == label] - centroid
        class_scatter.append(float(np.mean(np.sum(deviations * deviations, axis=1))))
    denominator = class_scatter[0] + class_scatter[1]
    fisher_ratio = float(centroid_distance**2 / max(denominator, np.finfo(float).eps))

    distances = cosine_distances(normalized)
    np.fill_diagonal(distances, np.inf)
    effective_neighbors = min(n_neighbors, len(features) - 1)
    neighbor_indices = np.argsort(distances, axis=1)[:, :effective_neighbors]
    local_purity = (labels[neighbor_indices] == labels[:, None]).mean(axis=1)
    silhouette = float(silhouette_score(normalized, labels, metric="cosine"))

    return {
        "metrics": {
            "centroid_distance": centroid_distance,
            "within_class_scatter_0": class_scatter[0],
            "within_class_scatter_1": class_scatter[1],
            "fisher_ratio": fisher_ratio,
            "silhouette_cosine": silhouette,
            "mean_local_purity": float(local_purity.mean()),
            "n_neighbors": int(effective_neighbors),
        },
        "local_purity": local_purity.astype(np.float64, copy=False),
    }


def _validate_pair_name(pair_name):
    if pair_name not in PAIR_SPECS:
        raise ValueError(f"unknown pair_name {pair_name!r}")
    return PAIR_SPECS[pair_name]


def build_seed_sample_frame(
    *,
    seed,
    image_paths: Sequence[str],
    true_labels,
    probabilities,
    pair_name,
    probe_predictions,
    knn_predictions,
    local_purity,
):
    """Build one seed's sample-level table for a single class boundary."""

    spec = _validate_pair_name(pair_name)
    true_labels = np.asarray(true_labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    image_paths = np.asarray([str(path) for path in image_paths], dtype=object)
    if true_labels.ndim != 1 or len(true_labels) != len(image_paths):
        raise ValueError("true_labels and image_paths must have matching lengths")
    if probabilities.shape != (len(image_paths), 5):
        raise ValueError("probabilities must have shape [samples, 5]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    if pd.Index(image_paths).has_duplicates:
        raise ValueError("image_paths must be unique")

    _, binary_labels, indices = select_pair(
        np.zeros((len(true_labels), 1), dtype=np.float32),
        true_labels,
        spec["labels"],
    )
    pair_probabilities = probabilities[indices][:, spec["labels"]]
    pair_mass = pair_probabilities.sum(axis=1)
    if np.any(pair_mass <= 0.0):
        raise ValueError("pair probability mass must be positive")
    m0_scores = pair_probabilities[:, 1] / pair_mass
    m0_predictions = (m0_scores >= 0.5).astype(np.int64)

    probe_predictions = np.asarray(probe_predictions)
    knn_predictions = np.asarray(knn_predictions)
    local_purity = np.asarray(local_purity, dtype=np.float64)
    pair_count = len(indices)
    for values, name in (
        (probe_predictions, "probe_predictions"),
        (knn_predictions, "knn_predictions"),
        (local_purity, "local_purity"),
    ):
        if values.ndim != 1 or len(values) != pair_count:
            raise ValueError(f"{name} must match the selected pair sample count")
    if not np.isin(probe_predictions, (0, 1)).all():
        raise ValueError("probe_predictions must be binary")
    if not np.isin(knn_predictions, (0, 1)).all():
        raise ValueError("knn_predictions must be binary")
    if not np.isfinite(local_purity).all() or np.any((local_purity < 0) | (local_purity > 1)):
        raise ValueError("local_purity must contain finite values in [0, 1]")

    return pd.DataFrame(
        {
            "seed": int(seed),
            "image_path": image_paths[indices],
            "pair_name": pair_name,
            "true_label": true_labels[indices].astype(np.int64),
            "pair_true_binary": binary_labels,
            "m0_pair_score": m0_scores,
            "m0_pair_prediction": m0_predictions,
            "m0_pair_margin": np.abs(m0_scores - 0.5) * 2.0,
            "m0_correct": (m0_predictions == binary_labels).astype(np.int64),
            "probe_prediction": probe_predictions.astype(np.int64),
            "probe_correct": (probe_predictions == binary_labels).astype(np.int64),
            "knn_prediction": knn_predictions.astype(np.int64),
            "knn_correct": (knn_predictions == binary_labels).astype(np.int64),
            "local_purity": local_purity,
        }
    )


def aggregate_sample_frames(frames: Iterable[pd.DataFrame]):
    """Aggregate cross-seed stability while rejecting path or label drift."""

    frames = [frame.copy() for frame in frames]
    if not frames:
        raise ValueError("at least one seed sample frame is required")
    required = {
        "seed",
        "image_path",
        "pair_name",
        "true_label",
        "pair_true_binary",
        "m0_pair_score",
        "m0_pair_prediction",
        "m0_pair_margin",
        "m0_correct",
        "probe_correct",
        "knn_correct",
        "local_purity",
    }
    first_paths = frames[0]["image_path"].astype(str).tolist()
    expected_paths = set(first_paths)
    seen_seeds = set()
    for frame in frames:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"sample frame is missing columns: {sorted(missing)}")
        if frame.empty or frame["image_path"].astype(str).duplicated().any():
            raise ValueError("each seed frame must have unique nonempty image paths")
        seeds = set(frame["seed"].astype(int).tolist())
        if len(seeds) != 1:
            raise ValueError("each sample frame must contain exactly one seed")
        seed = next(iter(seeds))
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed frame: {seed}")
        seen_seeds.add(seed)
        if set(frame["image_path"].astype(str)) != expected_paths:
            raise ValueError("cross-seed sample path drift detected")

    combined = pd.concat(frames, ignore_index=True)
    identity_counts = combined.groupby("image_path")[[
        "pair_name", "true_label", "pair_true_binary"
    ]].nunique()
    if (identity_counts > 1).any(axis=None):
        raise ValueError("cross-seed label or pair drift detected")

    rows = []
    for image_path, group in combined.groupby("image_path", sort=False):
        group = group.sort_values("seed")
        m0_correct_seeds = int(group["m0_correct"].sum())
        rows.append(
            {
                "image_path": str(image_path),
                "pair_name": str(group["pair_name"].iloc[0]),
                "true_label": int(group["true_label"].iloc[0]),
                "pair_true_binary": int(group["pair_true_binary"].iloc[0]),
                "seed_count": int(group["seed"].nunique()),
                "m0_correct_seeds": m0_correct_seeds,
                "probe_correct_seeds": int(group["probe_correct"].sum()),
                "knn_correct_seeds": int(group["knn_correct"].sum()),
                "m0_seed_unstable": bool(group["m0_pair_prediction"].nunique() > 1),
                "m0_persistent_error": bool(m0_correct_seeds == 0),
                "m0_pair_score_mean": float(group["m0_pair_score"].mean()),
                "m0_pair_score_std": float(group["m0_pair_score"].std(ddof=0)),
                "m0_pair_margin_mean": float(group["m0_pair_margin"].mean()),
                "local_purity_mean": float(group["local_purity"].mean()),
                "m0_predictions_by_seed": ";".join(
                    f"{int(row.seed)}:{int(row.m0_pair_prediction)}"
                    for row in group.itertuples()
                ),
            }
        )
    result = pd.DataFrame(rows).set_index("image_path")
    return result.reindex(first_paths)
