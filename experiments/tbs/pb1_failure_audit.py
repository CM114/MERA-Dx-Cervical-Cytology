"""Pure calculations for the frozen M0-versus-PB1 failure audit."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from experiments.tbs.boundary_audit import (
    PAIR_SPECS,
    binary_metrics,
    geometry_metrics,
    select_pair,
)


METRIC_KEYS = (
    "pair_macro_f1",
    "pair_balanced_accuracy",
    "pair_roc_auc",
    "centroid_distance",
    "within_class_scatter_0",
    "within_class_scatter_1",
    "fisher_ratio",
    "silhouette_cosine",
    "mean_local_purity",
)


def _feature_matrix(features, expected_rows=None):
    array = np.asarray(features)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("features must be a nonempty two-dimensional array")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError("features and labels must contain the same samples")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("features must be real numeric values")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def _labels(labels, expected_rows=None):
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError("labels must match the sample count")
    if not np.isin(array, (0, 1, 2, 3, 4)).all():
        raise ValueError("labels must use diagnosis values 0 through 4")
    return array.astype(np.int64, copy=False)


def _probabilities(probabilities, expected_rows):
    raw = np.asarray(probabilities)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("probabilities must be real numeric values")
    array = raw.astype(np.float64, copy=False)
    if array.shape != (expected_rows, 5):
        raise ValueError("probabilities must have shape [samples, 5]")
    if not np.isfinite(array).all() or np.any(array < -1e-8):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    return array


def _pair_scores(probabilities, pair_labels):
    pair = probabilities[:, list(pair_labels)]
    mass = pair.sum(axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("pair probability mass must be positive")
    return pair[:, 1] / mass


def _empty_geometry():
    return {
        "centroid_distance": np.nan,
        "within_class_scatter_0": np.nan,
        "within_class_scatter_1": np.nan,
        "fisher_ratio": np.nan,
        "silhouette_cosine": np.nan,
        "mean_local_purity": np.nan,
        "n_neighbors": 0,
    }


def audit_model_split(
    model_name,
    split_name,
    features,
    labels,
    probabilities,
    n_neighbors=5,
):
    """Return pair-level metrics and row-aligned local-purity observations."""

    labels = _labels(labels)
    features = _feature_matrix(features, len(labels))
    probabilities = _probabilities(probabilities, len(labels))
    metric_rows = []
    sample_rows = []
    for pair_name, spec in PAIR_SPECS.items():
        pair_features, binary_labels, indices = select_pair(
            features, labels, spec["labels"]
        )
        scores = _pair_scores(probabilities[indices], spec["labels"])
        predictions = (scores >= 0.5).astype(np.int64)
        classification = binary_metrics(binary_labels, scores, predictions)
        geometry = geometry_metrics(
            pair_features,
            binary_labels,
            n_neighbors=n_neighbors,
        )
        metric_rows.append(
            {
                "model_name": str(model_name),
                "split": str(split_name),
                "pair_name": pair_name,
                "class_0": spec["names"][0],
                "class_1": spec["names"][1],
                "n_samples": int(len(indices)),
                "n_class_0": int((binary_labels == 0).sum()),
                "n_class_1": int((binary_labels == 1).sum()),
                "pair_macro_f1": classification["macro_f1"],
                "pair_balanced_accuracy": classification["balanced_accuracy"],
                "pair_roc_auc": classification["roc_auc"],
                **geometry["metrics"],
            }
        )
        for offset, row_index in enumerate(indices):
            sample_rows.append(
                {
                    "model_name": str(model_name),
                    "split": str(split_name),
                    "row_index": int(row_index),
                    "pair_name": pair_name,
                    "local_purity": float(geometry["local_purity"][offset]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(sample_rows)


def compare_metric_tables(m0_metrics, pb1_metrics):
    """Create one PB1-minus-M0 row for every split and boundary."""

    keys = ["split", "pair_name", "class_0", "class_1"]
    required = set(keys) | set(METRIC_KEYS) | {"model_name", "n_samples"}
    for name, frame in (("m0", m0_metrics), ("pb1", pb1_metrics)):
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
            raise ValueError(f"{name} metric table is missing required columns")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} metric table has duplicate split/pair rows")
    merged = m0_metrics.merge(
        pb1_metrics,
        on=keys,
        how="outer",
        suffixes=("_m0", "_pb1"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("M0 and PB1 metric identities do not match")
    result = merged[keys].copy()
    result["n_samples"] = merged["n_samples_m0"].astype(int)
    if not np.array_equal(merged["n_samples_m0"], merged["n_samples_pb1"]):
        raise ValueError("M0 and PB1 metric sample counts differ")
    for metric in METRIC_KEYS:
        result[f"m0_{metric}"] = merged[f"{metric}_m0"]
        result[f"pb1_{metric}"] = merged[f"{metric}_pb1"]
        result[f"delta_{metric}"] = (
            merged[f"{metric}_pb1"] - merged[f"{metric}_m0"]
        )
    return result


def _entropy(probabilities):
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _pair_sample_fields(labels, probabilities):
    count = len(labels)
    pair_name = np.full(count, "not_applicable", dtype=object)
    confidence = np.full(count, np.nan, dtype=np.float64)
    true_margin = np.full(count, np.nan, dtype=np.float64)
    entropy = np.full(count, np.nan, dtype=np.float64)
    for current_name, spec in PAIR_SPECS.items():
        indices = np.flatnonzero(np.isin(labels, spec["labels"]))
        conditional = probabilities[indices][:, list(spec["labels"])]
        conditional /= conditional.sum(axis=1, keepdims=True)
        binary_true = (labels[indices] == spec["labels"][1]).astype(np.int64)
        true_probability = conditional[np.arange(len(indices)), binary_true]
        other_probability = conditional[np.arange(len(indices)), 1 - binary_true]
        pair_name[indices] = current_name
        confidence[indices] = conditional.max(axis=1)
        true_margin[indices] = true_probability - other_probability
        entropy[indices] = _entropy(conditional)
    return pair_name, confidence, true_margin, entropy


def _local_purity_vector(local_table, count, model_name):
    result = np.full(count, np.nan, dtype=np.float64)
    if local_table is None:
        return result
    required = {"row_index", "local_purity"}
    if not isinstance(local_table, pd.DataFrame) or not required.issubset(
        local_table.columns
    ):
        raise ValueError(f"{model_name} local-purity table is invalid")
    if local_table["row_index"].duplicated().any():
        raise ValueError(f"{model_name} local-purity row indices must be unique")
    indices = local_table["row_index"].to_numpy(dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError(f"{model_name} local-purity row index is out of range")
    values = local_table["local_purity"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{model_name} local purity must be finite in [0, 1]")
    result[indices] = values
    return result


def build_transition_audit(
    index_frame,
    m0_probabilities,
    pb1_probabilities,
    m0_local_purity=None,
    pb1_local_purity=None,
):
    """Build sample-level prediction transitions and boundary evidence fields."""

    required = {"row_index", "image_path", "true_label"}
    if not isinstance(index_frame, pd.DataFrame) or not required.issubset(
        index_frame.columns
    ):
        raise ValueError("index frame is missing required columns")
    if not np.array_equal(
        index_frame["row_index"].to_numpy(), np.arange(len(index_frame))
    ):
        raise ValueError("index frame row_index must be contiguous and ordered")
    if index_frame["image_path"].astype(str).duplicated().any():
        raise ValueError("index frame image paths must be unique")
    labels = _labels(index_frame["true_label"].to_numpy(), len(index_frame))
    m0 = _probabilities(m0_probabilities, len(labels))
    pb1 = _probabilities(pb1_probabilities, len(labels))
    m0_pred = m0.argmax(axis=1)
    pb1_pred = pb1.argmax(axis=1)
    m0_correct = m0_pred == labels
    pb1_correct = pb1_pred == labels
    transitions = np.select(
        [m0_correct & pb1_correct, m0_correct & ~pb1_correct, ~m0_correct & pb1_correct],
        ["both_correct", "harmed", "rescued"],
        default="persistent_error",
    )
    pair_names, m0_pair_conf, m0_margin, m0_pair_entropy = _pair_sample_fields(
        labels, m0
    )
    pb1_pair_names, pb1_pair_conf, pb1_margin, pb1_pair_entropy = (
        _pair_sample_fields(labels, pb1)
    )
    if not np.array_equal(pair_names, pb1_pair_names):
        raise AssertionError("pair assignment must depend only on true labels")
    result = index_frame.copy()
    result["m0_pred_label"] = m0_pred
    result["pb1_pred_label"] = pb1_pred
    result["m0_correct"] = m0_correct
    result["pb1_correct"] = pb1_correct
    result["transition"] = transitions
    result["normal_harmed"] = (labels == 0) & m0_correct & ~pb1_correct
    result["pair_name"] = pair_names
    result["m0_confidence"] = m0.max(axis=1)
    result["pb1_confidence"] = pb1.max(axis=1)
    result["confidence_delta"] = result["pb1_confidence"] - result["m0_confidence"]
    result["m0_entropy"] = _entropy(m0)
    result["pb1_entropy"] = _entropy(pb1)
    result["entropy_delta"] = result["pb1_entropy"] - result["m0_entropy"]
    result["m0_pair_confidence"] = m0_pair_conf
    result["pb1_pair_confidence"] = pb1_pair_conf
    result["m0_pair_entropy"] = m0_pair_entropy
    result["pb1_pair_entropy"] = pb1_pair_entropy
    result["m0_pair_true_margin"] = m0_margin
    result["pb1_pair_true_margin"] = pb1_margin
    result["pair_true_margin_delta"] = pb1_margin - m0_margin
    result["m0_local_purity"] = _local_purity_vector(
        m0_local_purity, len(labels), "m0"
    )
    result["pb1_local_purity"] = _local_purity_vector(
        pb1_local_purity, len(labels), "pb1"
    )
    result["local_purity_delta"] = (
        result["pb1_local_purity"] - result["m0_local_purity"]
    )
    return result


def build_maturity_audit(
    model_name,
    split_name,
    features,
    labels,
    probabilities,
    maturity_labels,
    maturity_names,
    n_neighbors=5,
):
    """Audit each observed maturity stratum without inventing absent classes."""

    labels = _labels(labels)
    features = _feature_matrix(features, len(labels))
    probabilities = _probabilities(probabilities, len(labels))
    maturity_labels = np.asarray(maturity_labels)
    maturity_names = np.asarray(maturity_names, dtype=object)
    if maturity_labels.ndim != 1 or len(maturity_labels) != len(labels):
        raise ValueError("maturity labels must match the samples")
    if maturity_names.ndim != 1 or len(maturity_names) != len(labels):
        raise ValueError("maturity names must match the samples")
    rows = []
    identities = sorted(
        {(int(label), str(name)) for label, name in zip(maturity_labels, maturity_names)}
    )
    for maturity_label, maturity_name in identities:
        maturity_mask = (maturity_labels == maturity_label) & (
            maturity_names.astype(str) == maturity_name
        )
        for pair_name, spec in PAIR_SPECS.items():
            indices = np.flatnonzero(maturity_mask & np.isin(labels, spec["labels"]))
            present = set(labels[indices].tolist())
            row = {
                "model_name": str(model_name),
                "split": str(split_name),
                "pair_name": pair_name,
                "maturity_label": maturity_label,
                "maturity_name": maturity_name,
                "n_samples": int(len(indices)),
                "n_class_0": int((labels[indices] == spec["labels"][0]).sum()),
                "n_class_1": int((labels[indices] == spec["labels"][1]).sum()),
            }
            if present != set(spec["labels"]):
                rows.append(
                    {
                        **row,
                        "status": "insufficient_two_classes",
                        "geometry_status": "not_applicable",
                        **{key: np.nan for key in METRIC_KEYS},
                    }
                )
                continue
            binary = (labels[indices] == spec["labels"][1]).astype(np.int64)
            scores = _pair_scores(probabilities[indices], spec["labels"])
            classification = binary_metrics(binary, scores, (scores >= 0.5).astype(int))
            geometry_values = _empty_geometry()
            geometry_status = "insufficient_samples"
            if len(indices) >= 3 and min(np.bincount(binary, minlength=2)) >= 1:
                geometry = geometry_metrics(
                    features[indices], binary, n_neighbors=n_neighbors
                )
                geometry_values = geometry["metrics"]
                geometry_status = "ok"
            rows.append(
                {
                    **row,
                    "status": "ok",
                    "geometry_status": geometry_status,
                    "pair_macro_f1": classification["macro_f1"],
                    "pair_balanced_accuracy": classification["balanced_accuracy"],
                    "pair_roc_auc": classification["roc_auc"],
                    **geometry_values,
                }
            )
    return pd.DataFrame(rows)


def build_review_list(transitions, per_group_limit=20):
    """Select deterministic pair-transition examples and retain all Normal harms."""

    if not isinstance(per_group_limit, int) or per_group_limit <= 0:
        raise ValueError("per_group_limit must be a positive integer")
    required = {
        "image_path",
        "pair_name",
        "transition",
        "normal_harmed",
        "pair_true_margin_delta",
        "pb1_pair_true_margin",
    }
    if not isinstance(transitions, pd.DataFrame) or not required.issubset(
        transitions.columns
    ):
        raise ValueError("transition table is missing review-list columns")
    selected = []
    for pair_name in PAIR_SPECS:
        for transition in ("rescued", "harmed", "persistent_error"):
            subset = transitions[
                (transitions["pair_name"] == pair_name)
                & (transitions["transition"] == transition)
            ].copy()
            if transition == "rescued":
                subset = subset.sort_values(
                    ["pair_true_margin_delta", "image_path"],
                    ascending=[False, True],
                )
            elif transition == "harmed":
                subset = subset.sort_values(
                    ["pair_true_margin_delta", "image_path"],
                    ascending=[True, True],
                )
            else:
                subset = subset.sort_values(
                    ["pb1_pair_true_margin", "image_path"],
                    ascending=[True, True],
                )
            subset = subset.head(per_group_limit)
            subset["review_reason"] = f"{pair_name}:{transition}"
            selected.append(subset)
    normal_harms = transitions[transitions["normal_harmed"].astype(bool)].copy()
    normal_harms = normal_harms.sort_values("image_path")
    normal_harms["review_reason"] = "normal_harmed"
    selected.append(normal_harms)
    result = pd.concat(selected, ignore_index=True)
    result = result.drop_duplicates("image_path", keep="first").reset_index(drop=True)
    result.insert(0, "review_rank", np.arange(1, len(result) + 1, dtype=np.int64))
    return result


def route_decision(
    train_mean_local_purity_delta,
    dev_mean_local_purity_delta,
    dev_boundary_composite_f1_delta,
    dev_pair_f1_deltas,
):
    """Apply the pre-registered PB1 failure-audit routing thresholds."""

    values = {
        "train_mean_local_purity_delta": train_mean_local_purity_delta,
        "dev_mean_local_purity_delta": dev_mean_local_purity_delta,
        "dev_boundary_composite_f1_delta": dev_boundary_composite_f1_delta,
    }
    if set(dev_pair_f1_deltas) != set(PAIR_SPECS):
        raise ValueError("dev pair deltas must contain low_grade and high_grade")
    values.update(
        {f"dev_{name}_f1_delta": value for name, value in dev_pair_f1_deltas.items()}
    )
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        values[name] = float(value)
    train_good = values["train_mean_local_purity_delta"] >= 0.010
    dev_good = values["dev_mean_local_purity_delta"] >= 0.010
    boundary_good = values["dev_boundary_composite_f1_delta"] >= 0.005
    pairs_safe = all(
        values[f"dev_{name}_f1_delta"] >= -0.003 for name in PAIR_SPECS
    )
    if not train_good and not dev_good:
        decision = "OBJECTIVE_MISMATCH_STOP_GLOBAL_PB"
    elif train_good and not dev_good:
        decision = "GENERALIZATION_FAILURE_STOP_AND_REVIEW"
    elif train_good and dev_good and not boundary_good:
        decision = "GEOMETRY_ONLY_RUN_FROZEN_HEAD_PROBE"
    elif train_good and dev_good and boundary_good and pairs_safe:
        decision = "PB2_DESIGN_ELIGIBLE"
    else:
        decision = "MIXED_GEOMETRY_STOP_AND_REVIEW"
    return {
        "decision": decision,
        "pb1_promoted": False,
        "thresholds": {
            "mean_local_purity_delta_min": 0.010,
            "boundary_composite_f1_delta_min": 0.005,
            "each_pair_f1_delta_min": -0.003,
        },
        "observed": values,
    }
