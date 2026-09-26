import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.tbs.labels import DIAGNOSIS_NAMES


def normalize_probability_rows(y_prob, max_sum_deviation=0.01):
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinite values")
    if probabilities.size == 0:
        raise ValueError("probabilities are empty")
    if probabilities.min() < -1e-8:
        raise ValueError("probabilities contain negative values")

    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have a positive sum")

    maximum_deviation = float(np.abs(row_sums[:, 0] - 1.0).max())
    if maximum_deviation > float(max_sum_deviation):
        raise ValueError(
            "probability row sums deviate too far from 1.0: "
            f"maximum deviation={maximum_deviation:.8f}, "
            f"allowed={float(max_sum_deviation):.8f}"
        )
    return probabilities / row_sums


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _macro_specificity(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    values = []
    for index in range(5):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        values.append(_safe_divide(true_negative, true_negative + false_positive))
    return float(np.mean(values))


def _expected_calibration_error(y_true, y_prob, bins=15):
    confidence = y_prob.max(axis=1)
    prediction = y_prob.argmax(axis=1)
    correctness = (prediction == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(error)


def _binary_screen_metrics(y_true, screen_prob, prefix):
    true_screen = (y_true > 0).astype(np.int64)
    pred_screen = (screen_prob >= 0.5).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true_screen, pred_screen, labels=[0, 1]
    ).ravel()
    metrics = {
        f"{prefix}_sensitivity": _safe_divide(
            true_positive, true_positive + false_negative
        ),
        f"{prefix}_specificity": _safe_divide(
            true_negative, true_negative + false_positive
        ),
        f"{prefix}_fnr": _safe_divide(false_negative, true_positive + false_negative),
        f"{prefix}_ppv": _safe_divide(true_positive, true_positive + false_positive),
        f"{prefix}_npv": _safe_divide(true_negative, true_negative + false_negative),
    }
    try:
        metrics[f"{prefix}_auroc"] = float(roc_auc_score(true_screen, screen_prob))
        metrics[f"{prefix}_auprc"] = float(
            average_precision_score(true_screen, screen_prob)
        )
    except ValueError:
        metrics[f"{prefix}_auroc"] = float("nan")
        metrics[f"{prefix}_auprc"] = float("nan")
    return metrics


def _binary_semantic_metrics(y_true, y_prob, prefix):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError(f"{prefix} probabilities must be finite and within [0, 1]")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(f"{prefix} labels must be 0 or 1")

    y_pred = (y_prob >= 0.5).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_sensitivity": _safe_divide(
            true_positive, true_positive + false_negative
        ),
        f"{prefix}_specificity": _safe_divide(
            true_negative, true_negative + false_positive
        ),
        f"{prefix}_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics[f"{prefix}_auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics[f"{prefix}_auprc"] = float(
            average_precision_score(y_true, y_prob)
        )
    except ValueError:
        metrics[f"{prefix}_auroc"] = float("nan")
        metrics[f"{prefix}_auprc"] = float("nan")
    return metrics


def compute_semantic_metrics(
    morph_true,
    morph_high_prob,
    evidence_true,
    evidence_definitive_prob,
    semantic_mask,
    diagnosis_prob=None,
):
    morph_true = np.asarray(morph_true, dtype=np.int64)
    morph_high_prob = np.asarray(morph_high_prob, dtype=np.float64)
    evidence_true = np.asarray(evidence_true, dtype=np.int64)
    evidence_definitive_prob = np.asarray(
        evidence_definitive_prob, dtype=np.float64
    )
    semantic_mask = np.asarray(semantic_mask, dtype=bool)
    arrays = (
        morph_true,
        morph_high_prob,
        evidence_true,
        evidence_definitive_prob,
        semantic_mask,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("semantic labels, probabilities, and mask must be one-dimensional")
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("semantic arrays must contain the same number of samples")
    if not semantic_mask.any():
        raise ValueError("semantic metrics require at least one abnormal sample")

    masked_morph_true = morph_true[semantic_mask]
    masked_morph_prob = morph_high_prob[semantic_mask]
    masked_evidence_true = evidence_true[semantic_mask]
    masked_evidence_prob = evidence_definitive_prob[semantic_mask]
    metrics = _binary_semantic_metrics(
        masked_morph_true, masked_morph_prob, "morph"
    )
    metrics.update(
        _binary_semantic_metrics(
            masked_evidence_true,
            masked_evidence_prob,
            "evidence",
        )
    )

    if diagnosis_prob is not None:
        diagnosis_prob = normalize_probability_rows(diagnosis_prob)
        if diagnosis_prob.shape[0] != semantic_mask.shape[0]:
            raise ValueError("diagnosis and semantic arrays must have equal length")
        abnormal_mass = diagnosis_prob[:, 1:].sum(axis=1)
        masked_abnormal_mass = np.clip(abnormal_mass[semantic_mask], 1e-12, None)
        diagnosis_morph_high = (
            diagnosis_prob[semantic_mask, 3] + diagnosis_prob[semantic_mask, 4]
        ) / masked_abnormal_mass
        diagnosis_evidence_definitive = (
            diagnosis_prob[semantic_mask, 2] + diagnosis_prob[semantic_mask, 4]
        ) / masked_abnormal_mass
        metrics["morph_consistency_mae"] = float(
            np.abs(masked_morph_prob - diagnosis_morph_high).mean()
        )
        metrics["evidence_consistency_mae"] = float(
            np.abs(
                masked_evidence_prob - diagnosis_evidence_definitive
            ).mean()
        )
        metrics["morph_consistency_disagreement"] = float(
            np.mean(
                (masked_morph_prob >= 0.5)
                != (diagnosis_morph_high >= 0.5)
            )
        )
        metrics["evidence_consistency_disagreement"] = float(
            np.mean(
                (masked_evidence_prob >= 0.5)
                != (diagnosis_evidence_definitive >= 0.5)
            )
        )
    return metrics


def compute_stage1_metrics(y_true, y_prob, auxiliary_screen_prob=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != 5:
        raise ValueError("y_prob must have shape [samples, 5]")
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and y_prob must contain the same number of samples")
    y_prob = normalize_probability_rows(y_prob)

    y_pred = y_prob.argmax(axis=1)
    clipped = np.clip(y_prob, 1e-12, 1.0)
    one_hot = np.eye(5, dtype=np.float64)[y_true]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_specificity": _macro_specificity(y_true, y_pred),
        "nll": float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean()),
        "brier": float(np.square(y_prob - one_hot).sum(axis=1).mean()),
        "ece": _expected_calibration_error(y_true, y_prob),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(5)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["macro_auc"] = float("nan")

    for index, class_name in enumerate(DIAGNOSIS_NAMES):
        class_key = class_name.lower().replace("-", "_")
        binary_true = (y_true == index).astype(np.int64)
        binary_pred = (y_pred == index).astype(np.int64)
        metrics[f"{class_key}_sensitivity"] = float(
            recall_score(binary_true, binary_pred, zero_division=0)
        )
        metrics[f"{class_key}_precision"] = float(
            precision_score(binary_true, binary_pred, zero_division=0)
        )
        metrics[f"{class_key}_f1"] = float(
            f1_score(binary_true, binary_pred, zero_division=0)
        )

    metrics.update(_binary_screen_metrics(y_true, y_prob[:, 1:].sum(axis=1), "screen"))
    if auxiliary_screen_prob is not None:
        auxiliary_screen_prob = np.asarray(auxiliary_screen_prob, dtype=np.float64)
        metrics.update(
            _binary_screen_metrics(y_true, auxiliary_screen_prob, "aux_screen")
        )
        diagnosis_screen_prob = y_prob[:, 1:].sum(axis=1)
        metrics["screen_consistency_mae"] = float(
            np.abs(auxiliary_screen_prob - diagnosis_screen_prob).mean()
        )
        metrics["screen_decision_disagreement"] = float(
            np.mean(
                (auxiliary_screen_prob >= 0.5)
                != (diagnosis_screen_prob >= 0.5)
            )
        )

    pair_definitions = (
        (1, 2, "asc_us_to_lsil_rate"),
        (2, 1, "lsil_to_asc_us_rate"),
        (3, 4, "asc_h_to_hsil_rate"),
        (4, 3, "hsil_to_asc_h_rate"),
    )
    for true_label, predicted_label, name in pair_definitions:
        mask = y_true == true_label
        metrics[name] = _safe_divide(
            np.sum(mask & (y_pred == predicted_label)), np.sum(mask)
        )

    abnormal_mask = y_true > 0
    metrics["abnormal_to_normal_rate"] = _safe_divide(
        np.sum(abnormal_mask & (y_pred == 0)), np.sum(abnormal_mask)
    )
    asc_h_hsil_mask = np.isin(y_true, [3, 4])
    metrics["asc_h_hsil_to_normal_rate"] = _safe_divide(
        np.sum(asc_h_hsil_mask & (y_pred == 0)), np.sum(asc_h_hsil_mask)
    )
    metrics["asc_h_hsil_to_normal_lowgrade_rate"] = _safe_divide(
        np.sum(asc_h_hsil_mask & np.isin(y_pred, [0, 1, 2])),
        np.sum(asc_h_hsil_mask),
    )
    return metrics


def save_evaluation_artifacts(
    output_dir,
    split_name,
    y_true,
    y_prob,
    image_paths,
    maturity_labels,
    maturity_names,
    auxiliary_screen_prob=None,
    morph_labels=None,
    morph_high_prob=None,
    evidence_labels=None,
    evidence_definitive_prob=None,
    semantic_mask=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = normalize_probability_rows(y_prob)
    y_pred = y_prob.argmax(axis=1)

    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true,
            "true_name": [DIAGNOSIS_NAMES[index] for index in y_true],
            "pred_label": y_pred,
            "pred_name": [DIAGNOSIS_NAMES[index] for index in y_pred],
            "maturity_label": maturity_labels,
            "maturity_name": maturity_names,
            "screen_prob_from_diagnosis": y_prob[:, 1:].sum(axis=1),
        }
    )
    for index, class_name in enumerate(DIAGNOSIS_NAMES):
        predictions[f"prob_{class_name}"] = y_prob[:, index]
    if auxiliary_screen_prob is not None:
        predictions["aux_screen_prob"] = auxiliary_screen_prob

    semantic_values = (
        morph_labels,
        morph_high_prob,
        evidence_labels,
        evidence_definitive_prob,
        semantic_mask,
    )
    if any(value is not None for value in semantic_values):
        if not all(value is not None for value in semantic_values):
            raise ValueError("all semantic artifact arrays must be provided together")
        morph_labels = np.asarray(morph_labels, dtype=np.int64)
        morph_high_prob = np.asarray(morph_high_prob, dtype=np.float64)
        evidence_labels = np.asarray(evidence_labels, dtype=np.int64)
        evidence_definitive_prob = np.asarray(
            evidence_definitive_prob, dtype=np.float64
        )
        semantic_mask = np.asarray(semantic_mask, dtype=bool)
        morph_pred = np.where(semantic_mask, morph_high_prob >= 0.5, -1).astype(int)
        evidence_pred = np.where(
            semantic_mask, evidence_definitive_prob >= 0.5, -1
        ).astype(int)
        predictions["semantic_mask"] = semantic_mask.astype(int)
        predictions["morph_true"] = morph_labels
        predictions["morph_pred"] = morph_pred
        predictions["morph_prob_high"] = morph_high_prob
        predictions["evidence_true"] = evidence_labels
        predictions["evidence_pred"] = evidence_pred
        predictions["evidence_prob_definitive"] = evidence_definitive_prob
    predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    pd.DataFrame(matrix, index=DIAGNOSIS_NAMES, columns=DIAGNOSIS_NAMES).to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(5)),
        target_names=DIAGNOSIS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{split_name}_classification_report.csv"
    )

    metrics = compute_stage1_metrics(y_true, y_prob, auxiliary_screen_prob)
    if all(value is not None for value in semantic_values):
        metrics.update(
            compute_semantic_metrics(
                morph_labels,
                morph_high_prob,
                evidence_labels,
                evidence_definitive_prob,
                semantic_mask,
                y_prob,
            )
        )
    with (output_dir / f"{split_name}_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metrics
