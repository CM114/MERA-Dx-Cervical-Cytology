"""Additional audit metrics for full-view TBS factorized candidates."""

from __future__ import annotations

import numpy as np

from experiments.xudata_gain_common import compute_locked_candidate_metrics


def _factor_targets(labels):
    labels = np.asarray(labels, dtype=np.int64)
    mask = labels > 0
    morph = np.where((labels == 3) | (labels == 4), 1, 0)
    evidence = np.where((labels == 2) | (labels == 4), 1, 0)
    return morph, evidence, mask


def compute_tbs_branch_metrics(y_true, y_prob, output_audits):
    """Combine locked candidate metrics with factor/prototype diagnostics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    metrics = compute_locked_candidate_metrics(y_true, y_prob)
    morph_probs = np.asarray(output_audits["morph_probs"], dtype=float)
    evidence_probs = np.asarray(output_audits["evidence_probs"], dtype=float)
    morph_targets, evidence_targets, semantic_mask = _factor_targets(y_true)
    if semantic_mask.any():
        metrics["morph_accuracy"] = float(
            (morph_probs[semantic_mask].argmax(axis=1) == morph_targets[semantic_mask]).mean()
        )
        metrics["evidence_accuracy"] = float(
            (evidence_probs[semantic_mask].argmax(axis=1) == evidence_targets[semantic_mask]).mean()
        )
    else:
        metrics["morph_accuracy"] = 0.0
        metrics["evidence_accuracy"] = 0.0
    residual_logits = np.asarray(output_audits["residual_logits"], dtype=float)
    if not np.isfinite(residual_logits).all():
        raise ValueError("residual logits contain nonfinite values")
    metrics["residual_logit_abs_mean"] = float(np.abs(residual_logits).mean())
    metrics["residual_logit_max_abs"] = float(np.abs(residual_logits).max())
    if "prototype_margin" in output_audits:
        margins = np.asarray(output_audits["prototype_margin"], dtype=float)
        if not np.isfinite(margins).all():
            raise ValueError("prototype margins contain nonfinite values")
        metrics["prototype_margin_mean"] = float(margins.mean())
    return {key: float(value) for key, value in metrics.items()}


__all__ = ("compute_tbs_branch_metrics",)
