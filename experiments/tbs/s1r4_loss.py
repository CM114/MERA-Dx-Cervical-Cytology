"""Locked ASC-H/HSIL conditional boundary loss for S1-R4."""

from __future__ import annotations

from experiments.tbs.s1r4_protocol import LOCKED_S1R4_CONFIG


def _validate_inputs(diagnosis_log_probs, diagnosis_labels):
    if diagnosis_log_probs.ndim != 2 or diagnosis_log_probs.shape[1] != 5:
        raise ValueError("diagnosis_log_probs must have shape [batch, 5]")
    if diagnosis_labels.ndim != 1 or len(diagnosis_labels) != len(
        diagnosis_log_probs
    ):
        raise ValueError("diagnosis_labels must have shape [batch]")
    if diagnosis_labels.dtype.is_floating_point or diagnosis_labels.dtype.is_complex:
        raise ValueError("diagnosis_labels must use an integer dtype")
    if ((diagnosis_labels < 0) | (diagnosis_labels > 4)).any():
        raise ValueError("diagnosis_labels must be in [0, 4]")


def high_grade_pair_boundary_loss(diagnosis_log_probs, diagnosis_labels):
    """Separate ASC-H and HSIL conditional probabilities on true high-grade samples."""
    _validate_inputs(diagnosis_log_probs, diagnosis_labels)
    high_grade_mask = (diagnosis_labels == 3) | (diagnosis_labels == 4)
    if not high_grade_mask.any():
        return diagnosis_log_probs.sum() * 0.0
    pair_log_mass = diagnosis_log_probs[high_grade_mask, 3:5].logsumexp(dim=1)
    true_log_prob = diagnosis_log_probs[high_grade_mask].gather(
        1, diagnosis_labels[high_grade_mask].unsqueeze(1)
    ).squeeze(1)
    return -(true_log_prob - pair_log_mass).mean()


def singleview_s1r4_loss(output, targets):
    """Return S1-R3 losses plus the fixed conditional high-grade boundary term."""
    from experiments.tbs.s1r3_loss import singleview_s1r3_loss

    losses = singleview_s1r3_loss(output, targets)
    boundary = high_grade_pair_boundary_loss(
        output["diagnosis_log_probs"], targets["diagnosis_labels"]
    )
    result = dict(losses)
    result["high_grade_pair_boundary_loss"] = boundary
    result["loss"] = (
        losses["loss"]
        + LOCKED_S1R4_CONFIG["lambda_high_grade_pair_boundary"] * boundary
    )
    return result


__all__ = ("high_grade_pair_boundary_loss", "singleview_s1r4_loss")
