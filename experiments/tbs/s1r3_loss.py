"""Locked high-grade undercall penalty for the S1-R3 experiment."""

from __future__ import annotations

from experiments.tbs.s1r3_protocol import LOCKED_S1R3_CONFIG


def high_grade_risk_loss(diagnosis_log_probs, diagnosis_labels):
    """Penalize low total ASC-H/HSIL probability on true high-grade samples."""
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
    high_grade_mask = (diagnosis_labels == 3) | (diagnosis_labels == 4)
    if not high_grade_mask.any():
        return diagnosis_log_probs.sum() * 0.0
    high_grade_log_mass = diagnosis_log_probs[high_grade_mask, 3:5].logsumexp(
        dim=1
    )
    return -high_grade_log_mass.mean()


def singleview_s1r3_loss(output, targets):
    """Return the unchanged S1-R2 objective plus locked high-grade risk."""
    from experiments.tbs.singleview_model import singleview_tbs_loss

    losses = singleview_tbs_loss(
        output,
        targets,
        stage="s1",
        lambda_screen=LOCKED_S1R3_CONFIG["lambda_screen"],
        lambda_morph=LOCKED_S1R3_CONFIG["lambda_morph"],
        lambda_evidence=LOCKED_S1R3_CONFIG["lambda_evidence"],
        lambda_decorr=LOCKED_S1R3_CONFIG["lambda_decorr"],
    )
    risk = high_grade_risk_loss(
        output["diagnosis_log_probs"], targets["diagnosis_labels"]
    )
    result = dict(losses)
    result["high_grade_risk_loss"] = risk
    result["loss"] = (
        losses["loss"]
        + LOCKED_S1R3_CONFIG["lambda_high_grade_risk"] * risk
    )
    return result


__all__ = ("high_grade_risk_loss", "singleview_s1r3_loss")
