"""Small, locked boundary-stability losses for C0-R2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.tbs.c0_loss import localizer_geometry_loss


def _validate(log_probs, labels):
    if labels.ndim != 1:
        raise ValueError("diagnosis_labels must have shape [batch]")
    if log_probs.ndim != 2 or log_probs.shape[1] != 5:
        raise ValueError("diagnosis_log_probs must have shape [batch, 5]")
    if log_probs.shape[0] != labels.shape[0]:
        raise ValueError("diagnosis labels and output must have equal batch size")
    if labels.numel() and (labels.min() < 0 or labels.max() > 4):
        raise ValueError("diagnosis_labels must be in [0, 4]")


def low_grade_pair_loss(log_probs, labels):
    """Conditional CE inside the LSIL/ASC-US pair (labels 1 and 2)."""

    _validate(log_probs, labels)
    mask = (labels == 1) | (labels == 2)
    if not bool(mask.any()):
        return log_probs.new_zeros(())
    selected = log_probs[mask]
    selected_labels = labels[mask]
    pair_log_mass = torch.logsumexp(selected[:, 1:3], dim=1)
    true_log_prob = selected.gather(1, selected_labels[:, None]).squeeze(1)
    return -(true_log_prob - pair_log_mass).mean()


def high_grade_mass_loss(log_probs, labels):
    """Protect total mass assigned to the ASC-H/HSIL pair (labels 3 and 4)."""

    _validate(log_probs, labels)
    mask = (labels == 3) | (labels == 4)
    if not bool(mask.any()):
        return log_probs.new_zeros(())
    return -torch.logsumexp(log_probs[mask][:, 3:5], dim=1).mean()


def _weight(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def c0r2_loss(
    output,
    diagnosis_labels,
    lambda_geometry=0.02,
    lambda_residual=0.01,
    lambda_full_anchor=0.25,
    lambda_low_grade_pair=0.05,
    lambda_high_grade_mass=0.05,
):
    """Full-view-primary C0-R1 model with bounded boundary stabilizers."""

    _validate(output["diagnosis_log_probs"], diagnosis_labels)
    geometry_weight = _weight("lambda_geometry", lambda_geometry)
    residual_weight = _weight("lambda_residual", lambda_residual)
    full_weight = _weight("lambda_full_anchor", lambda_full_anchor)
    low_weight = _weight("lambda_low_grade_pair", lambda_low_grade_pair)
    high_weight = _weight("lambda_high_grade_mass", lambda_high_grade_mass)
    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    full_anchor_loss = F.nll_loss(
        output["full_diagnosis_log_probs"], diagnosis_labels
    )
    pair_loss = low_grade_pair_loss(output["diagnosis_log_probs"], diagnosis_labels)
    mass_loss = high_grade_mass_loss(output["diagnosis_log_probs"], diagnosis_labels)
    geometry_loss = localizer_geometry_loss(output["theta"])
    residual_loss = output["residual_logits"].square().mean()
    total = (
        diagnosis_loss
        + full_weight * full_anchor_loss
        + low_weight * pair_loss
        + high_weight * mass_loss
        + geometry_weight * geometry_loss
        + residual_weight * residual_loss
    )
    return {
        "loss": total,
        "diagnosis_loss": diagnosis_loss,
        "full_anchor_loss": full_anchor_loss,
        "low_grade_pair_loss": pair_loss,
        "high_grade_mass_loss": mass_loss,
        "geometry_loss": geometry_loss,
        "residual_loss": residual_loss,
    }


__all__ = ("c0r2_loss", "high_grade_mass_loss", "low_grade_pair_loss")
