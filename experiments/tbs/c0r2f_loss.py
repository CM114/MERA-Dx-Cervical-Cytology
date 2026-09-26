"""C0-R2F loss: boundary auxiliaries supervise the full-view branch only."""

from __future__ import annotations

import torch.nn.functional as F

from experiments.tbs.c0_loss import localizer_geometry_loss
from experiments.tbs.c0r2_loss import (
    _validate,
    _weight,
    high_grade_mass_loss,
    low_grade_pair_loss,
)


def c0r2f_loss(
    output,
    diagnosis_labels,
    lambda_geometry=0.02,
    lambda_residual=0.01,
    lambda_full_anchor=0.25,
    lambda_low_grade_pair=0.05,
    lambda_high_grade_mass=0.05,
):
    """Use boundary losses on full logits to avoid direct localizer pressure."""

    _validate(output["diagnosis_log_probs"], diagnosis_labels)
    geometry_weight = _weight("lambda_geometry", lambda_geometry)
    residual_weight = _weight("lambda_residual", lambda_residual)
    full_weight = _weight("lambda_full_anchor", lambda_full_anchor)
    low_weight = _weight("lambda_low_grade_pair", lambda_low_grade_pair)
    high_weight = _weight("lambda_high_grade_mass", lambda_high_grade_mass)
    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    full_anchor_loss = F.nll_loss(output["full_diagnosis_log_probs"], diagnosis_labels)
    pair_loss = low_grade_pair_loss(output["full_diagnosis_log_probs"], diagnosis_labels)
    mass_loss = high_grade_mass_loss(output["full_diagnosis_log_probs"], diagnosis_labels)
    geometry_loss = localizer_geometry_loss(output["theta"])
    residual_loss = output["residual_logits"].square().mean()
    return {
        "loss": (
            diagnosis_loss
            + full_weight * full_anchor_loss
            + low_weight * pair_loss
            + high_weight * mass_loss
            + geometry_weight * geometry_loss
            + residual_weight * residual_loss
        ),
        "diagnosis_loss": diagnosis_loss,
        "full_anchor_loss": full_anchor_loss,
        "low_grade_pair_loss": pair_loss,
        "high_grade_mass_loss": mass_loss,
        "geometry_loss": geometry_loss,
        "residual_loss": residual_loss,
    }


__all__ = ("c0r2f_loss",)
