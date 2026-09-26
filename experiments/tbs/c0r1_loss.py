"""Losses for the C0-R1 bounded residual control."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.tbs.c0_loss import localizer_geometry_loss


def c0r1_loss(
    output,
    diagnosis_labels,
    lambda_geometry=0.02,
    lambda_residual=0.01,
):
    if diagnosis_labels.ndim != 1:
        raise ValueError("diagnosis_labels must have shape [batch]")
    if output["diagnosis_log_probs"].shape[0] != diagnosis_labels.shape[0]:
        raise ValueError("diagnosis labels and output must have equal batch size")
    geometry_weight = float(lambda_geometry)
    residual_weight = float(lambda_residual)
    if not math.isfinite(geometry_weight) or geometry_weight < 0.0:
        raise ValueError("lambda_geometry must be finite and nonnegative")
    if not math.isfinite(residual_weight) or residual_weight < 0.0:
        raise ValueError("lambda_residual must be finite and nonnegative")
    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    geometry_loss = localizer_geometry_loss(output["theta"])
    residual_loss = output["residual_logits"].square().mean()
    return {
        "loss": diagnosis_loss
        + geometry_weight * geometry_loss
        + residual_weight * residual_loss,
        "diagnosis_loss": diagnosis_loss,
        "geometry_loss": geometry_loss,
        "residual_loss": residual_loss,
    }


__all__ = ("c0r1_loss",)
