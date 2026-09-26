"""Losses and geometry constraints for the locked C0 dual-view control."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def localizer_geometry_loss(
    theta,
    target_area=0.36,
    minimum_area=0.20,
    maximum_area=0.64,
    maximum_translation=0.35,
    shear_weight=0.25,
    reflection_weight=1.0,
):
    """Penalize non-local, reflected, sheared, or off-center affine views."""

    if theta.ndim != 3 or tuple(theta.shape[1:]) != (2, 3):
        raise ValueError("theta must have shape [batch, 2, 3]")
    values = {
        "target_area": target_area,
        "minimum_area": minimum_area,
        "maximum_area": maximum_area,
        "maximum_translation": maximum_translation,
        "shear_weight": shear_weight,
        "reflection_weight": reflection_weight,
    }
    for name, value in values.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(minimum_area) > float(maximum_area):
        raise ValueError("minimum_area must not exceed maximum_area")

    linear = theta[:, :, :2].float()
    translation = theta[:, :, 2].float()
    determinant = linear[:, 0, 0] * linear[:, 1, 1] - linear[:, 0, 1] * linear[:, 1, 0]
    area = determinant.abs()
    target = (area - float(target_area)).square()
    too_small = F.relu(float(minimum_area) - area).square()
    too_large = F.relu(area - float(maximum_area)).square()
    reflection = F.relu(-determinant).square()
    off_center = F.relu(translation.abs() - float(maximum_translation)).square().mean(dim=1)
    shear = linear[:, 0, 1].square() + linear[:, 1, 0].square()
    return (
        target
        + too_small
        + too_large
        + float(reflection_weight) * reflection
        + off_center
        + float(shear_weight) * shear
    ).mean()


def c0_loss(output, diagnosis_labels, lambda_geometry=0.02):
    """Return diagnosis CE plus the fixed localizer geometry penalty."""

    if diagnosis_labels.ndim != 1:
        raise ValueError("diagnosis_labels must have shape [batch]")
    if output["diagnosis_log_probs"].shape[0] != diagnosis_labels.shape[0]:
        raise ValueError("diagnosis_labels and diagnosis output must have equal batch size")
    weight = float(lambda_geometry)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("lambda_geometry must be finite and nonnegative")
    diagnosis_loss = F.nll_loss(
        output["diagnosis_log_probs"], diagnosis_labels
    )
    geometry_loss = localizer_geometry_loss(output["theta"])
    return {
        "loss": diagnosis_loss + weight * geometry_loss,
        "diagnosis_loss": diagnosis_loss,
        "geometry_loss": geometry_loss,
    }


__all__ = ("c0_loss", "localizer_geometry_loss")
