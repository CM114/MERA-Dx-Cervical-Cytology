"""Losses and preflight contracts for the C0-C2 TBS factorized pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.tbs.losses import cross_covariance_loss


def _weight(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def localizer_geometry_loss(
    theta,
    target_area=0.36,
    minimum_area=0.20,
    maximum_area=0.64,
    maximum_translation=0.35,
    shear_weight=0.25,
    reflection_weight=1.0,
):
    """Keep the local view local, valid, centered, and weakly axis aligned."""
    if theta.ndim != 3 or theta.shape[1:] != (2, 3):
        raise ValueError("theta must have shape [batch, 2, 3]")
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


def factorized_tbs_loss(
    output,
    targets,
    stage="c2",
    lambda_screen=0.2,
    lambda_morph=0.3,
    lambda_evidence=0.3,
    lambda_decorr=0.01,
    lambda_prototype=0.1,
    lambda_geometry=0.02,
):
    """Train diagnosis plus protected TBS factors and C2 prototype spaces."""
    weights = {
        name: _weight(name, value)
        for name, value in {
            "lambda_screen": lambda_screen,
            "lambda_morph": lambda_morph,
            "lambda_evidence": lambda_evidence,
            "lambda_decorr": lambda_decorr,
            "lambda_prototype": lambda_prototype,
            "lambda_geometry": lambda_geometry,
        }.items()
    }
    diagnosis_labels = targets["diagnosis_labels"]
    semantic_mask = targets["semantic_mask"].bool()
    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    geometry_loss = localizer_geometry_loss(output["theta"])

    if stage == "c0":
        zero = diagnosis_loss.new_zeros(())
        total = diagnosis_loss + weights["lambda_geometry"] * geometry_loss
        return {
            "loss": total,
            "diagnosis_loss": diagnosis_loss,
            "screen_loss": zero,
            "morph_loss": zero,
            "evidence_loss": zero,
            "decorr_loss": zero,
            "prototype_loss": zero,
            "geometry_loss": geometry_loss,
        }

    screen_loss = F.binary_cross_entropy_with_logits(
        output["screen_logits"], targets["screen_labels"].float()
    )
    if semantic_mask.any():
        morph_labels = targets["morph_labels"][semantic_mask]
        evidence_labels = targets["evidence_labels"][semantic_mask]
        morph_loss = F.cross_entropy(
            output["morph_semantic_logits"][semantic_mask], morph_labels
        )
        evidence_loss = F.cross_entropy(
            output["evidence_semantic_logits"][semantic_mask], evidence_labels
        )
        if stage == "c2":
            morph_prototype_loss = F.cross_entropy(
                output["morph_prototype_logits"][semantic_mask], morph_labels
            )
            evidence_prototype_loss = F.cross_entropy(
                output["evidence_prototype_logits"][semantic_mask], evidence_labels
            )
            prototype_loss = (morph_prototype_loss + evidence_prototype_loss) / 2.0
        else:
            prototype_loss = diagnosis_loss.new_zeros(())
    else:
        morph_loss = output["morph_semantic_logits"].sum() * 0.0
        evidence_loss = output["evidence_semantic_logits"].sum() * 0.0
        prototype_loss = (output["morph_features"].sum() + output["evidence_features"].sum()) * 0.0

    decorr_loss = cross_covariance_loss(
        output["morph_features"], output["evidence_features"], semantic_mask
    )
    total = (
        diagnosis_loss
        + weights["lambda_screen"] * screen_loss
        + weights["lambda_morph"] * morph_loss
        + weights["lambda_evidence"] * evidence_loss
        + weights["lambda_decorr"] * decorr_loss
        + weights["lambda_prototype"] * prototype_loss
        + weights["lambda_geometry"] * geometry_loss
    )
    return {
        "loss": total,
        "diagnosis_loss": diagnosis_loss,
        "screen_loss": screen_loss,
        "morph_loss": morph_loss,
        "evidence_loss": evidence_loss,
        "decorr_loss": decorr_loss,
        "prototype_loss": prototype_loss,
        "geometry_loss": geometry_loss,
    }
