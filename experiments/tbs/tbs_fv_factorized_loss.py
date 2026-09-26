"""Losses for the full-view TBS factorized residual candidates."""

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


def _structural_labels(diagnosis_labels):
    if diagnosis_labels.ndim != 1:
        raise ValueError("diagnosis_labels must have shape [batch]")
    if diagnosis_labels.numel() and not torch.all((diagnosis_labels >= 0) & (diagnosis_labels < 5)):
        raise ValueError("diagnosis_labels must be in [0, 4]")
    semantic_mask = diagnosis_labels > 0
    morph_labels = torch.zeros_like(diagnosis_labels)
    evidence_labels = torch.zeros_like(diagnosis_labels)
    morph_labels[(diagnosis_labels == 3) | (diagnosis_labels == 4)] = 1
    evidence_labels[(diagnosis_labels == 2) | (diagnosis_labels == 4)] = 1
    screen_labels = semantic_mask.to(dtype=torch.float32)
    return screen_labels, morph_labels, evidence_labels, semantic_mask


def _masked_cross_entropy(logits, labels, mask):
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


def prototype_boundary_loss(output, diagnosis_labels, lambda_margin=0.05):
    """Apply stronger core and weaker boundary margins in each factor space."""
    margin = _weight("lambda_margin", lambda_margin)
    if "morph_prototype_logits" not in output or "evidence_prototype_logits" not in output:
        return output["diagnosis_log_probs"].sum() * 0.0
    _, morph_labels, evidence_labels, semantic_mask = _structural_labels(diagnosis_labels)
    if not bool(semantic_mask.any()):
        return output["morph_prototype_logits"].sum() * 0.0
    morph_logits = output["morph_prototype_logits"][semantic_mask]
    evidence_logits = output["evidence_prototype_logits"][semantic_mask]
    morph_targets = morph_labels[semantic_mask]
    evidence_targets = evidence_labels[semantic_mask]
    morph_true = morph_logits.gather(1, morph_targets[:, None]).squeeze(1)
    evidence_true = evidence_logits.gather(1, evidence_targets[:, None]).squeeze(1)
    morph_other = morph_logits.gather(1, (1 - morph_targets)[:, None]).squeeze(1)
    evidence_other = evidence_logits.gather(1, (1 - evidence_targets)[:, None]).squeeze(1)
    # LSIL/HSIL are the core structural examples; ASC-US/ASC-H are treated as
    # boundary examples and receive a deliberately weaker margin.
    core = ((diagnosis_labels[semantic_mask] == 2) | (diagnosis_labels[semantic_mask] == 4))
    boundary_margin = margin * 0.4
    morph_margins = torch.where(core, torch.as_tensor(margin, device=morph_logits.device), torch.as_tensor(boundary_margin, device=morph_logits.device))
    evidence_margins = morph_margins
    morph_margin_loss = F.relu(morph_margins - morph_true + morph_other).mean()
    evidence_margin_loss = F.relu(evidence_margins - evidence_true + evidence_other).mean()
    return (morph_margin_loss + evidence_margin_loss) / 2.0


def low_grade_pair_protection_loss(final_log_probs, base_log_probs, diagnosis_labels):
    """Penalize final conditional ASC-US/LSIL quality falling below base.

    The base probabilities are treated as a detached reference.  This is a
    protection term, not a second classifier objective: it is zero when the
    final branch matches or improves the base branch for the true low-grade
    class, and only activates on a relative regression.
    """
    if final_log_probs.shape != base_log_probs.shape:
        raise ValueError("final_log_probs and base_log_probs must have the same shape")
    if final_log_probs.ndim != 2 or final_log_probs.shape[1] != 5:
        raise ValueError("log probabilities must have shape [batch, 5]")
    if diagnosis_labels.ndim != 1 or diagnosis_labels.shape[0] != final_log_probs.shape[0]:
        raise ValueError("diagnosis_labels must have shape [batch]")
    mask = (diagnosis_labels == 1) | (diagnosis_labels == 2)
    if not bool(mask.any()):
        return final_log_probs.sum() * 0.0
    final_pair = final_log_probs[mask][:, 1:3]
    base_pair = base_log_probs[mask][:, 1:3].detach()
    final_pair_log_mass = torch.logsumexp(final_pair, dim=1)
    base_pair_log_mass = torch.logsumexp(base_pair, dim=1)
    true_labels = diagnosis_labels[mask]
    final_true = final_log_probs[mask].gather(1, true_labels[:, None]).squeeze(1)
    base_true = base_log_probs[mask].detach().gather(1, true_labels[:, None]).squeeze(1)
    final_conditional = final_true - final_pair_log_mass
    base_conditional = base_true - base_pair_log_mass
    return F.relu(base_conditional - final_conditional).mean()


def high_grade_mass_protection_loss(final_log_probs, base_log_probs, diagnosis_labels):
    """Penalize final ASC-H/HSIL total mass falling below the base branch."""
    if final_log_probs.shape != base_log_probs.shape:
        raise ValueError("final_log_probs and base_log_probs must have the same shape")
    if final_log_probs.ndim != 2 or final_log_probs.shape[1] != 5:
        raise ValueError("log probabilities must have shape [batch, 5]")
    if diagnosis_labels.ndim != 1 or diagnosis_labels.shape[0] != final_log_probs.shape[0]:
        raise ValueError("diagnosis_labels must have shape [batch]")
    mask = (diagnosis_labels == 3) | (diagnosis_labels == 4)
    if not bool(mask.any()):
        return final_log_probs.sum() * 0.0
    final_mass = torch.logsumexp(final_log_probs[mask][:, 3:5], dim=1)
    base_mass = torch.logsumexp(base_log_probs[mask][:, 3:5].detach(), dim=1)
    return F.relu(base_mass - final_mass).mean()


def _validate_log_probs_and_labels(log_probs, diagnosis_labels):
    if log_probs.ndim != 2 or log_probs.shape[1] != 5:
        raise ValueError("log probabilities must have shape [batch, 5]")
    if diagnosis_labels.ndim != 1 or diagnosis_labels.shape[0] != log_probs.shape[0]:
        raise ValueError("diagnosis_labels must have shape [batch]")


def _hard_fraction(name, value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and within (0, 1]")
    return value


def _hard_mean(values, fraction):
    fraction = _hard_fraction("hard_fraction", fraction)
    count = max(1, int(math.ceil(values.numel() * fraction)))
    return values.topk(count, largest=True).values.mean()


def conditional_pair_cross_entropy(
    log_probs, diagnosis_labels, pair_start, pair_end, hard_fraction=1.0
):
    """Compute CE only within one diagnosis pair, with pair renormalization."""
    _validate_log_probs_and_labels(log_probs, diagnosis_labels)
    pair_start = int(pair_start)
    pair_end = int(pair_end)
    if not 0 <= pair_start < pair_end <= 5 or pair_end - pair_start < 2:
        raise ValueError("pair must be a valid slice of at least two diagnosis classes")
    mask = (diagnosis_labels >= pair_start) & (diagnosis_labels < pair_end)
    if not bool(mask.any()):
        return log_probs.sum() * 0.0
    pair_log_probs = log_probs[mask][:, pair_start:pair_end]
    pair_log_probs = pair_log_probs - torch.logsumexp(
        pair_log_probs, dim=1, keepdim=True
    )
    targets = diagnosis_labels[mask] - pair_start
    per_sample = F.nll_loss(pair_log_probs, targets, reduction="none")
    return _hard_mean(per_sample, hard_fraction)


def _high_grade_boundary_components(
    log_probs, diagnosis_labels, margin, hard_fraction=1.0
):
    _validate_log_probs_and_labels(log_probs, diagnosis_labels)
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and nonnegative")
    mask = (diagnosis_labels == 3) | (diagnosis_labels == 4)
    if not bool(mask.any()):
        zero = log_probs.sum() * 0.0
        return zero, zero
    high_mass = torch.logsumexp(log_probs[mask][:, 3:5], dim=1)
    non_high_mass = torch.logsumexp(log_probs[mask][:, :3], dim=1)
    violations = F.relu(
        torch.as_tensor(margin, device=log_probs.device, dtype=log_probs.dtype)
        - (high_mass - non_high_mass)
    )
    return _hard_mean(violations, hard_fraction), (violations > 0).to(dtype=log_probs.dtype).mean()


def high_grade_boundary_margin_loss(
    log_probs, diagnosis_labels, margin=0.10, hard_fraction=1.0
):
    """Penalize true high-grade samples whose high-vs-non-high gap is too small."""
    return _high_grade_boundary_components(
        log_probs, diagnosis_labels, margin, hard_fraction
    )[0]


def factorized_tbs_loss(
    output,
    diagnosis_labels,
    lambda_screen=0.20,
    lambda_morph=0.30,
    lambda_evidence=0.30,
    lambda_decorr=0.01,
    lambda_prototype=0.0,
    lambda_base_anchor=0.25,
    lambda_residual=0.01,
    lambda_low_grade_protection=0.0,
    lambda_high_grade_mass_protection=0.0,
    lambda_low_grade_pair_ce=0.0,
    lambda_high_grade_pair_ce=0.0,
    lambda_high_grade_boundary=0.0,
    high_grade_boundary_margin=0.10,
    hard_example_fraction=1.0,
    boundary_hard_example_fraction=None,
):
    """Train final five-class logits while protecting explicit TBS factors."""
    screen_weight = _weight("lambda_screen", lambda_screen)
    morph_weight = _weight("lambda_morph", lambda_morph)
    evidence_weight = _weight("lambda_evidence", lambda_evidence)
    decorr_weight = _weight("lambda_decorr", lambda_decorr)
    prototype_weight = _weight("lambda_prototype", lambda_prototype)
    anchor_weight = _weight("lambda_base_anchor", lambda_base_anchor)
    residual_weight = _weight("lambda_residual", lambda_residual)
    low_grade_protection_weight = _weight(
        "lambda_low_grade_protection", lambda_low_grade_protection
    )
    high_grade_mass_protection_weight = _weight(
        "lambda_high_grade_mass_protection", lambda_high_grade_mass_protection
    )
    low_grade_pair_ce_weight = _weight(
        "lambda_low_grade_pair_ce", lambda_low_grade_pair_ce
    )
    high_grade_pair_ce_weight = _weight(
        "lambda_high_grade_pair_ce", lambda_high_grade_pair_ce
    )
    high_grade_boundary_weight = _weight(
        "lambda_high_grade_boundary", lambda_high_grade_boundary
    )
    hard_example_fraction = _hard_fraction(
        "hard_example_fraction", hard_example_fraction
    )
    if boundary_hard_example_fraction is None:
        boundary_hard_example_fraction = hard_example_fraction
    boundary_hard_example_fraction = _hard_fraction(
        "boundary_hard_example_fraction", boundary_hard_example_fraction
    )
    screen_labels, morph_labels, evidence_labels, semantic_mask = _structural_labels(
        diagnosis_labels
    )
    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    base_anchor_loss = F.nll_loss(
        output["base_diagnosis_log_probs"], diagnosis_labels
    )
    screen_loss = F.binary_cross_entropy_with_logits(
        output["screen_logits"], screen_labels
    )
    morph_loss = _masked_cross_entropy(output["morph_logits"], morph_labels, semantic_mask)
    evidence_loss = _masked_cross_entropy(
        output["evidence_logits"], evidence_labels, semantic_mask
    )
    decorr_loss = cross_covariance_loss(
        output["morph_features"], output["evidence_features"], semantic_mask
    )
    prototype_loss = prototype_boundary_loss(output, diagnosis_labels)
    residual_loss = output["residual_logits"].square().mean()
    low_grade_protection = low_grade_pair_protection_loss(
        output["diagnosis_log_probs"],
        output["base_diagnosis_log_probs"],
        diagnosis_labels,
    )
    high_grade_mass_protection = high_grade_mass_protection_loss(
        output["diagnosis_log_probs"],
        output["base_diagnosis_log_probs"],
        diagnosis_labels,
    )
    low_grade_pair_ce = conditional_pair_cross_entropy(
        output["diagnosis_log_probs"], diagnosis_labels, 1, 3,
        hard_fraction=hard_example_fraction,
    )
    high_grade_pair_ce = conditional_pair_cross_entropy(
        output["diagnosis_log_probs"], diagnosis_labels, 3, 5,
        hard_fraction=hard_example_fraction,
    )
    high_grade_boundary, high_grade_boundary_active_fraction = (
        _high_grade_boundary_components(
            output["diagnosis_log_probs"],
            diagnosis_labels,
            high_grade_boundary_margin,
            hard_fraction=boundary_hard_example_fraction,
        )
    )
    low_grade_pair_ce_weighted = low_grade_pair_ce_weight * low_grade_pair_ce
    high_grade_pair_ce_weighted = high_grade_pair_ce_weight * high_grade_pair_ce
    high_grade_boundary_weighted = high_grade_boundary_weight * high_grade_boundary
    total = (
        diagnosis_loss
        + anchor_weight * base_anchor_loss
        + screen_weight * screen_loss
        + morph_weight * morph_loss
        + evidence_weight * evidence_loss
        + decorr_weight * decorr_loss
        + prototype_weight * prototype_loss
        + residual_weight * residual_loss
        + low_grade_protection_weight * low_grade_protection
        + high_grade_mass_protection_weight * high_grade_mass_protection
        + low_grade_pair_ce_weighted
        + high_grade_pair_ce_weighted
        + high_grade_boundary_weighted
    )
    return {
        "loss": total,
        "diagnosis_loss": diagnosis_loss,
        "base_anchor_loss": base_anchor_loss,
        "screen_loss": screen_loss,
        "morph_loss": morph_loss,
        "evidence_loss": evidence_loss,
        "decorr_loss": decorr_loss,
        "prototype_loss": prototype_loss,
        "residual_loss": residual_loss,
        "low_grade_protection_loss": low_grade_protection,
        "high_grade_mass_protection_loss": high_grade_mass_protection,
        "low_grade_pair_ce_loss": low_grade_pair_ce,
        "high_grade_pair_ce_loss": high_grade_pair_ce,
        "high_grade_boundary_loss": high_grade_boundary,
        "low_grade_pair_ce_weighted": low_grade_pair_ce_weighted,
        "high_grade_pair_ce_weighted": high_grade_pair_ce_weighted,
        "high_grade_boundary_weighted": high_grade_boundary_weighted,
        "high_grade_boundary_active_fraction": high_grade_boundary_active_fraction,
    }


__all__ = (
    "factorized_tbs_loss",
    "conditional_pair_cross_entropy",
    "high_grade_boundary_margin_loss",
    "high_grade_mass_protection_loss",
    "low_grade_pair_protection_loss",
    "prototype_boundary_loss",
)
