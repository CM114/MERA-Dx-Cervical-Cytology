import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PAIR_BOUNDARY_LABELS = {
    "low": (1, 2),
    "high": (3, 4),
}


@dataclass(frozen=True)
class PairBoundarySupConResult:
    loss: torch.Tensor
    low_loss: torch.Tensor
    high_loss: torch.Tensor
    low_candidates: int
    low_valid: int
    low_coverage: float
    high_candidates: int
    high_valid: int
    high_coverage: float


def _pair_supcon_term(features, labels, pair_labels, temperature):
    mask = (labels == pair_labels[0]) | (labels == pair_labels[1])
    pair_features = features[mask]
    pair_targets = labels[mask]
    candidate_count = int(pair_targets.numel())
    zero = features.sum() * 0.0
    if candidate_count == 0:
        return zero, 0, 0

    similarity = pair_features @ pair_features.transpose(0, 1)
    similarity = similarity / temperature
    identity = torch.eye(candidate_count, dtype=torch.bool, device=features.device)
    positive = pair_targets[:, None].eq(pair_targets[None, :]) & ~identity
    negative = pair_targets[:, None].ne(pair_targets[None, :])
    valid = positive.any(dim=1) & negative.any(dim=1)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return zero, candidate_count, 0

    denominator_logits = similarity.masked_fill(identity, float("-inf"))
    log_prob = similarity - torch.logsumexp(
        denominator_logits, dim=1, keepdim=True
    )
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count
    )
    return per_anchor[valid].mean(), candidate_count, valid_count


def pair_boundary_supcon_loss(features, diagnosis_labels, temperature=0.1):
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dimensions]")
    if diagnosis_labels.ndim != 1:
        raise ValueError("diagnosis_labels must have shape [batch]")
    if features.shape[0] != diagnosis_labels.shape[0]:
        raise ValueError("features and labels must use the same batch size")
    if diagnosis_labels.device != features.device:
        raise ValueError("diagnosis_labels and features must use the same device")
    if (
        torch.is_floating_point(diagnosis_labels)
        or torch.is_complex(diagnosis_labels)
        or diagnosis_labels.dtype == torch.bool
    ):
        raise ValueError("diagnosis_labels must use an integer dtype")
    if not torch.all((diagnosis_labels >= 0) & (diagnosis_labels < 5)):
        raise ValueError("diagnosis labels must be in [0, 4]")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    float_features = features.float()
    if not torch.isfinite(float_features).all():
        raise FloatingPointError("features must be finite")
    normalized = F.normalize(float_features, p=2, dim=1)
    low_loss, low_candidates, low_valid = _pair_supcon_term(
        normalized, diagnosis_labels, PAIR_BOUNDARY_LABELS["low"], temperature
    )
    high_loss, high_candidates, high_valid = _pair_supcon_term(
        normalized, diagnosis_labels, PAIR_BOUNDARY_LABELS["high"], temperature
    )
    low_coverage = low_valid / low_candidates if low_candidates else 0.0
    high_coverage = high_valid / high_candidates if high_candidates else 0.0
    total = (low_loss + high_loss) / 2.0
    if not torch.isfinite(total):
        raise FloatingPointError("pair-boundary loss is nonfinite")
    return PairBoundarySupConResult(
        loss=total,
        low_loss=low_loss,
        high_loss=high_loss,
        low_candidates=low_candidates,
        low_valid=low_valid,
        low_coverage=low_coverage,
        high_candidates=high_candidates,
        high_valid=high_valid,
        high_coverage=high_coverage,
    )


def validate_label_smoothing(value):
    label_smoothing = float(value)
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            "label_smoothing must be finite and satisfy 0 <= value < 1"
        )
    return label_smoothing


def _smoothed_nll_loss(log_probs, labels, label_smoothing):
    if label_smoothing == 0.0:
        return F.nll_loss(log_probs, labels)

    per_sample_nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    per_sample_uniform_loss = -log_probs.mean(dim=1)
    return (
        (1.0 - label_smoothing) * per_sample_nll
        + label_smoothing * per_sample_uniform_loss
    ).mean()


def compute_stage1_loss(
    output,
    diagnosis_labels,
    screen_labels,
    variant,
    lambda_screen=0.3,
    label_smoothing=0.0,
):
    label_smoothing = validate_label_smoothing(label_smoothing)

    if variant == "m0":
        diagnosis_loss = F.cross_entropy(
            output["diagnosis_logits"],
            diagnosis_labels,
            label_smoothing=label_smoothing,
        )
        return {
            "loss": diagnosis_loss,
            "diagnosis_loss": diagnosis_loss,
            "screen_loss": diagnosis_loss.new_zeros(()),
        }

    if variant == "m1":
        diagnosis_loss = F.cross_entropy(
            output["diagnosis_logits"],
            diagnosis_labels,
            label_smoothing=label_smoothing,
        )
        screen_loss = F.binary_cross_entropy_with_logits(
            output["screen_logits"], screen_labels.float()
        )
        return {
            "loss": diagnosis_loss + float(lambda_screen) * screen_loss,
            "diagnosis_loss": diagnosis_loss,
            "screen_loss": screen_loss,
        }

    if variant == "m2":
        lambda_screen = float(lambda_screen)
        if not math.isfinite(lambda_screen) or lambda_screen != 0.0:
            raise ValueError("M2 lambda_screen must be 0.0")
        if diagnosis_labels.ndim != 1 or screen_labels.ndim != 1:
            raise ValueError(
                "diagnosis_labels and screen_labels must have shape [batch]"
            )
        if diagnosis_labels.shape != screen_labels.shape:
            raise ValueError(
                "diagnosis_labels and screen_labels must use the same batch size"
            )
        if not torch.all((diagnosis_labels >= 0) & (diagnosis_labels < 5)):
            raise ValueError("diagnosis_labels must be in [0, 4]")
        expected_screen_labels = (diagnosis_labels > 0).to(screen_labels.dtype)
        if not torch.equal(screen_labels, expected_screen_labels):
            raise ValueError("screen_labels must equal (diagnosis_labels > 0)")

        joint_nll = _smoothed_nll_loss(
            output["diagnosis_log_probs"],
            diagnosis_labels,
            label_smoothing,
        )
        screen_bce = F.binary_cross_entropy_with_logits(
            output["screen_logits"], screen_labels.float()
        )
        abnormal_mask = diagnosis_labels > 0
        if abnormal_mask.any():
            abnormal_cond_ce = F.cross_entropy(
                output["abnormal_logits"][abnormal_mask],
                diagnosis_labels[abnormal_mask] - 1,
            )
        else:
            abnormal_cond_ce = output["abnormal_logits"].sum() * 0.0
        return {
            "loss": joint_nll,
            "diagnosis_loss": joint_nll,
            "screen_loss": screen_bce,
            "joint_nll": joint_nll,
            "screen_bce": screen_bce,
            "abnormal_cond_ce": abnormal_cond_ce,
        }

    raise ValueError(f"Unknown Stage-1 variant: {variant}")


def _validate_nonnegative_weight(name, value):
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return weight


def cross_covariance_loss(
    morph_features,
    evidence_features,
    semantic_mask,
    eps=1e-6,
):
    semantic_mask = semantic_mask.bool()
    if semantic_mask.ndim != 1:
        raise ValueError("semantic_mask must have shape [batch]")
    if morph_features.ndim != 2 or evidence_features.ndim != 2:
        raise ValueError("semantic features must have shape [batch, dimensions]")
    if morph_features.shape[0] != evidence_features.shape[0]:
        raise ValueError("semantic features must use the same batch size")
    if morph_features.shape[0] != semantic_mask.shape[0]:
        raise ValueError("semantic_mask and features must use the same batch size")

    morph_abnormal = morph_features[semantic_mask].float()
    evidence_abnormal = evidence_features[semantic_mask].float()
    if morph_abnormal.shape[0] < 2:
        return (morph_features.sum() + evidence_features.sum()) * 0.0

    morph_centered = morph_abnormal - morph_abnormal.mean(dim=0, keepdim=True)
    evidence_centered = evidence_abnormal - evidence_abnormal.mean(
        dim=0, keepdim=True
    )
    morph_scale = morph_centered.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    evidence_scale = (
        evidence_centered.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    )
    morph_standardized = morph_centered / morph_scale
    evidence_standardized = evidence_centered / evidence_scale
    cross_covariance = (
        morph_standardized.transpose(0, 1) @ evidence_standardized
    ) / float(morph_abnormal.shape[0])
    return cross_covariance.square().mean()


def compute_m3_loss(
    output,
    diagnosis_labels,
    screen_labels,
    morph_labels,
    evidence_labels,
    semantic_mask,
    lambda_morph=0.3,
    lambda_evidence=0.3,
    lambda_decorr=0.01,
):
    lambda_morph = _validate_nonnegative_weight("lambda_morph", lambda_morph)
    lambda_evidence = _validate_nonnegative_weight(
        "lambda_evidence", lambda_evidence
    )
    lambda_decorr = _validate_nonnegative_weight("lambda_decorr", lambda_decorr)
    semantic_mask = semantic_mask.bool()

    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], diagnosis_labels)
    screen_loss = F.binary_cross_entropy_with_logits(
        output["screen_logits"], screen_labels.float()
    )

    if semantic_mask.any():
        masked_morph_labels = morph_labels[semantic_mask]
        masked_evidence_labels = evidence_labels[semantic_mask]
        if not ((masked_morph_labels >= 0) & (masked_morph_labels <= 1)).all():
            raise ValueError("masked morph labels must be 0 or 1")
        if not (
            (masked_evidence_labels >= 0) & (masked_evidence_labels <= 1)
        ).all():
            raise ValueError("masked evidence labels must be 0 or 1")
        morph_loss = F.cross_entropy(
            output["morph_logits"][semantic_mask],
            masked_morph_labels,
        )
        evidence_loss = F.cross_entropy(
            output["evidence_logits"][semantic_mask],
            masked_evidence_labels,
        )
    else:
        morph_loss = output["morph_logits"].sum() * 0.0
        evidence_loss = output["evidence_logits"].sum() * 0.0

    decorr_loss = cross_covariance_loss(
        output["morph_features"],
        output["evidence_features"],
        semantic_mask,
    )
    total_loss = (
        diagnosis_loss
        + lambda_morph * morph_loss
        + lambda_evidence * evidence_loss
        + lambda_decorr * decorr_loss
    )
    return {
        "loss": total_loss,
        "diagnosis_loss": diagnosis_loss,
        "screen_loss": screen_loss,
        "morph_loss": morph_loss,
        "evidence_loss": evidence_loss,
        "decorr_loss": decorr_loss,
    }
