"""Single-view S0 baseline and S1 explicit TBS factorized classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGES = ("s0", "s1")


def compose_tbs_probabilities(screen_probs, morph_probs, evidence_probs):
    abnormal = screen_probs
    return torch.stack(
        (
            1.0 - abnormal,
            abnormal * morph_probs[:, 0] * evidence_probs[:, 0],
            abnormal * morph_probs[:, 0] * evidence_probs[:, 1],
            abnormal * morph_probs[:, 1] * evidence_probs[:, 0],
            abnormal * morph_probs[:, 1] * evidence_probs[:, 1],
        ),
        dim=1,
    )


class TBSFactorizedSingleViewModel(nn.Module):
    def __init__(self, backbone, feature_dim, semantic_dim=128, stage="s1"):
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.stage = stage
        if stage == "s0":
            self.diagnosis_head = nn.Linear(self.feature_dim, 5)
            return
        self.screen_head = nn.Linear(self.feature_dim, 1)
        self.morph_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.evidence_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.morph_head = nn.Linear(self.semantic_dim, 2)
        self.evidence_head = nn.Linear(self.semantic_dim, 2)

    def _encode(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("backbone output does not match feature_dim")
        return features

    def forward(self, images):
        features = self._encode(images)
        if self.stage == "s0":
            logits = self.diagnosis_head(features)
            probs = F.softmax(logits.float(), dim=1)
            return {
                "features": features,
                "diagnosis_logits": logits,
                "diagnosis_log_probs": probs.clamp_min(1e-12).log(),
                "diagnosis_probs": probs,
                "screen_probs": probs[:, 1:].sum(dim=1),
                "screen_logits": torch.logit(probs[:, 1:].sum(dim=1).clamp(1e-6, 1 - 1e-6)),
            }
        screen_logits = self.screen_head(features).squeeze(1)
        morph_features = F.normalize(self.morph_projector(features), dim=1)
        evidence_features = F.normalize(self.evidence_projector(features), dim=1)
        morph_logits = self.morph_head(morph_features)
        evidence_logits = self.evidence_head(evidence_features)
        morph_probs = F.softmax(morph_logits.float(), dim=1)
        evidence_probs = F.softmax(evidence_logits.float(), dim=1)
        screen_probs = torch.sigmoid(screen_logits.float())
        diagnosis_probs = compose_tbs_probabilities(screen_probs, morph_probs, evidence_probs)
        return {
            "features": features,
            "diagnosis_logits": None,
            "diagnosis_log_probs": diagnosis_probs.clamp_min(1e-12).log(),
            "diagnosis_probs": diagnosis_probs,
            "screen_logits": screen_logits,
            "screen_probs": screen_probs,
            "morph_features": morph_features,
            "evidence_features": evidence_features,
            "morph_semantic_logits": morph_logits,
            "evidence_semantic_logits": evidence_logits,
            "morph_logits": morph_logits,
            "evidence_logits": evidence_logits,
            "morph_probs": morph_probs,
            "evidence_probs": evidence_probs,
        }


def singleview_tbs_loss(output, targets, stage="s1", lambda_screen=0.2, lambda_morph=0.3, lambda_evidence=0.3, lambda_decorr=0.01):
    from experiments.tbs.losses import cross_covariance_loss

    diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], targets["diagnosis_labels"])
    zero = diagnosis_loss.new_zeros(())
    if stage == "s0":
        return {"loss": diagnosis_loss, "diagnosis_loss": diagnosis_loss, "screen_loss": zero, "morph_loss": zero, "evidence_loss": zero, "decorr_loss": zero}
    mask = targets["semantic_mask"].bool()
    screen_loss = F.binary_cross_entropy_with_logits(output["screen_logits"], targets["screen_labels"].float())
    if mask.any():
        morph_loss = F.cross_entropy(output["morph_semantic_logits"][mask], targets["morph_labels"][mask])
        evidence_loss = F.cross_entropy(output["evidence_semantic_logits"][mask], targets["evidence_labels"][mask])
    else:
        morph_loss = output["morph_features"].sum() * 0.0
        evidence_loss = output["evidence_features"].sum() * 0.0
    decorr_loss = cross_covariance_loss(output["morph_features"], output["evidence_features"], mask)
    total = diagnosis_loss + lambda_screen * screen_loss + lambda_morph * morph_loss + lambda_evidence * evidence_loss + lambda_decorr * decorr_loss
    return {"loss": total, "diagnosis_loss": diagnosis_loss, "screen_loss": screen_loss, "morph_loss": morph_loss, "evidence_loss": evidence_loss, "decorr_loss": decorr_loss}


def build_tbs_singleview_model(stage, model_name="caformer_s18", pretrained=True, semantic_dim=128):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the single-view model") from exc
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFactorizedSingleViewModel(backbone, feature_dim, semantic_dim=semantic_dim, stage=stage)
