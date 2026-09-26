"""Single-view TBS model with a safe factorized diagnosis residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tbs.singleview_model import compose_tbs_probabilities


class TBSResidualSingleViewModel(nn.Module):
    """Direct diagnosis classifier augmented by a zero-initialized TBS residual."""

    def __init__(self, backbone, feature_dim, semantic_dim=128):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.diagnosis_head = nn.Linear(self.feature_dim, 5)
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
        self.residual_adapter = nn.Linear(5, 5, bias=False)
        nn.init.zeros_(self.residual_adapter.weight)

    def _encode(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("backbone output does not match feature_dim")
        return features

    def forward(self, images):
        features = self._encode(images)
        direct_logits = self.diagnosis_head(features)
        screen_logits = self.screen_head(features).squeeze(1)
        morph_features = F.normalize(self.morph_projector(features), dim=1)
        evidence_features = F.normalize(self.evidence_projector(features), dim=1)
        morph_logits = self.morph_head(morph_features)
        evidence_logits = self.evidence_head(evidence_features)
        morph_probs = F.softmax(morph_logits.float(), dim=1)
        evidence_probs = F.softmax(evidence_logits.float(), dim=1)
        screen_probs = torch.sigmoid(screen_logits.float())
        factorized_probs = compose_tbs_probabilities(
            screen_probs, morph_probs, evidence_probs
        )
        factorized_log_probs = factorized_probs.clamp_min(1e-12).log()
        centered_factorized_log_probs = factorized_log_probs - factorized_log_probs.mean(
            dim=1, keepdim=True
        )
        residual_logits = self.residual_adapter(centered_factorized_log_probs)
        diagnosis_logits = direct_logits.float() + residual_logits.float()
        diagnosis_log_probs = F.log_softmax(diagnosis_logits, dim=1)
        diagnosis_probs = diagnosis_log_probs.exp()
        return {
            "features": features,
            "direct_diagnosis_logits": direct_logits,
            "diagnosis_logits": diagnosis_logits,
            "diagnosis_log_probs": diagnosis_log_probs,
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
            "factorized_probs": factorized_probs,
            "factorized_residual_logits": residual_logits,
            "residual_adapter_weight_norm": torch.linalg.vector_norm(
                self.residual_adapter.weight.float()
            ),
            "residual_logit_abs_mean": residual_logits.float().abs().mean(),
        }


def build_tbs_s1r2_model(
    model_name="caformer_s18", pretrained=True, semantic_dim=128
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the S1-R2 model") from exc
    backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSResidualSingleViewModel(
        backbone, feature_dim, semantic_dim=semantic_dim
    )
