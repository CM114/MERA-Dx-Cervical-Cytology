"""Full-view TBS factorization with an M0-primary bounded residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compose_tbs_factor_probabilities(screen_probs, morph_probs, evidence_probs):
    """Compose Normal plus the abnormal 2x2 TBS semantic grid."""
    if screen_probs.ndim != 1:
        raise ValueError("screen_probs must have shape [batch]")
    batch = int(screen_probs.shape[0])
    if morph_probs.shape != (batch, 2):
        raise ValueError("morph_probs must have shape [batch, 2]")
    if evidence_probs.shape != (batch, 2):
        raise ValueError("evidence_probs must have shape [batch, 2]")
    if not torch.isfinite(screen_probs).all() or not torch.isfinite(morph_probs).all():
        raise ValueError("factor probabilities must be finite")
    if not torch.isfinite(evidence_probs).all():
        raise ValueError("factor probabilities must be finite")
    abnormal = screen_probs.float().clamp(0.0, 1.0)
    morph = morph_probs.float().clamp_min(0.0)
    evidence = evidence_probs.float().clamp_min(0.0)
    probabilities = torch.stack(
        (
            1.0 - abnormal,
            abnormal * morph[:, 0] * evidence[:, 0],
            abnormal * morph[:, 0] * evidence[:, 1],
            abnormal * morph[:, 1] * evidence[:, 0],
            abnormal * morph[:, 1] * evidence[:, 1],
        ),
        dim=1,
    )
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


class TBSFVFactorizedResidualModel(nn.Module):
    """M0 full-view classifier with explicit TBS factors and bounded residual."""

    def __init__(
        self,
        backbone,
        feature_dim,
        semantic_dim=128,
        residual_logit_bound=0.10,
        activation_checkpointing=True,
    ):
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        bound = float(residual_logit_bound)
        if not 0.0 < bound <= 0.10:
            raise ValueError("residual_logit_bound must be within (0, 0.10]")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.residual_logit_bound = bound
        self.activation_checkpointing = bool(activation_checkpointing)

        self.base_head = nn.Linear(self.feature_dim, 5)
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
        residual_input_dim = self.feature_dim + (2 * self.semantic_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, self.semantic_dim),
            nn.GELU(),
            nn.Linear(self.semantic_dim, 5),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _encode(self, images):
        if (
            self.training
            and self.activation_checkpointing
            and torch.is_grad_enabled()
        ):
            from torch.utils.checkpoint import checkpoint

            features = checkpoint(self.backbone, images, use_reentrant=False)
        else:
            features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("backbone output does not match feature_dim")
        return features.float()

    @staticmethod
    def _log_probs(logits):
        return F.log_softmax(logits.float(), dim=1)

    def _factor_branch(self, full_features):
        morph_features = F.normalize(self.morph_projector(full_features), dim=1)
        evidence_features = F.normalize(self.evidence_projector(full_features), dim=1)
        screen_logits = self.screen_head(full_features).squeeze(1)
        morph_logits = self.morph_head(morph_features)
        evidence_logits = self.evidence_head(evidence_features)
        screen_probs = torch.sigmoid(screen_logits.float())
        morph_probs = F.softmax(morph_logits.float(), dim=1)
        evidence_probs = F.softmax(evidence_logits.float(), dim=1)
        factorized_probs = compose_tbs_factor_probabilities(
            screen_probs, morph_probs, evidence_probs
        )
        return {
            "screen_logits": screen_logits,
            "screen_probs": screen_probs,
            "morph_features": morph_features,
            "evidence_features": evidence_features,
            "morph_logits": morph_logits,
            "evidence_logits": evidence_logits,
            "morph_probs": morph_probs,
            "evidence_probs": evidence_probs,
            "factorized_probs": factorized_probs,
        }

    def _residual_context(self, branch):
        """Optional bounded context exposed to a specialized residual path."""
        return None

    def _residual_input(self, full_features, branch):
        return torch.cat(
            (full_features, branch["morph_features"], branch["evidence_features"]), dim=1
        )

    def forward(self, images):
        full_features = self._encode(images)
        base_logits = self.base_head(full_features)
        branch = self._factor_branch(full_features)

        residual_context = self._residual_context(branch)
        residual_input = self._residual_input(full_features, branch)
        raw_residual = self.residual_head(residual_input)
        residual_logits = self.residual_logit_bound * torch.tanh(raw_residual)
        diagnosis_logits = base_logits + residual_logits
        base_log_probs = self._log_probs(base_logits)
        diagnosis_log_probs = self._log_probs(diagnosis_logits)
        diagnosis_probs = diagnosis_log_probs.exp()
        result = {
            "full_features": full_features,
            "base_diagnosis_logits": base_logits,
            "base_diagnosis_log_probs": base_log_probs,
            "base_diagnosis_probs": base_log_probs.exp(),
            "diagnosis_logits": diagnosis_logits,
            "diagnosis_log_probs": diagnosis_log_probs,
            "diagnosis_probs": diagnosis_probs,
            "residual_logits": residual_logits,
            "residual_logit_abs_mean": residual_logits.abs().mean(),
            "residual_logit_max_abs": residual_logits.abs().max(),
            **branch,
        }
        if residual_context is not None:
            result["prototype_context"] = residual_context
            result["prototype_context_abs_mean"] = residual_context.abs().mean()
        return result


class TBSFVDualPrototypeResidualModel(TBSFVFactorizedResidualModel):
    """Independent dual-space prototype extension of the full-view C1 branch.

    Prototypes live in the morph/evidence semantic spaces and only contribute a
    bounded residual to the factor logits.  The five-class M0 head and its
    bounded diagnosis residual remain the primary prediction path.
    """

    def __init__(
        self,
        backbone,
        feature_dim,
        semantic_dim=128,
        residual_logit_bound=0.10,
        prototype_temperature=10.0,
        prototype_residual_bound=0.10,
        activation_checkpointing=True,
    ):
        super().__init__(
            backbone,
            feature_dim,
            semantic_dim=semantic_dim,
            residual_logit_bound=residual_logit_bound,
            activation_checkpointing=activation_checkpointing,
        )
        temperature = float(prototype_temperature)
        if temperature <= 0.0:
            raise ValueError("prototype_temperature must be positive")
        bound = float(prototype_residual_bound)
        if not 0.0 < bound <= 0.10:
            raise ValueError("prototype_residual_bound must be within (0, 0.10]")
        self.prototype_temperature = temperature
        self.prototype_residual_bound = bound
        self.morph_prototypes = nn.Parameter(torch.randn(2, self.semantic_dim))
        self.evidence_prototypes = nn.Parameter(torch.randn(2, self.semantic_dim))
        self.prototype_log_temperature = nn.Parameter(
            torch.log(torch.full((1,), temperature))
        )

    def _prototype_logits(self, features, prototypes):
        normalized_prototypes = F.normalize(prototypes, dim=1)
        temperature = self.prototype_log_temperature.exp().clamp(1.0, 100.0)
        return temperature * features.float().matmul(normalized_prototypes.t())

    def _factor_branch(self, full_features):
        branch = super()._factor_branch(full_features)
        morph_prototype_logits = self._prototype_logits(
            branch["morph_features"], self.morph_prototypes
        )
        evidence_prototype_logits = self._prototype_logits(
            branch["evidence_features"], self.evidence_prototypes
        )
        morph_logits = branch["morph_logits"] + self.prototype_residual_bound * torch.tanh(
            morph_prototype_logits
        )
        evidence_logits = branch["evidence_logits"] + self.prototype_residual_bound * torch.tanh(
            evidence_prototype_logits
        )
        branch.update(
            {
                "morph_logits": morph_logits,
                "evidence_logits": evidence_logits,
                "morph_probs": F.softmax(morph_logits.float(), dim=1),
                "evidence_probs": F.softmax(evidence_logits.float(), dim=1),
                "morph_prototype_logits": morph_prototype_logits,
                "evidence_prototype_logits": evidence_prototype_logits,
            }
        )
        branch["factorized_probs"] = compose_tbs_factor_probabilities(
            branch["screen_probs"], branch["morph_probs"], branch["evidence_probs"]
        )
        return branch

    @staticmethod
    def _validate_prototype_inputs(features, labels, name):
        if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError(f"{name} features/labels have incompatible shapes")
        if features.shape[1] <= 0 or not torch.isfinite(features).all():
            raise ValueError(f"{name} features must be finite and non-empty")
        labels = labels.long()
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(f"{name} labels must be binary")
        return labels

    @torch.no_grad()
    def initialize_prototypes_from_train_features(
        self, morph_features, morph_labels, evidence_features, evidence_labels
    ):
        morph_labels = self._validate_prototype_inputs(
            morph_features, morph_labels, "morph"
        )
        evidence_labels = self._validate_prototype_inputs(
            evidence_features, evidence_labels, "evidence"
        )
        if morph_features.shape[1] != self.semantic_dim or evidence_features.shape[1] != self.semantic_dim:
            raise ValueError("prototype feature dimension does not match semantic_dim")
        for name, features, labels, parameter in (
            ("morph", morph_features, morph_labels, self.morph_prototypes),
            ("evidence", evidence_features, evidence_labels, self.evidence_prototypes),
        ):
            means = []
            for group in (0, 1):
                selected = features[labels == group]
                if selected.shape[0] == 0:
                    raise ValueError(f"{name} factor group {group} is empty")
                means.append(F.normalize(selected.float().mean(dim=0, keepdim=True), dim=1).squeeze(0))
            parameter.copy_(torch.stack(means, dim=0).to(parameter))


def build_tbs_dualprototype_model(
    model_name="caformer_s18",
    pretrained=True,
    semantic_dim=128,
    residual_logit_bound=0.10,
    prototype_temperature=10.0,
    prototype_residual_bound=0.10,
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the full-view dual-prototype model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFVDualPrototypeResidualModel(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
        residual_logit_bound=residual_logit_bound,
        prototype_temperature=prototype_temperature,
        prototype_residual_bound=prototype_residual_bound,
    )


def build_tbs_fv_factorized_model(
    model_name="caformer_s18",
    pretrained=True,
    semantic_dim=128,
    residual_logit_bound=0.10,
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the full-view factorized model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFVFactorizedResidualModel(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
        residual_logit_bound=residual_logit_bound,
    )


__all__ = (
    "TBSFVFactorizedResidualModel",
    "TBSFVDualPrototypeResidualModel",
    "build_tbs_fv_factorized_model",
    "build_tbs_dualprototype_model",
    "compose_tbs_factor_probabilities",
)
