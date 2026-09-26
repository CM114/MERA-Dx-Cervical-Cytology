"""C0-C2 TBS factorized dual-view and dual-space prototype model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGES = ("c0", "c1", "c2")


def compose_tbs_probabilities(screen_probs, morph_probs, evidence_probs):
    """Compose Normal plus the abnormal 2x2 TBS semantic grid."""
    if screen_probs.ndim != 1:
        raise ValueError("screen_probs must have shape [batch]")
    if morph_probs.shape != (screen_probs.shape[0], 2):
        raise ValueError("morph_probs must have shape [batch, 2]")
    if evidence_probs.shape != (screen_probs.shape[0], 2):
        raise ValueError("evidence_probs must have shape [batch, 2]")

    abnormal = screen_probs
    probabilities = torch.stack(
        (
            1.0 - abnormal,
            abnormal * morph_probs[:, 0] * evidence_probs[:, 0],
            abnormal * morph_probs[:, 0] * evidence_probs[:, 1],
            abnormal * morph_probs[:, 1] * evidence_probs[:, 0],
            abnormal * morph_probs[:, 1] * evidence_probs[:, 1],
        ),
        dim=1,
    )
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


class LocalViewGenerator(nn.Module):
    """Learn a conservative local-cell affine crop from the full image."""

    def __init__(self, in_channels=3, initial_scale=0.60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.affine = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.affine[-1].weight)
        self.affine[-1].bias.data.copy_(
            torch.tensor([initial_scale, 0.0, 0.0, 0.0, initial_scale, 0.0])
        )

    def forward(self, images):
        theta = self.affine(self.features(images)).view(-1, 2, 3)
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        local_images = F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return local_images, theta


class TBSFactorizedDualSpaceModel(nn.Module):
    """C0 dual view, C1 factorized TBS semantics, and C2 dual-space prototypes."""

    def __init__(
        self,
        backbone,
        feature_dim,
        semantic_dim=128,
        stage="c2",
        prototype_residual=0.25,
    ):
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        if not 0.0 <= float(prototype_residual) <= 1.0:
            raise ValueError("prototype_residual must be within [0, 1]")

        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.stage = stage
        self.prototype_residual = float(prototype_residual)
        self.local_view = LocalViewGenerator()
        self.view_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )
        self.c0_head = nn.Linear(self.feature_dim * 2, 5)

        self.screen_head = nn.Linear(self.feature_dim * 2, 1)
        self.morph_projector = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.evidence_projector = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.morph_head = nn.Linear(self.semantic_dim, 2)
        self.evidence_head = nn.Linear(self.semantic_dim, 2)

        if stage == "c2":
            self.morph_prototypes = nn.Parameter(torch.randn(2, self.semantic_dim))
            self.evidence_prototypes = nn.Parameter(
                torch.randn(2, self.semantic_dim)
            )
            self.log_morph_temperature = nn.Parameter(torch.tensor(math.log(10.0)))
            self.log_evidence_temperature = nn.Parameter(
                torch.tensor(math.log(10.0))
            )
        else:
            self.register_parameter("morph_prototypes", None)
            self.register_parameter("evidence_prototypes", None)
            self.register_parameter("log_morph_temperature", None)
            self.register_parameter("log_evidence_temperature", None)

    def _encode(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("backbone output does not match feature_dim")
        return features

    @staticmethod
    def _prototype_logits(features, prototypes, log_temperature):
        normalized_features = F.normalize(features.float(), p=2, dim=1)
        normalized_prototypes = F.normalize(prototypes.float(), p=2, dim=1)
        temperature = F.softplus(log_temperature.float()) + 0.1
        return temperature * (normalized_features @ normalized_prototypes.transpose(0, 1))

    def initialize_factor_prototypes(
        self,
        morph_features,
        morph_labels,
        evidence_features,
        evidence_labels,
    ):
        """Initialize C2 prototypes from train-only factor centroids."""
        if self.stage != "c2":
            raise ValueError("factor prototypes are available only for stage c2")
        pairs = (
            (self.morph_prototypes, morph_features, morph_labels, "morph"),
            (
                self.evidence_prototypes,
                evidence_features,
                evidence_labels,
                "evidence",
            ),
        )
        with torch.no_grad():
            for parameter, features, labels, name in pairs:
                if features.ndim != 2 or features.shape[1] != self.semantic_dim:
                    raise ValueError(f"{name}_features have an invalid shape")
                if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                    raise ValueError(f"{name}_labels have an invalid shape")
                for factor in range(2):
                    selected = features[labels == factor]
                    if selected.shape[0] == 0:
                        raise ValueError(f"{name} factor {factor} has no samples")
                    parameter[factor].copy_(
                        F.normalize(selected.float().mean(dim=0), p=2, dim=0)
                    )

    def forward(self, full_images):
        local_images, theta = self.local_view(full_images)
        full_features = self._encode(full_images)
        local_features = self._encode(local_images)
        concatenated = torch.cat((full_features, local_features), dim=1)
        gate = self.view_gate(concatenated)
        gated_features = torch.cat(
            (full_features, gate * local_features), dim=1
        )

        output = {
            "full_features": full_features,
            "local_features": local_features,
            "theta": theta,
            "view_gate": gate,
        }
        if self.stage == "c0":
            diagnosis_logits = self.c0_head(gated_features)
            output.update(
                {
                    "diagnosis_logits": diagnosis_logits,
                    "diagnosis_log_probs": F.log_softmax(
                        diagnosis_logits.float(), dim=1
                    ),
                    "diagnosis_probs": F.softmax(diagnosis_logits.float(), dim=1),
                }
            )
            output["screen_probs"] = output["diagnosis_probs"][:, 1:].sum(dim=1)
            output["screen_logits"] = torch.logit(
                output["screen_probs"].clamp(1e-6, 1.0 - 1e-6)
            )
            return output

        screen_logits = self.screen_head(gated_features).squeeze(1)
        morph_features = F.normalize(self.morph_projector(gated_features), dim=1)
        evidence_features = F.normalize(
            self.evidence_projector(gated_features), dim=1
        )
        morph_semantic_logits = self.morph_head(morph_features)
        evidence_semantic_logits = self.evidence_head(evidence_features)
        morph_logits = morph_semantic_logits
        evidence_logits = evidence_semantic_logits

        if self.stage == "c2":
            morph_prototype_logits = self._prototype_logits(
                morph_features,
                self.morph_prototypes,
                self.log_morph_temperature,
            )
            evidence_prototype_logits = self._prototype_logits(
                evidence_features,
                self.evidence_prototypes,
                self.log_evidence_temperature,
            )
            weight = self.prototype_residual
            morph_logits = morph_semantic_logits + weight * morph_prototype_logits
            evidence_logits = (
                evidence_semantic_logits + weight * evidence_prototype_logits
            )
            output.update(
                {
                    "morph_prototype_logits": morph_prototype_logits,
                    "evidence_prototype_logits": evidence_prototype_logits,
                }
            )

        screen_probs = torch.sigmoid(screen_logits.float())
        morph_probs = F.softmax(morph_logits.float(), dim=1)
        evidence_probs = F.softmax(evidence_logits.float(), dim=1)
        diagnosis_probs = compose_tbs_probabilities(
            screen_probs, morph_probs, evidence_probs
        )
        output.update(
            {
                "diagnosis_logits": None,
                "diagnosis_probs": diagnosis_probs,
                "diagnosis_log_probs": diagnosis_probs.clamp_min(1e-12).log(),
                "screen_logits": screen_logits,
                "screen_probs": screen_probs,
                "morph_features": morph_features,
                "evidence_features": evidence_features,
                "morph_semantic_logits": morph_semantic_logits,
                "evidence_semantic_logits": evidence_semantic_logits,
                "morph_logits": morph_logits,
                "evidence_logits": evidence_logits,
                "morph_probs": morph_probs,
                "evidence_probs": evidence_probs,
            }
        )
        return output


def build_tbs_factorized_model(
    stage,
    model_name="caformer_s18",
    pretrained=True,
    semantic_dim=128,
    prototype_residual=0.25,
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the factorized model") from exc

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFactorizedDualSpaceModel(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
        stage=stage,
        prototype_residual=prototype_residual,
    )
