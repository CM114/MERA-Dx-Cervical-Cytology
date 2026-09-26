"""Full-view-primary C0-R1 model with a bounded local residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tbs.c0_loss import localizer_geometry_loss
from experiments.tbs.c0_model import LocalCellViewGenerator


class TBSC0R1ResidualModel(nn.Module):
    """Shared full/local encoder with a hard-bounded zero-init correction."""

    def __init__(
        self,
        backbone,
        feature_dim,
        initial_local_scale=0.60,
        residual_logit_bound=0.10,
        activation_checkpointing=True,
    ):
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        bound = float(residual_logit_bound)
        if not 0.0 < bound <= 0.10:
            raise ValueError("residual_logit_bound must be within (0, 0.10]")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.residual_logit_bound = bound
        self.activation_checkpointing = bool(activation_checkpointing)
        self.local_view = LocalCellViewGenerator(initial_scale=initial_local_scale)
        self.full_head = nn.Linear(self.feature_dim, 5)
        self.residual_head = nn.Linear(self.feature_dim, 5)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

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
    def _probabilities(logits):
        log_probs = F.log_softmax(logits.float(), dim=1)
        return log_probs, log_probs.exp()

    def forward(self, images):
        local_images, theta = self.local_view(images)
        full_features = self._encode(images)
        local_features = self._encode(local_images)
        full_logits = self.full_head(full_features)
        raw_residual = self.residual_head(local_features)
        residual_logits = self.residual_logit_bound * torch.tanh(raw_residual)
        diagnosis_logits = full_logits + residual_logits
        diagnosis_log_probs, diagnosis_probs = self._probabilities(diagnosis_logits)
        full_log_probs, full_probs = self._probabilities(full_logits)
        determinant = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        return {
            "diagnosis_logits": diagnosis_logits,
            "diagnosis_log_probs": diagnosis_log_probs,
            "diagnosis_probs": diagnosis_probs,
            "screen_probs": diagnosis_probs[:, 1:].sum(dim=1),
            "screen_logits": torch.logit(
                diagnosis_probs[:, 1:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
            ),
            "full_diagnosis_logits": full_logits,
            "full_diagnosis_log_probs": full_log_probs,
            "full_diagnosis_probs": full_probs,
            "residual_logits": residual_logits,
            "residual_logit_abs_mean": residual_logits.abs().mean(),
            "residual_logit_max_abs": residual_logits.abs().max(),
            "full_features": full_features,
            "local_features": local_features,
            "theta": theta,
            "local_area": determinant.abs(),
            "local_translation_abs": theta[:, :, 2].abs().mean(dim=1),
            "local_cosine_similarity": F.cosine_similarity(
                full_features, local_features, dim=1
            ),
            "geometry_loss": localizer_geometry_loss(theta),
        }


def build_tbs_c0r1_model(model_name="caformer_s18", pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the C0-R1 model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSC0R1ResidualModel(backbone, feature_dim)


__all__ = ("TBSC0R1ResidualModel", "build_tbs_c0r1_model")
