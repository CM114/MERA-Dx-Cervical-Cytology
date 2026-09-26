"""M0-compatible dual-view C0 model with a bounded local-cell view."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tbs.c0_loss import localizer_geometry_loss


class LocalCellViewGenerator(nn.Module):
    """Predict a conservative affine local-cell view from a full image."""

    def __init__(self, in_channels=3, initial_scale=0.60):
        super().__init__()
        scale = float(initial_scale)
        if not 0.0 < scale < 1.0:
            raise ValueError("initial_scale must be within (0, 1)")
        self.initial_scale = scale
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
            torch.tensor([scale, 0.0, 0.0, 0.0, scale, 0.0])
        )

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        raw = self.affine(self.features(images)).view(-1, 6)
        # Hard bounds keep the learned view local even if the geometry loss is
        # temporarily underweighted or an affine head becomes unstable.
        bounded = torch.stack(
            (
                self.initial_scale + 0.12 * torch.tanh(raw[:, 0]),
                0.08 * torch.tanh(raw[:, 1]),
                0.35 * torch.tanh(raw[:, 2]),
                0.08 * torch.tanh(raw[:, 3]),
                self.initial_scale + 0.12 * torch.tanh(raw[:, 4]),
                0.35 * torch.tanh(raw[:, 5]),
            ),
            dim=1,
        )
        theta = bounded.view(-1, 2, 3)
        local_images = F.grid_sample(
            images,
            F.affine_grid(theta, images.size(), align_corners=False),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return local_images, theta


class TBSDualViewC0Model(nn.Module):
    """Shared-backbone full/local fusion with an M0-compatible five-class head."""

    def __init__(
        self,
        backbone,
        feature_dim,
        initial_local_scale=0.60,
        activation_checkpointing=True,
    ):
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.local_view = LocalCellViewGenerator(initial_scale=initial_local_scale)
        self.view_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )
        self.diagnosis_head = nn.Linear(self.feature_dim * 2, 5)
        # The local half is zero, so the initial fused classifier equals full-only.
        nn.init.zeros_(self.diagnosis_head.weight[:, self.feature_dim :])
        nn.init.zeros_(self.diagnosis_head.bias)

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
    def _logits_to_output(logits):
        log_probs = F.log_softmax(logits.float(), dim=1)
        return log_probs, log_probs.exp()

    def forward(self, images):
        local_images, theta = self.local_view(images)
        full_features = self._encode(images)
        local_features = self._encode(local_images)
        concatenated = torch.cat((full_features, local_features), dim=1)
        view_gate = self.view_gate(concatenated)
        gated_local = view_gate * local_features
        zero_local = torch.zeros_like(gated_local)
        zero_full = torch.zeros_like(full_features)

        full_logits = self.diagnosis_head(torch.cat((full_features, zero_local), dim=1))
        local_logits = self.diagnosis_head(torch.cat((zero_full, gated_local), dim=1))
        diagnosis_logits = self.diagnosis_head(
            torch.cat((full_features, gated_local), dim=1)
        )
        diagnosis_log_probs, diagnosis_probs = self._logits_to_output(diagnosis_logits)
        full_log_probs, full_probs = self._logits_to_output(full_logits)
        local_log_probs, local_probs = self._logits_to_output(local_logits)
        determinant = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        local_area = determinant.abs()
        local_translation_abs = theta[:, :, 2].abs().mean(dim=1)
        local_cosine_similarity = F.cosine_similarity(full_features, local_features, dim=1)
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
            "local_diagnosis_logits": local_logits,
            "local_diagnosis_log_probs": local_log_probs,
            "local_diagnosis_probs": local_probs,
            "full_features": full_features,
            "local_features": local_features,
            "view_gate": view_gate,
            "theta": theta,
            "local_area": local_area,
            "local_translation_abs": local_translation_abs,
            "local_cosine_similarity": local_cosine_similarity,
            "geometry_loss": localizer_geometry_loss(theta),
        }


def build_tbs_c0_model(model_name="caformer_s18", pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the C0 model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSDualViewC0Model(backbone, feature_dim)


__all__ = (
    "LocalCellViewGenerator",
    "TBSDualViewC0Model",
    "build_tbs_c0_model",
)
