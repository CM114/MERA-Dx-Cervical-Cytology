"""C0-R3 model with a locked conservative local translation trust region."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.tbs.c0_model import LocalCellViewGenerator
from experiments.tbs.c0r1_model import TBSC0R1ResidualModel


class StableLocalCellViewGenerator(LocalCellViewGenerator):
    """Original affine localizer with only the translation bound tightened."""

    def __init__(self, in_channels=3, initial_scale=0.60, translation_bound=0.12):
        super().__init__(in_channels=in_channels, initial_scale=initial_scale)
        bound = float(translation_bound)
        if not 0.0 < bound <= 0.35:
            raise ValueError("translation_bound must be within (0, 0.35]")
        self.translation_bound = bound

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        raw = self.affine(self.features(images)).view(-1, 6)
        bounded = torch.stack(
            (
                self.initial_scale + 0.12 * torch.tanh(raw[:, 0]),
                0.08 * torch.tanh(raw[:, 1]),
                self.translation_bound * torch.tanh(raw[:, 2]),
                0.08 * torch.tanh(raw[:, 3]),
                self.initial_scale + 0.12 * torch.tanh(raw[:, 4]),
                self.translation_bound * torch.tanh(raw[:, 5]),
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


class TBSC0R3ResidualModel(TBSC0R1ResidualModel):
    """C0-R2F residual model with a conservative local translation bound."""

    def __init__(self, backbone, feature_dim, local_translation_bound=0.12, **kwargs):
        super().__init__(backbone, feature_dim, **kwargs)
        self.local_view = StableLocalCellViewGenerator(
            initial_scale=kwargs.get("initial_local_scale", 0.60),
            translation_bound=local_translation_bound,
        )
        self.local_translation_bound = float(local_translation_bound)


def build_tbs_c0r3_model(model_name="caformer_s18", pretrained=True, local_translation_bound=0.12):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the C0-R3 model") from exc
    backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSC0R3ResidualModel(
        backbone,
        feature_dim,
        local_translation_bound=local_translation_bound,
        residual_logit_bound=0.10,
    )


__all__ = (
    "StableLocalCellViewGenerator",
    "TBSC0R3ResidualModel",
    "build_tbs_c0r3_model",
)
