"""RGB-only learnable local-view model for xudata."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # Allow parser and static tests without the GPU stack.
    torch = None
    nn = None
    F = None


if nn is not None:

    class SpatialTransformerLocalizer(nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 6),
            )
            nn.init.zeros_(self.fc[-1].weight)
            self.fc[-1].bias.data.copy_(
                torch.tensor([0.70, 0.0, 0.0, 0.0, 0.70, 0.0])
            )

        def forward(self, images):
            return self.fc(self.features(images)).view(-1, 2, 3)


    class XUDataDualViewClassifier(nn.Module):
        def __init__(self, shared_backbone, feature_dim, m0_head, num_classes=5):
            super().__init__()
            self.backbone = shared_backbone
            self.feature_dim = int(feature_dim)
            self.localizer = SpatialTransformerLocalizer(3)
            self.fusion = nn.Linear(self.feature_dim * 2, num_classes)
            with torch.no_grad():
                self.fusion.weight.zero_()
                self.fusion.weight[:, : self.feature_dim].copy_(m0_head.weight)
                self.fusion.bias.copy_(m0_head.bias)

        def _features(self, images):
            values = self.backbone(images)
            if values.ndim > 2:
                values = values.flatten(1)
            return values

        def forward(self, full_images):
            theta = self.localizer(full_images)
            grid = F.affine_grid(
                theta,
                size=full_images.size(),
                align_corners=False,
            )
            local_images = F.grid_sample(
                full_images,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            full_features = self._features(full_images)
            local_features = self._features(local_images)
            logits = self.fusion(torch.cat([full_features, local_features], dim=1))
            return {
                "logits": logits,
                "full_features": full_features,
                "local_features": local_features,
                "theta": theta,
            }

else:

    class SpatialTransformerLocalizer:  # pragma: no cover - server only.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for the dual-view model")


    class XUDataDualViewClassifier:  # pragma: no cover - server only.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for the dual-view model")


def build_xudata_dualview_from_m0(checkpoint, device, model_name="caformer_s18"):
    if torch is None:
        raise RuntimeError("PyTorch is required for the dual-view model")
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model("m0", model_name=model_name, pretrained=False).to(device)
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_state", payload.get("state_dict", payload))
    m0.load_state_dict(state, strict=True)
    return XUDataDualViewClassifier(
        m0.backbone,
        m0.feature_dim,
        m0.diagnosis_head,
    ).to(device)
