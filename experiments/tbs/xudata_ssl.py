"""Small SimSiam components used by the xudata-only pretraining route."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # Keep pure CLI tests importable on the CPU workstation.
    torch = None
    nn = None
    F = None


def negative_cosine_similarity(prediction, target):
    if torch is None:
        raise RuntimeError("PyTorch is required for SimSiam training")
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return -(prediction * target).sum(dim=1).mean()


if nn is not None:

    class _ProjectionMLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False),
            )

        def forward(self, values):
            return self.layers(values)


    class _PredictionMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, values):
            return self.layers(values)


    class SimSiamModel(nn.Module):
        def __init__(self, backbone, feature_dim, projection_dim=256, hidden_dim=512):
            super().__init__()
            self.backbone = backbone
            self.projector = _ProjectionMLP(feature_dim, hidden_dim, projection_dim)
            self.predictor = _PredictionMLP(projection_dim, hidden_dim)

        def _encode(self, images):
            values = self.backbone(images)
            if values.ndim > 2:
                values = values.flatten(1)
            return self.projector(values)

        def forward(self, first_view, second_view):
            first_z = self._encode(first_view)
            second_z = self._encode(second_view)
            first_p = self.predictor(first_z)
            second_p = self.predictor(second_z)
            loss = 0.5 * (
                negative_cosine_similarity(first_p, second_z)
                + negative_cosine_similarity(second_p, first_z)
            )
            return loss, first_p, second_p

else:

    class SimSiamModel:  # pragma: no cover - exercised on the server.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SimSiam training")


def build_ssl_backbone(model_name="caformer_s18", pretrained=True):
    if torch is None:
        raise RuntimeError("PyTorch and timm are required for SimSiam training")
    import timm

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return backbone, int(feature_dim)
