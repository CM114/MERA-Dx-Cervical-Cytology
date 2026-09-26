import sys
import types

import pytest


def _fake_ultralytics(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeClassify(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 16, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(16, 1000)

        def forward(self, x):
            return self.linear(self.pool(self.conv(x)).flatten(1))

    class FakeYOLO:
        def __init__(self, source):
            self.source = source
            self.model = types.SimpleNamespace(
                model=nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    FakeClassify(),
                )
            )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return torch


def test_yolo26_backbone_returns_pooled_features(monkeypatch):
    torch = _fake_ultralytics(monkeypatch)
    from experiments.tbs.yolo26_backbone import YOLO26ClassificationBackbone

    backbone = YOLO26ClassificationBackbone(pretrained=False)
    features = backbone(torch.randn(2, 3, 32, 32))

    assert backbone.feature_dim == 16
    assert features.shape == (2, 16)
    assert torch.isfinite(features).all()


def test_yolo26_backbone_uses_yaml_without_pretraining(monkeypatch):
    _fake_ultralytics(monkeypatch)
    from experiments.tbs.yolo26_backbone import YOLO26ClassificationBackbone

    backbone = YOLO26ClassificationBackbone(pretrained=False)

    assert backbone.model_source == "yolo26l-cls.yaml"


def test_yolo26_backbone_rejects_missing_classification_head(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeYOLO:
        def __init__(self, source):
            self.model = types.SimpleNamespace(model=nn.Sequential(nn.Conv2d(3, 8, 1)))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    from experiments.tbs.yolo26_backbone import YOLO26ClassificationBackbone

    with pytest.raises(ValueError, match="classification head"):
        YOLO26ClassificationBackbone(pretrained=False)

