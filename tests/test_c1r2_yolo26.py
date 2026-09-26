import sys
import types

import pytest


def test_c1r2_factory_accepts_yolo26_backbone(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeClassify(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 16, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(16, 1000)

    class FakeYOLO:
        def __init__(self, source):
            assert source == "yolo26l-cls.yaml"
            self.model = types.SimpleNamespace(
                model=nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), FakeClassify())
            )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    from experiments.C1R2.model import build_c1r2_model
    from experiments.C1R2.protocol import LOCKED_C1R2_CONFIG

    assert LOCKED_C1R2_CONFIG["model_name"] == "yolo26l-cls"
    model = build_c1r2_model(
        torch.device("cpu"), model_name="yolo26l-cls", pretrained=False
    )
    output = model(torch.randn(2, 3, 32, 32))

    assert model.feature_dim == 16
    assert output["p_final"].shape == (2, 5)
    assert torch.allclose(output["p_final"].sum(dim=1), torch.ones(2), atol=1e-6)

