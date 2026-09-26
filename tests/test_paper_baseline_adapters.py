import importlib.util

import pytest

from experiments.paper_baseline_adapters import build_method


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is installed on the remote trainer")
def test_a2sdnet121_outputs_five_logits():
    import torch

    model = build_method("A2SDNet121", pretrained=False)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is installed on the remote trainer")
def test_msenet_outputs_five_logits_with_vgg_member():
    import torch

    model = build_method("MSENet", pretrained=False)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 5)


def test_unknown_method_is_rejected():
    with pytest.raises(KeyError):
        build_method("unknown-method", pretrained=False)
