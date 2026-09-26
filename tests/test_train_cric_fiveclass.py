import pytest


def test_meradx_direct_ce_mode_uses_only_final_diagnosis_loss():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from experiments.train_cric_fiveclass import _loss_for_output

    logits = torch.log_softmax(torch.randn(3, 5), dim=1).requires_grad_()
    output = {"diagnosis_log_probs": logits}
    labels = torch.tensor([0, 2, 4], dtype=torch.long)

    loss, parts = _loss_for_output(
        "mera_dx", output, labels, objective="direct_ce"
    )

    assert torch.allclose(loss, F.nll_loss(logits, labels))
    assert set(parts) == {"diagnosis_loss"}
    loss.backward()
    assert logits.grad is not None
