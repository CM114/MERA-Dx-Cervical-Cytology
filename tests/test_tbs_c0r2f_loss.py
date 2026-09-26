import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.c0r2f_loss import c0r2f_loss


@unittest.skipIf(torch is None, "PyTorch is required for C0-R2F loss tests")
class C0R2FLossTests(unittest.TestCase):
    def test_boundary_terms_are_taken_from_full_view_branch(self):
        fused = torch.log_softmax(torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]), dim=1)
        full = torch.log_softmax(torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]]), dim=1)
        output = {
            "diagnosis_log_probs": fused,
            "full_diagnosis_log_probs": full,
            "theta": torch.tensor([[[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]]]),
            "residual_logits": torch.zeros(1, 5),
        }
        result = c0r2f_loss(output, torch.tensor([1]))
        self.assertLess(float(result["low_grade_pair_loss"]), 0.01)


if __name__ == "__main__":
    unittest.main()
