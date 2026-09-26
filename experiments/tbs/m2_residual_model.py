"""M2-Residual-OOF: zero-init residual experts for boundary pairs.

Expert-L (ASC-US/LSIL) and Expert-H (ASC-H/HSIL) each output a scalar
correction δ to M0 logits.  Experts are only activated when M0's top-2
fall within the target pair and the pair margin is below a threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualExpert(nn.Module):
    """Zero-init adapter → 2-dim correction for a specific boundary pair.

    δ = W2 @ GELU(W1 @ LN(h)), W2 zero-initialised.
    """

    def __init__(self, dim=512, bottleneck=64, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, 2)  # δ for pair classes
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        x = self.ln(h)
        x = self.down(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up(x)  # (B, 2)
        return x


class M2ResidualModel(nn.Module):
    """Frozen M0 + two residual experts for ASC-US/LSIL and ASC-H/HSIL.

    At inference, expert correction is applied ONLY when:
      - M0 top-2 are exactly {ASC-US, LSIL} → Expert-L
      - M0 top-2 are exactly {ASC-H, HSIL} → Expert-H
      - AND pair margin < activation threshold τ
    Otherwise, M0 output unchanged.
    """

    def __init__(self, backbone, m0_diagnosis_head, feature_dim=512, bottleneck=64):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Frozen M0 head
        self.m0_head = m0_diagnosis_head
        for p in self.m0_head.parameters():
            p.requires_grad = False

        # Two residual experts
        self.expert_low = ResidualExpert(feature_dim, bottleneck)
        self.expert_high = ResidualExpert(feature_dim, bottleneck)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self

    def forward(self, images, return_corrections=False):
        """Returns corrected logits and optionally corrections.

        During training: always apply experts for boundary-class samples.
        During eval: only apply when activation criteria are met.
        """
        with torch.no_grad():
            features = self.backbone(images)
            if features.ndim > 2:
                features = features.flatten(1)
            m0_logits = self.m0_head(features)  # (B, 5)

        # Expert corrections
        delta_low = self.expert_low(features)  # (B, 2)  [ASC-US, LSIL]
        delta_high = self.expert_high(features)  # (B, 2)  [ASC-H, HSIL]

        corrected = m0_logits.clone()
        # Apply corrections at the corresponding class indices
        corrected[:, 1] = corrected[:, 1] + delta_low[:, 0]  # ASC-US
        corrected[:, 2] = corrected[:, 2] + delta_low[:, 1]  # LSIL
        corrected[:, 3] = corrected[:, 3] + delta_high[:, 0]  # ASC-H
        corrected[:, 4] = corrected[:, 4] + delta_high[:, 1]  # HSIL

        if return_corrections:
            return corrected, m0_logits, delta_low, delta_high, features
        return corrected


def residual_loss(corrected_logits, m0_logits, delta_low, delta_high, labels,
                  low_mask, high_mask, easy_mask, lambda_kl=1.0, lambda_delta=0.01):
    """Combined loss for M2-Residual-OOF.

    - Pair CE on difficult boundary samples (low_mask, high_mask)
    - KL preservation on easy / non-target samples
    - L2 regularisation on δ magnitude
    """
    device = corrected_logits.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Pair CE on difficult samples for each expert
    if low_mask.any():
        # Low pair: classes 1,2 (ASC-US, LSIL)
        low_logits = corrected_logits[low_mask][:, [1, 2]]
        low_labels_mapped = labels[low_mask] - 1  # 1→0, 2→1
        ce_low = F.cross_entropy(low_logits, low_labels_mapped)
        total_loss = total_loss + ce_low

    if high_mask.any():
        # High pair: classes 3,4 (ASC-H, HSIL)
        high_logits = corrected_logits[high_mask][:, [3, 4]]
        high_labels_mapped = labels[high_mask] - 3  # 3→0, 4→1
        ce_high = F.cross_entropy(high_logits, high_labels_mapped)
        total_loss = total_loss + ce_high

    # 2. KL preservation on easy / non-boundary samples
    if easy_mask.any():
        m0_probs = torch.softmax(m0_logits[easy_mask].float(), dim=1)
        corrected_probs = torch.softmax(corrected_logits[easy_mask].float(), dim=1)
        kl = (m0_probs * (torch.log(m0_probs + 1e-12) - torch.log(corrected_probs + 1e-12))).sum(dim=1).mean()
        total_loss = total_loss + lambda_kl * kl

    # 3. L2 penalty on corrections
    if delta_low is not None:
        total_loss = total_loss + lambda_delta * (delta_low ** 2).mean()
    if delta_high is not None:
        total_loss = total_loss + lambda_delta * (delta_high ** 2).mean()

    return total_loss


def build_m2_residual(m0_checkpoint_path, device):
    """Build M2ResidualModel from frozen M0 checkpoint."""
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = M2ResidualModel(
        backbone=m0.backbone,
        m0_diagnosis_head=m0.diagnosis_head,
        feature_dim=m0.feature_dim,
    ).to(device)
    return model
