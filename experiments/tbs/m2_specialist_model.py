"""M2-Specialist: frozen M0 gate + zero-init adapter + conditional 4-class expert.

P(N) = 1 - a0          (a0 frozen from M0)
P(c) = a0 * q(c)       (q trained via adapter, abnormal samples only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitAdapter(nn.Module):
    """Residual adapter: h → LN → Linear→GELU→Linear(zero_init) → +h.

    With W2=0, h_adapted = h at initialisation (identity).
    """

    def __init__(self, dim=512, bottleneck=64, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        # Zero-init the up projection
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.down(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class M2SpecialistModel(nn.Module):
    """Frozen M0 backbone + gate + zero-init adapter + conditional expert."""

    def __init__(self, backbone, m0_diagnosis_head, feature_dim=512, num_classes=5,
                 adapter_bottleneck=64, adapter_dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Frozen M0 gate (5-class head, weights frozen)
        self.gate_head = m0_diagnosis_head  # nn.Linear(512, 5)
        for p in self.gate_head.parameters():
            p.requires_grad = False

        # Trainable adapter
        self.adapter = ZeroInitAdapter(feature_dim, adapter_bottleneck, adapter_dropout)

        # Conditional 4-class head (weights copied from M0 abnormal rows)
        self.conditional_head = nn.Linear(feature_dim, 4)

    def _init_conditional_head_from_m0(self):
        """Copy abnormal-class weights (rows 1-4) from frozen gate head."""
        with torch.no_grad():
            self.conditional_head.weight.copy_(self.gate_head.weight[1:5])
            self.conditional_head.bias.copy_(self.gate_head.bias[1:5])
        print("  Conditional head initialised from M0 abnormal weights", flush=True)

    def forward(self, images):
        """Returns (logits_5, q_logits, a0_probs)."""
        with torch.no_grad():
            features = self.backbone(images)
            if features.ndim > 2:
                features = features.flatten(1)
            # Frozen gate forward
            gate_logits = self.gate_head(features)  # (B, 5)
            gate_probs = torch.softmax(gate_logits.float(), dim=1)
            a0 = gate_probs[:, 1:].sum(dim=1)  # P(Abnormal), frozen

        # Adapted features for conditional expert
        adapted = self.adapter(features)  # zero-init → initially = features
        q_logits = self.conditional_head(adapted)  # (B, 4)

        return gate_logits, q_logits, a0

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self


def specialist_loss(q_logits, labels, a0, num_classes=5):
    """CE loss on abnormal samples only. Normal samples contribute 0.

    q_logits: (B, 4) — logits for ASC-US(0), LSIL(1), ASC-H(2), HSIL(3)
    labels: (B,) — 0=Normal, 1=ASC-US, 2=LSIL, 3=ASC-H, 4=HSIL
    a0: (B,) — frozen P(Abnormal) for each sample

    Returns scalar loss averaged over abnormal samples.
    """
    abnormal_mask = labels != 0  # Normal=0
    if abnormal_mask.sum() == 0:
        return torch.tensor(0.0, device=q_logits.device, requires_grad=True)

    # Map original labels (1-4) to conditional labels (0-3)
    cond_labels = labels[abnormal_mask] - 1  # 1→0, 2→1, 3→2, 4→3

    q_logits_abn = q_logits[abnormal_mask]  # (N_abn, 4)
    ce = F.cross_entropy(q_logits_abn, cond_labels, reduction="mean")
    return ce


def combine_probs(q_logits, a0):
    """Combine gate and conditional expert into 5-class probabilities.

    P(N) = 1 - a0
    P(c) = a0 × softmax(q_logits)[c]
    """
    q_probs = torch.softmax(q_logits.float(), dim=1)  # (B, 4)
    a0 = a0.float().unsqueeze(1)  # (B, 1)
    abnormal_probs = a0 * q_probs  # (B, 4)
    normal_prob = (1.0 - a0.squeeze(1)).unsqueeze(1)  # (B, 1)
    full_probs = torch.cat([normal_prob, abnormal_probs], dim=1)
    return full_probs


def build_m2_specialist(m0_checkpoint_path, device):
    """Build M2SpecialistModel from frozen M0 checkpoint."""
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = M2SpecialistModel(
        backbone=m0.backbone,
        m0_diagnosis_head=m0.diagnosis_head,
        feature_dim=m0.feature_dim,
    ).to(device)
    model._init_conditional_head_from_m0()
    return model
