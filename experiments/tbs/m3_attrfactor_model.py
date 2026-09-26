"""M3-AttrFactor: spatial nucleus pooling + weak morphology attributes + family-specific evidence.

P(Normal) = a0 (frozen M0 gate)
P(ASC-US) = a0 × (1-m) × (1-e_low)
P(LSIL)   = a0 × (1-m) × e_low
P(ASC-H)  = a0 × m × (1-e_high)
P(HSIL)   = a0 × m × e_high
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialNucleusPool(nn.Module):
    """Pool features from nucleus and cytoplasm regions using predicted mask.

    mask: (B, H, W) float in [0,1], same spatial size as feature map.
    Returns h_nuc (mean-pooled nucleus features) and h_cyto (mean-pooled cytoplasm).
    """

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.nuc_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cyto_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map, mask):
        """feat_map: (B, C, H, W), mask: (B, H, W)"""
        if feat_map.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1), size=feat_map.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        m = mask.clamp(0, 1)
        m_exp = m.unsqueeze(1)  # (B, 1, H, W)

        # Nucleus pooling
        nuc_sum = (feat_map * m_exp).sum(dim=[2, 3])
        nuc_area = m_exp.sum(dim=[2, 3]).clamp(min=1)
        h_nuc = nuc_sum / nuc_area  # (B, C)

        # Cytoplasm pooling (1 - mask)
        cyto_m = (1.0 - m).unsqueeze(1)
        cyto_sum = (feat_map * cyto_m).sum(dim=[2, 3])
        cyto_area = cyto_m.sum(dim=[2, 3]).clamp(min=1)
        h_cyto = cyto_sum / cyto_area  # (B, C)

        h_nuc = self.nuc_proj(h_nuc.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        h_cyto = self.cyto_proj(h_cyto.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return h_nuc, h_cyto


class MorphologyHead(nn.Module):
    """Predict 4 weak morphology attributes from spatial nucleus features."""

    def __init__(self, in_dim=256, hidden=64, n_attrs=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_attrs),  # nucleus_area, NC_ratio, border_grad, stain_intensity
        )
        self.n_attrs = n_attrs

    def forward(self, h_nuc, h_cyto):
        x = torch.cat([h_nuc, h_cyto], dim=1)
        return self.mlp(x)


class AttrFactorModel(nn.Module):
    """M3-AttrFactor: spatial morphology + family-specific evidence."""

    def __init__(self, backbone, m0_diagnosis_head, feature_dim=512,
                 stage2_channels=320, stage3_channels=320, morph_hidden=64):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Freeze backbone stem/stage1 (done externally via freeze_stages)

        # Frozen M0 gate
        self.m0_head = m0_diagnosis_head
        for p in self.m0_head.parameters():
            p.requires_grad = False

        # Spatial pooling on stage2 + stage3
        self.pool_stage2 = SpatialNucleusPool(stage2_channels, 128)
        self.pool_stage3 = SpatialNucleusPool(stage3_channels, 128)

        # Morphology head (weak supervision from M4-C2plus attributes)
        self.morph_head = MorphologyHead(128 + 128, morph_hidden)

        # Family-specific evidence + morphology gate (reading global + morph features)
        fusion_dim = feature_dim + 64  # h_global + morph_feat
        self.m_proj = nn.Linear(fusion_dim, 1)  # P(high-grade morphology)
        self.e_low_proj = nn.Linear(fusion_dim, 1)  # P(LSIL criteria | low)
        self.e_high_proj = nn.Linear(fusion_dim, 1)  # P(HSIL criteria | high)

        # Stage2/3 feature hooks
        self._stage2_feat = None
        self._stage3_feat = None
        self._register_hooks()

    def _register_hooks(self):
        def hook2(module, inp, out):
            self._stage2_feat = out
        def hook3(module, inp, out):
            self._stage3_feat = out
        for name, mod in self.backbone.named_modules():
            if name.endswith("stages.2") and name.count("stages") == 1:
                mod.register_forward_hook(hook2)
            if name.endswith("stages.3") and name.count("stages") == 1:
                mod.register_forward_hook(hook3)

    def freeze_stages(self, frozen_prefixes=("stages.0", "stages.1")):
        for pname, param in self.backbone.named_parameters():
            if any(pname.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        return frozen, trainable

    def forward(self, images, nucleus_masks=None):
        """Returns full 5-class probs, plus morph_pred, m, e_low, e_high for loss computation.

        nucleus_masks: (B, H, W) float tensor in [0,1], same spatial size as input images.
        """
        self._stage2_feat = None
        self._stage3_feat = None

        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)

        # Frozen M0 gate
        with torch.no_grad():
            m0_logits = self.m0_head(features)
            m0_probs = torch.softmax(m0_logits.float(), dim=1)
            a0 = m0_probs[:, 1:].sum(dim=1)  # P(Abnormal), frozen

        # Spatial pooling
        morph_pred = None
        morph_feat = torch.zeros(features.size(0), 64, device=features.device)

        if self._stage2_feat is not None and self._stage3_feat is not None and nucleus_masks is not None:
            h_nuc2, h_cyto2 = self.pool_stage2(self._stage2_feat, nucleus_masks)
            h_nuc3, h_cyto3 = self.pool_stage3(self._stage3_feat, nucleus_masks)
            h_nuc = h_nuc2 + h_nuc3
            h_cyto = h_cyto2 + h_cyto3
            morph_pred = self.morph_head(h_nuc, h_cyto)
            # Pool morph features through a small MLP to get 64-d
            morph_feat = self.morph_head.mlp[0](torch.cat([h_nuc, h_cyto], dim=1))

        # Family-specific evidence
        fusion = torch.cat([features, morph_feat], dim=1)
        m_logit = self.m_proj(fusion).squeeze(1)  # (B,)
        e_low_logit = self.e_low_proj(fusion).squeeze(1)
        e_high_logit = self.e_high_proj(fusion).squeeze(1)

        m = torch.sigmoid(m_logit)
        e_low = torch.sigmoid(e_low_logit)
        e_high = torch.sigmoid(e_high_logit)

        # Factorised probabilities
        a0_clamped = a0.float().clamp(1e-8, 1 - 1e-8)
        p_ascus = a0_clamped * (1 - m) * (1 - e_low)
        p_lsil = a0_clamped * (1 - m) * e_low
        p_asch = a0_clamped * m * (1 - e_high)
        p_hsil = a0_clamped * m * e_high
        p_normal = 1.0 - a0_clamped

        probs = torch.stack([p_normal, p_ascus, p_lsil, p_asch, p_hsil], dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return probs, morph_pred, m, e_low, e_high, m0_probs

    def train(self, mode=True):
        super().train(mode)
        return self


def build_attrfactor(m0_checkpoint_path, device):
    """Build AttrFactorModel from M0 checkpoint."""
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = AttrFactorModel(
        backbone=m0.backbone,
        m0_diagnosis_head=m0.diagnosis_head,
        feature_dim=m0.feature_dim,
        stage2_channels=320,
        stage3_channels=512,
    ).to(device)

    frozen, trainable = model.freeze_stages(frozen_prefixes=("stages.0", "stages.1"))
    print(f"  Frozen: {frozen:,}  Trainable backbone: {trainable:,} params", flush=True)
    return model
