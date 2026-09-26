"""M4-NucProto: frozen M0 + nucleus spatial pooling + learnable class prototypes.

Variants:
  P0: M0 global features + original linear head (baseline replay)
  P1: global features + fixed train centroids → cosine
  P2: global features + learnable prototypes → cosine
  P3: global + nucleus spatial + context spatial → learnable prototypes → cosine
  P4: global + shifted-nucleus spatial (control) → learnable prototypes → cosine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialMaskPool(nn.Module):
    """Pool features from masked regions of a feature map."""

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map, mask, valid_mask=None):
        """feat_map: (B, C, H, W), mask: (B, H, W) in [0,1], valid_mask: (B,) bool"""
        if feat_map.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1), size=feat_map.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        m = mask.clamp(0, 1).unsqueeze(1)  # (B, 1, H, W)
        pooled = (feat_map * m).sum(dim=[2, 3]) / m.sum(dim=[2, 3]).clamp(min=1)  # (B, C)
        h = self.proj(pooled.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (B, out)
        if valid_mask is not None:
            h = h * valid_mask.float().unsqueeze(1)
        return h


class M4NucProtoModel(nn.Module):
    """M4-NucProto with optional nucleus/context spatial branches.

    Architecture:
      Frozen M0 backbone → h_global (512-d)
      [nucleus mask] → stage3 pool → h_nuc (128-d)
      [nucleus mask] → stage3 context pool → h_ctx (128-d)
      Fusion: h = proj_global(h_global) + g_n * h_nuc + g_c * h_ctx
      Cosine: logits = τ * cos(h, prototypes)
    """

    def __init__(self, backbone, m0_diagnosis_head, feature_dim=512,
                 stage3_channels=512, proto_dim=128, num_classes=5,
                 variant="P3"):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.variant = variant

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Frozen M0 head for P0
        self.m0_head = m0_diagnosis_head
        for p in self.m0_head.parameters():
            p.requires_grad = False

        # Global projection
        self.proj_global = nn.Linear(feature_dim, proto_dim)

        # Spatial branches (P3, P4)
        if variant in ("P3", "P4"):
            self.nuc_pool = SpatialMaskPool(stage3_channels, proto_dim)
            self.ctx_pool = SpatialMaskPool(stage3_channels, proto_dim)
            self.gate_n = nn.Parameter(torch.tensor(0.5))
            self.gate_c = nn.Parameter(torch.tensor(0.5))
        else:
            self.nuc_pool = None
            self.ctx_pool = None

        # Learnable prototypes (P2, P3, P4) — initialised later
        if variant != "P0":
            self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
            self.log_tau = nn.Parameter(torch.tensor(2.3026))  # ln(10)

        # Stage3 hook
        self._stage3_feat = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, inp, out):
            self._stage3_feat = out
        for name, mod in self.backbone.named_modules():
            if name.endswith("stages.3") and name.count("stages") == 1:
                mod.register_forward_hook(hook_fn)
                return

    def init_prototypes_from_train(self, train_loader, device, amp_enabled):
        """Initialise prototypes as normalised train class centroids."""
        if self.variant == "P0":
            return
        sums = torch.zeros(self.num_classes, self.proto_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)
        self.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    features = self.backbone(images)
                    if features.ndim > 2:
                        features = features.flatten(1)
                    h = self.proj_global(features)
                    h_norm = F.normalize(h, p=2, dim=1)
                for c in range(self.num_classes):
                    mask = labels == c
                    if mask.any():
                        sums[c] += h_norm[mask].sum(dim=0)
                        counts[c] += mask.sum()
        for c in range(self.num_classes):
            if counts[c] > 0:
                self.prototypes.data[c] = F.normalize(sums[c] / counts[c], p=2, dim=0)
        print(f"  Prototypes initialised from train centroids", flush=True)

    def forward(self, images, nucleus_masks=None, mask_valid=None):
        self._stage3_feat = None

        with torch.no_grad():
            features = self.backbone(images)
            if features.ndim > 2:
                features = features.flatten(1)

        # P0: original M0 linear head
        if self.variant == "P0":
            with torch.no_grad():
                logits = self.m0_head(features)
            return logits, None

        # Project global features
        h_global = self.proj_global(features)  # (B, D)

        # Add spatial branches
        h_nuc = torch.zeros_like(h_global)
        h_ctx = torch.zeros_like(h_global)

        if self.variant in ("P3", "P4") and self._stage3_feat is not None and nucleus_masks is not None:
            gate_n_val = torch.sigmoid(self.gate_n)
            gate_c_val = torch.sigmoid(self.gate_c)
            if mask_valid is not None:
                valid = mask_valid.bool()
            else:
                valid = torch.ones(features.size(0), dtype=torch.bool, device=features.device)

            h_nuc = gate_n_val * self.nuc_pool(self._stage3_feat, nucleus_masks, valid)
            # Context = 1 - nucleus, but exclude padding (mask==0 region that's also background)
            ctx_mask = 1.0 - nucleus_masks.clamp(0, 1)
            h_ctx = gate_c_val * self.ctx_pool(self._stage3_feat, ctx_mask, valid)

        # Fusion
        h = h_global + h_nuc + h_ctx
        h_norm = F.normalize(h, p=2, dim=1)  # (B, D)

        # Cosine classification
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)  # (5, D)
        tau = F.softplus(self.log_tau) + 0.1  # ensure τ > 0
        logits = tau * (h_norm @ proto_norm.T)  # (B, 5)

        return logits, {"h_norm": h_norm, "proto_norm": proto_norm, "tau": tau}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self


def build_m4_nucproto(m0_checkpoint_path, device, variant="P3"):
    """Build M4NucProtoModel from frozen M0 checkpoint."""
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = M4NucProtoModel(
        backbone=m0.backbone,
        m0_diagnosis_head=m0.diagnosis_head,
        feature_dim=m0.feature_dim,
        stage3_channels=512,
        proto_dim=128,
        variant=variant,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Variant={variant}  Trainable: {trainable:,}", flush=True)
    return model
