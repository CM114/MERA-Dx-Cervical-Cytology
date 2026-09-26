"""M1-Seg-Joint: CaFormer-S18 with joint nucleus segmentation auxiliary.

Key difference from old M1-Seg: backbone stage2 receives gradients from
both classification AND segmentation losses.  The core hypothesis -- that
pixel-level nucleus supervision shapes intermediate spatial features -- is
now actually tested.

Gradient flow:
  L_seg → decoder → stage2 → stage1 (frozen) → stem (frozen)
  L_cls → classifier → stage4 → stage3 → stage2 → stage1 (frozen) → stem (frozen)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
    """stage2 features (320ch, 14×14) → 56×56 binary nucleus mask."""

    def __init__(self, in_channels=320, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv3(x).squeeze(1)


class M1SegJointModel(nn.Module):
    """M1-Seg with REAL gradient flow from seg decoder into backbone stage2."""

    def __init__(self, backbone, feature_dim=512, seg_in_channels=320, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Register persistent hook on stage2 (must NOT detach)
        self._stage2_feat = None
        self._register_stage2_hook()

        self.diagnosis_head = nn.Linear(feature_dim, num_classes)
        self.seg_decoder = SegDecoder(seg_in_channels)

    def _register_stage2_hook(self):
        def hook_fn(module, inp, out):
            self._stage2_feat = out  # NO detach() — gradient must flow
        for name, mod in self.backbone.named_modules():
            if name.endswith("stages.2") and name.count("stages") == 1:
                mod.register_forward_hook(hook_fn)
                return
        raise RuntimeError("Cannot find backbone stage2 module")

    def freeze_stages(self, frozen_prefixes=("stages.0", "stages.1")):
        """Freeze only specified stages by prefix match on parameter names."""
        for pname, param in self.backbone.named_parameters():
            if any(pname.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
        frozen, trainable = 0, 0
        for p in self.backbone.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        return frozen, trainable

    def forward(self, images):
        """Returns (logits, mask_pred). mask_pred is None when not training."""
        self._stage2_feat = None

        # NO torch.no_grad() — seg loss must backprop through backbone
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)

        logits = self.diagnosis_head(features)

        mask_pred = None
        if self.training and self._stage2_feat is not None:
            mask_pred = self.seg_decoder(self._stage2_feat)

        return logits, mask_pred


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def dice_loss(pred_logits, target, smooth=1.0):
    pred = torch.sigmoid(pred_logits)
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def seg_loss_weighted(mask_pred, mask_gt, quality_weights=None):
    """BCE + Dice, optionally weighted per sample."""
    bce = F.binary_cross_entropy_with_logits(mask_pred, mask_gt, reduction="none")
    bce = bce.mean(dim=[1, 2])  # per-sample mean
    dice = dice_loss(mask_pred, mask_gt)
    # dice_loss already returns scalar mean over batch; we need per-sample
    # Recompute per-sample dice for weighting
    pred = torch.sigmoid(mask_pred)
    p = pred.reshape(pred.shape[0], -1)
    t = mask_gt.reshape(mask_gt.shape[0], -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1) + 1.0
    dice_per_sample = 1.0 - (2.0 * inter + 1.0) / denom

    loss_per_sample = bce + dice_per_sample

    if quality_weights is not None:
        w = quality_weights.to(loss_per_sample.device)
        return (loss_per_sample * w).sum() / (w.sum() + 1e-8)
    return loss_per_sample.mean()


def compute_quality_weights(mask_tensors, nonempty_flags, edge_contact_flags, component_counts):
    """q=1 for normal, q=0.5 for edge-touching, q=0 for empty/near-full."""
    weights = torch.ones(len(mask_tensors), device=mask_tensors.device)
    for i in range(len(mask_tensors)):
        if not nonempty_flags[i]:
            weights[i] = 0.0
            continue
        area = mask_tensors[i].sum()
        if area < 10 or area > mask_tensors[i].numel() * 0.95:
            weights[i] = 0.0
        elif edge_contact_flags[i] or component_counts[i] > 3:
            weights[i] = 0.5
    return weights


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_m1_seg_joint(m0_checkpoint_path, device, freeze_stem_stage1=True):
    """Build M1SegJointModel from a frozen M0 checkpoint.

    Parameters
    ----------
    m0_checkpoint_path : Path
    device : torch.device
    freeze_stem_stage1 : bool
        If True, freeze stem and stage1; stage2/3/4 + head + decoder trainable.

    Returns
    -------
    M1SegJointModel
    """
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = M1SegJointModel(m0.backbone, feature_dim=m0.feature_dim).to(device)

    if freeze_stem_stage1:
        frozen_n, trainable_n = model.freeze_stages(frozen_prefixes=("stages.0", "stages.1"))
        print(f"  Frozen: {frozen_n:,}  Trainable: {trainable_n:,} params", flush=True)
        # Verify stage2 has requires_grad=True
        for name, p in model.backbone.named_parameters():
            if "stages.2" in name and p.requires_grad is False:
                raise RuntimeError(f"stage2 param {name} is frozen but must be trainable")

    return model
