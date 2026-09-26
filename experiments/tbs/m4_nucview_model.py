"""M4-NucView: RGB-level nucleus ROI extraction + lightweight local encoder + zero-init residual.

z = z0 + g(x) * Δz   where Δz is zero-init at start.
Variants: V0=M0 baseline, V1=real nucleus ROI, V2=center ROI, V3=random ROI, V4=mismatched ROI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# ROI extraction helpers (operate on numpy/PIL before tensor conversion)
# ---------------------------------------------------------------------------


def extract_nucleus_roi(rgb_pil, mask, expansion=1.15, output_size=128):
    """Extract nucleus ROI from RGB image using binary mask bounding box.

    Returns PIL Image (output_size × output_size, RGB) or None if mask empty.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    y1, y2 = rows.min(), rows.max() + 1
    x1, x2 = cols.min(), cols.max() + 1
    h, w = y2 - y1, x2 - x1
    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
    # Expand
    eh = int(h * expansion / 2)
    ew = int(w * expansion / 2)
    size = max(eh, ew)
    y1e = max(0, int(cy - size))
    y2e = min(rgb_pil.height, int(cy + size))
    x1e = max(0, int(cx - size))
    x2e = min(rgb_pil.width, int(cx + size))
    if y2e <= y1e or x2e <= x1e:
        return None
    crop = rgb_pil.crop((x1e, y1e, x2e, y2e))
    # Make square and resize with letterbox
    crop_sq = _letterbox_square(crop, output_size)
    return crop_sq


def extract_center_roi(rgb_pil, output_size=128):
    """Center ROI of same area as typical nucleus (~30% of image area)."""
    w, h = rgb_pil.size
    size = int(min(w, h) * 0.55)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    crop = rgb_pil.crop((x1, y1, x1 + size, y1 + size))
    return _letterbox_square(crop, output_size)


def extract_random_roi(rgb_pil, mask, output_size=128, rng=None):
    """Random ROI that has IoU < 0.1 with nucleus mask."""
    if rng is None:
        rng = np.random.RandomState()
    mask_arr = np.asarray(mask, dtype=bool)
    w, h = rgb_pil.size
    # Use nucleus size as reference
    area = mask_arr.sum()
    if area < 10:
        size = int(min(w, h) * 0.3)
    else:
        size = int(np.sqrt(area) * 1.8)
    size = min(size, min(w, h) - 1)
    size = max(16, size)

    for _ in range(50):
        x1 = rng.randint(0, w - size)
        y1 = rng.randint(0, h - size)
        test_mask = np.zeros((h, w), dtype=bool)
        test_mask[y1:y1+size, x1:x1+size] = True
        inter = (mask_arr & test_mask).sum()
        union = (mask_arr | test_mask).sum()
        iou = inter / max(union, 1)
        if iou < 0.1:
            crop = rgb_pil.crop((x1, y1, x1 + size, y1 + size))
            return _letterbox_square(crop, output_size)
    # Fallback: just return a random crop
    x1 = rng.randint(0, w - size)
    y1 = rng.randint(0, h - size)
    return _letterbox_square(rgb_pil.crop((x1, y1, x1 + size, y1 + size)), output_size)


def _letterbox_square(pil_img, size):
    """Resize to square while preserving aspect ratio, pad with border median."""
    w, h = pil_img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), _border_color(pil_img))
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _border_color(img):
    """Median color of image border pixels."""
    arr = np.asarray(img)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]])
    return tuple(int(np.median(border[:, c])) for c in range(3))


# ---------------------------------------------------------------------------
# Local encoder
# ---------------------------------------------------------------------------


class LocalEncoder(nn.Module):
    """Light CNN for 128×128 nucleus ROI views. No external pretrained weights needed."""

    def __init__(self, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 64
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 8
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),  # 4
            nn.AdaptiveAvgPool2d(1),  # → (B, 512, 1, 1)
        )
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        f = self.conv(x).flatten(1)
        return self.proj(f)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class M4NucViewModel(nn.Module):
    """M4-NucView: frozen M0 + local nucleus ROI encoder + zero-init residual."""

    def __init__(self, m0_backbone, m0_head, feature_dim=512, roi_dim=128, num_classes=5, variant="V1"):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        # Frozen M0
        self.backbone = m0_backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.m0_head = m0_head
        for p in self.m0_head.parameters():
            p.requires_grad = False

        # Local encoder (only for V1-V4)
        if variant != "V0":
            self.core_encoder = LocalEncoder(roi_dim)
            self.ctx_encoder = LocalEncoder(roi_dim)
            # Fusion: concat core + ctx → 128-d
            self.fusion = nn.Sequential(
                nn.Linear(roi_dim * 2, roi_dim),
                nn.ReLU(inplace=True),
            )
            # Gate
            self.gate = nn.Sequential(
                nn.Linear(roi_dim, 1),
                nn.Sigmoid(),
            )
            # Residual head (zero-init last layer)
            self.residual = nn.Sequential(
                nn.Linear(roi_dim, roi_dim),
                nn.ReLU(inplace=True),
                nn.Linear(roi_dim, num_classes),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        self.feature_dim = feature_dim
        self.roi_dim = roi_dim

    def forward(self, images, core_rois=None, ctx_rois=None):
        """images: full cell images. core_rois, ctx_rois: (B, 3, S, S) or None."""
        with torch.no_grad():
            features = self.backbone(images)
            if features.ndim > 2:
                features = features.flatten(1)
            z0 = self.m0_head(features)

        if self.variant == "V0" or core_rois is None:
            return z0, {"gate": 0.0, "delta_norm": 0.0}

        # Local encoding
        h_core = self.core_encoder(core_rois)  # (B, D)
        h_ctx = self.ctx_encoder(ctx_rois) if ctx_rois is not None else h_core
        h_local = self.fusion(torch.cat([h_core, h_ctx], dim=1))  # (B, D)
        gate = self.gate(h_local)  # (B, 1)
        delta = self.residual(h_local)  # (B, 5)
        alpha = 0.1  # initial scale to avoid overwhelming M0
        z = z0 + alpha * gate * delta

        return z, {"gate": gate.mean().item(), "delta_norm": delta.norm(dim=1).mean().item()}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self


def build_m4_nucview(m0_checkpoint_path, device, variant="V1"):
    """Build M4NucViewModel from M0 checkpoint."""
    from experiments.tbs.models import build_stage1_model

    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(m0_checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    model = M4NucViewModel(
        m0_backbone=m0.backbone,
        m0_head=m0.diagnosis_head,
        feature_dim=m0.feature_dim,
        roi_dim=128,
        variant=variant,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Variant={variant}  Trainable: {trainable:,}", flush=True)
    return model
