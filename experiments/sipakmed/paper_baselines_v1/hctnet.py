"""Paper-guided HCT-Net implementation for SIPaKMeD.

The public HCT-Net repository exposes the module names but its encoder and
attention tensor plumbing are incomplete.  This file keeps the recoverable
paper mechanisms while making every stage shape-explicit and testable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


HCT_CONFIG = {
    "input_size": 224,
    "embed_dims": (64, 128, 320, 512),
    "depths": (3, 4, 18, 3),
    "num_heads": (1, 2, 5, 8),
    "sr_ratios": (8, 4, 2, 1),
    "window_size": 7,
    "hmsa_stages": (0, 1, 2),
    "mff_channels": 128,
    "description": "PVTv2-like pyramid + HMSA + MFF + MCP",
}


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Keep MCP BatchNorm semantics while accepting a singleton tail batch."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.ndim == 2 and input.shape[0] == 1:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(input)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PVTBlock(nn.Module):
    """Spatial-reduction attention block used by the four-stage pyramid."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.dim = dim
        self.sr_ratio = int(sr_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim) if sr_ratio > 1 else nn.Identity()
        self.kv_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = x.flatten(2).transpose(1, 2)
        reduced = self.sr(x)
        key_value = reduced.flatten(2).transpose(1, 2)
        query_norm = self.norm1(query)
        key_value_norm = self.kv_norm(key_value)
        attended, _ = self.attn(query_norm, key_value_norm, key_value_norm, need_weights=False)
        query = query + attended
        query = query + self.mlp(self.norm2(query))
        return query.transpose(1, 2).reshape(batch, channels, height, width)


class PyramidStage(nn.Module):
    def __init__(self, in_channels: int, dim: int, depth: int, num_heads: int, sr_ratio: int, first: bool):
        super().__init__()
        kernel, stride, padding = (7, 4, 3) if first else (3, 2, 1)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.LayerNorm(dim),
        )
        self.blocks = nn.ModuleList(PVTBlock(dim, num_heads, sr_ratio) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed[0](x)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_embed[1](x)
        batch, tokens, channels = x.shape
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError(f"pyramid stage produced non-square token count: {tokens}")
        x = x.transpose(1, 2).reshape(batch, channels, side, side)
        for block in self.blocks:
            x = block(x)
        return x


class PVTv2Encoder(nn.Module):
    def __init__(self, embed_dims=HCT_CONFIG["embed_dims"], depths=HCT_CONFIG["depths"]):
        super().__init__()
        dims = tuple(int(value) for value in embed_dims)
        depths = tuple(int(value) for value in depths)
        heads = HCT_CONFIG["num_heads"]
        sr_ratios = HCT_CONFIG["sr_ratios"]
        stages = []
        in_channels = 3
        for index, (dim, depth) in enumerate(zip(dims, depths)):
            stages.append(PyramidStage(in_channels, dim, depth, heads[index], sr_ratios[index], first=index == 0))
            in_channels = dim
        self.stages = nn.ModuleList(stages)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = images
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def _window_partition(x: torch.Tensor, window_size: int, shift: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    batch, channels, height, width = x.shape
    if shift:
        x = torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))
    padded_height = (height + window_size - 1) // window_size * window_size
    padded_width = (width + window_size - 1) // window_size * window_size
    if padded_height != height or padded_width != width:
        x = F.pad(x, (0, padded_width - width, 0, padded_height - height))
    windows = x.view(batch, channels, padded_height // window_size, window_size, padded_width // window_size, window_size)
    windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, channels)
    return windows, (height, width, padded_height, padded_width)


def _window_reverse(windows: torch.Tensor, meta: tuple[int, int, int, int], window_size: int, shift: int) -> torch.Tensor:
    height, width, padded_height, padded_width = meta
    windows_per_image = (padded_height // window_size) * (padded_width // window_size)
    batch = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]
    x = windows.view(batch, padded_height // window_size, padded_width // window_size, window_size, window_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_height, padded_width)
    x = x[..., :height, :width]
    if shift:
        x = torch.roll(x, shifts=(shift, shift), dims=(-2, -1))
    return x


class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift: int | None = None):
        super().__init__()
        if dim % num_heads:
            raise ValueError("HMSA dimension must be divisible by the number of heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift = int(shift if shift is not None else window_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, meta = _window_partition(x, self.window_size, self.shift)
        normalized = self.norm(windows)
        batch_windows, tokens, _ = normalized.shape
        qkv = self.qkv(normalized).reshape(batch_windows, tokens, 3, self.num_heads, self.dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, self.dim)
        output = self.proj(output)
        return _window_reverse(output, meta, self.window_size, self.shift)


class HMSA(nn.Module):
    """Hybrid multi-dimensional self-attention from the HCT-Net design."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        hidden = max(dim // 4, 4)
        self.window_attention = ShiftedWindowAttention(dim, num_heads, window_size)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.branch_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.projection = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window_branch = self.window_attention(x)
        spatial_branch = x * self.spatial_attention(x)
        channel_branch = x * self.channel_attention(x)
        weights = self.branch_weights(x).softmax(dim=-1).view(x.shape[0], 3, 1, 1, 1)
        fused = weights[:, 0] * window_branch + weights[:, 1] * spatial_branch + weights[:, 2] * channel_branch
        return x + self.projection(fused)


class MFF(nn.Module):
    def __init__(self, in_channels=HCT_CONFIG["embed_dims"], output_channels: int = HCT_CONFIG["mff_channels"]):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, output_channels, kernel_size=1, bias=False), nn.BatchNorm2d(output_channels), nn.GELU())
            for channels in in_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(output_channels * len(in_channels), output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        aligned = []
        for feature, projection in zip(features, self.projections):
            mapped = projection(feature)
            if mapped.shape[-2:] != target_size:
                mapped = F.interpolate(mapped, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(mapped)
        return self.fusion(torch.cat(aligned, dim=1))


class HCTNet(nn.Module):
    def __init__(self, num_classes: int = 5, embed_dims=HCT_CONFIG["embed_dims"], depths=HCT_CONFIG["depths"]):
        super().__init__()
        self.embed_dims = tuple(embed_dims)
        self.depths = tuple(depths)
        self.encoder = PVTv2Encoder(embed_dims=self.embed_dims, depths=self.depths)
        heads = HCT_CONFIG["num_heads"]
        self.hmsa = nn.ModuleList(
            HMSA(self.embed_dims[index], heads[index], HCT_CONFIG["window_size"])
            for index in HCT_CONFIG["hmsa_stages"]
        )
        self.mff = MFF(self.embed_dims, HCT_CONFIG["mff_channels"])
        self.mcp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            SafeBatchNorm1d(HCT_CONFIG["mff_channels"]),
            nn.Linear(HCT_CONFIG["mff_channels"], num_classes),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        for index, module in enumerate(self.hmsa):
            features[index] = module(features[index])
        return self.mff(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.mcp(self.forward_features(images))


class JointLoss(nn.Module):
    """Weighted cross-entropy plus focal loss used by the public training entry."""

    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 2.0, alpha: float = 0.25, smoothing: float = 0.1):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, label_smoothing=self.smoothing)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        target_log_probability = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        focal = (-self.alpha * (1.0 - target_probability).pow(self.gamma) * target_log_probability).mean()
        return ce + focal
