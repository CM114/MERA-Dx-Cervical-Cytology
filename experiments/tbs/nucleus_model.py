"""Dual-view classification model: frozen CaFormer-S18 + nucleus MLP encoder."""

import torch
import torch.nn as nn


class NucleusEncoder(nn.Module):
    """Small MLP that maps 18 nucleus scalar features to a 128-d embedding.

    Architecture: Linear(18→64) → ReLU → Dropout(0.2) → Linear(64→128)
    """

    def __init__(self, input_dim=18, hidden_dim=64, output_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """x: (B, input_dim) float32."""
        return self.mlp(x)


class DualViewModel(nn.Module):
    """Frozen CaFormer-S18 backbone + nucleus encoder → linear classifier.

    Parameters
    ----------
    backbone : nn.Module
        Pre-trained CaFormer-S18 in eval mode (not updated during training).
    cell_dim : int
        Dimensionality of pooled backbone features (512 for CaFormer-S18).
    nucleus_dim : int
        Number of nucleus scalar features (default 18).
    num_classes : int
        Number of output classes (default 5).
    nucleus_hidden : int
        Hidden dimension of nucleus MLP (default 64).
    nucleus_emb_dim : int
        Output dimension of nucleus MLP (default 128).
    dropout : float
        Dropout rate in nucleus MLP.
    """

    def __init__(
        self,
        backbone,
        cell_dim=512,
        nucleus_dim=18,
        num_classes=5,
        nucleus_hidden=64,
        nucleus_emb_dim=128,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.nucleus_encoder = NucleusEncoder(
            input_dim=nucleus_dim,
            hidden_dim=nucleus_hidden,
            output_dim=nucleus_emb_dim,
            dropout=dropout,
        )

        fusion_dim = cell_dim + nucleus_emb_dim
        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.cell_dim = cell_dim
        self.nucleus_dim = nucleus_dim
        self.nucleus_emb_dim = nucleus_emb_dim
        self.num_classes = num_classes

    def forward(self, images, nucleus_features):
        """Forward pass.

        Parameters
        ----------
        images : torch.Tensor, shape (B, 3, H, W)
            Batch of letterbox-224 RGB images.
        nucleus_features : torch.Tensor, shape (B, nucleus_dim)
            Standardised nucleus scalar features.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
        cell_feat : torch.Tensor, shape (B, cell_dim)
            L2-normalised backbone features.
        nucleus_emb : torch.Tensor, shape (B, nucleus_emb_dim)
            Nucleus embedding.
        """
        import torch.nn.functional as F

        # Frozen backbone forward
        with torch.no_grad():
            cell_feat = self.backbone(images)
        # L2-normalise (matches M0 convention)
        cell_feat = F.normalize(cell_feat, p=2, dim=1)

        # Nucleus encoding
        nucleus_emb = self.nucleus_encoder(nucleus_features)

        # Fusion
        fused = torch.cat([cell_feat, nucleus_emb], dim=1)
        logits = self.classifier(fused)

        return logits, cell_feat, nucleus_emb

    def train(self, mode=True):
        """Override train() to keep backbone in eval mode always."""
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        return self


def build_dualview_model(checkpoint_path, device, nucleus_dim=18, num_classes=5):
    """Build DualViewModel from a frozen M0 checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to M0 ``best_model.pth``.
    device : torch.device
    nucleus_dim : int
    num_classes : int

    Returns
    -------
    DualViewModel
    """
    from experiments.tbs.models import build_stage1_model

    # Build a fresh M0 model and load checkpoint weights
    m0 = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    m0.load_state_dict(state, strict=True)
    m0.to(device)

    # Extract backbone (timm model that returns pooled 512-d features)
    backbone = m0.backbone
    # Keep backbone weights but freeze during DualViewModel training
    for param in backbone.parameters():
        param.requires_grad = False

    model = DualViewModel(
        backbone=backbone,
        cell_dim=512,
        nucleus_dim=nucleus_dim,
        num_classes=num_classes,
    )
    return model.to(device)
