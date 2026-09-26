"""Paper-guided baseline adapters used by the Xudata runner.

These adapters preserve the main mechanism named by each paper while using the
project's timm/PyTorch runtime. They are tagged as adaptations in every output;
they are not claims that incomplete author repositories are exact replicas.
"""

from __future__ import annotations

from typing import Iterable


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def _backbone(name: str, pretrained: bool, pooled: bool = True):
    import timm

    return timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg" if pooled else "",
    )


class TimmClassifier:
    """Lazy factory wrapper so importing the adapter does not require torch."""

    @staticmethod
    def build(name: str, num_classes: int = 5, pretrained: bool = True):
        torch, nn = _torch()
        backbone = _backbone(name, pretrained, pooled=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(backbone.num_features, num_classes)

            def forward(self, x):
                return self.head(self.backbone(x))

        return Model()


def _vector(backbone, x):
    y = backbone(x)
    if y.ndim == 4:
        y = y.mean(dim=(2, 3))
    return y.flatten(1)


def _feature_vector(backbone, x):
    """Return the pooled feature map whose width matches ``num_features``."""

    y = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
    if y.ndim == 4:
        y = y.mean(dim=(2, 3))
    return y.flatten(1)


def _fusion(names: Iterable[str], num_classes: int, pretrained: bool):
    torch, nn = _torch()
    branches = nn.ModuleList([_backbone(name, pretrained, pooled=True) for name in names])

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = branches
            self.head = nn.Sequential(
                nn.LazyLinear(1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes),
            )

        def forward(self, x):
            return self.head(torch.cat([_vector(branch, x) for branch in self.branches], dim=1))

    return Fusion()


def build_a2sdnet121(num_classes: int = 5, pretrained: bool = True):
    """A2SDNet121 adaptation: DenseNet121 map, SE and dilated multi-scale block."""

    torch, nn = _torch()
    backbone = _backbone("densenet121", pretrained, pooled=False)
    channels = int(backbone.num_features)

    class A2SD(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = max(128, channels // 4)
            self.backbone = backbone
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.atrous = nn.ModuleList(
                [nn.Conv2d(channels, hidden, 3, padding=dilation, dilation=dilation) for dilation in (1, 2, 3)]
            )
            self.head = nn.Linear(hidden * 3, num_classes)

        def forward(self, x):
            fmap = self.backbone(x)
            fmap = fmap * self.se(fmap)
            features = [layer(fmap).mean(dim=(2, 3)) for layer in self.atrous]
            return self.head(torch.cat(features, dim=1))

    return A2SD()


def build_deepcervix_hdff(num_classes: int = 5, pretrained: bool = True):
    return _fusion(("vgg16", "vgg19", "xception41", "resnet50"), num_classes, pretrained)


def build_msenet(num_classes: int = 5, pretrained: bool = True):
    """MSENet adaptation with mean/std calibrated three-member decisions."""

    torch, nn = _torch()
    members = nn.ModuleList([_backbone(name, pretrained, pooled=True) for name in ("inception_v3", "xception41", "vgg16")])
    heads = nn.ModuleList([nn.Linear(int(member.num_features), num_classes) for member in members])

    class MSENet(nn.Module):
        def __init__(self):
            super().__init__()
            self.members = members
            self.heads = heads

        def forward(self, x):
            logits = torch.stack([head(_feature_vector(member, x)) for member, head in zip(self.members, self.heads)], dim=0)
            probs = logits.softmax(dim=-1)
            mean = probs.mean(dim=0)
            std = probs.std(dim=0, unbiased=False)
            return (mean - std).clamp_min(1e-6).log()

    return MSENet()


def build_hctnet(num_classes: int = 5, pretrained: bool = True):
    """HCT-Net adaptation: ConvNeXt local branch plus Swin global branch."""

    return _fusion(("convnext_tiny", "swin_tiny_patch4_window7_224"), num_classes, pretrained)


def build_lgpnet(num_classes: int = 5, pretrained: bool = True):
    return _fusion(("resnet50", "vit_base_patch16_224"), num_classes, pretrained)


def build_msccnet(num_classes: int = 5, pretrained: bool = True):
    torch, nn = _torch()
    backbone = _backbone("resnet18", pretrained, pooled=False)
    channels = int(backbone.num_features)

    class MSCC(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.multi_scale = nn.ModuleList([nn.Conv2d(channels, 128, 3, padding=p, dilation=p) for p in (1, 2, 3)])
            self.head = nn.Linear(128 * 3, num_classes)

        def forward(self, x):
            fmap = self.backbone(x)
            return self.head(torch.cat([layer(fmap).mean(dim=(2, 3)) for layer in self.multi_scale], dim=1))

    return MSCC()


def build_cercan(num_classes: int = 5, pretrained: bool = True):
    return _fusion(("mobilenetv3_large_100", "efficientnet_b0", "resnet18"), num_classes, pretrained)


def build_diff(num_classes: int = 5, pretrained: bool = True):
    return _fusion(("convnext_tiny", "vit_base_patch16_224"), num_classes, pretrained)


def build_mtfm(num_classes: int = 5, pretrained: bool = True):
    """MTFM adaptation with a five-class head and an auxiliary screen branch."""

    torch, nn = _torch()
    backbone = _backbone("resnet50", pretrained, pooled=True)

    class MTFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.five = nn.Linear(backbone.num_features, num_classes)
            self.screen = nn.Linear(backbone.num_features, 2)

        def forward(self, x):
            feat = _vector(self.backbone, x)
            # The runner optimizes the five-class output; screen is retained for
            # inspection and can be enabled by a future multi-task loss.
            return self.five(feat)

    return MTFM()


def build_knowledge_distill(num_classes: int = 5, pretrained: bool = True):
    """Compact multi-exit proxy: student logits supervised by a frozen teacher in the runner."""

    return TimmClassifier.build("resnet18", num_classes, pretrained)


METHOD_BUILDERS = {
    "HCT-Net": build_hctnet,
    "LGPNet": build_lgpnet,
    "A2SDNet121": build_a2sdnet121,
    "DeepCervix-HDFF": build_deepcervix_hdff,
    "MSENet": build_msenet,
    "MSCCNet": build_msccnet,
    "CerCan-Net": build_cercan,
    "DIFF": build_diff,
    "MTFM": build_mtfm,
    "KnowledgeDistill": build_knowledge_distill,
}


def build_method(name: str, num_classes: int = 5, pretrained: bool = True):
    if name not in METHOD_BUILDERS:
        raise KeyError(f"unknown differentiable method: {name}")
    return METHOD_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)
