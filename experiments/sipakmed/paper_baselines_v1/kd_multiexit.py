"""Paper-guided multi-exit self-distillation model for SIPaKMeD."""
from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18, resnet50

class ExitHead(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, classes)
    def forward(self, x):
        return self.fc(self.norm(torch.flatten(self.pool(x), 1)))

class KDMultiExit(nn.Module):
    """Shared ResNet backbone with two early exits and a final self-teacher."""
    def __init__(self, num_classes=5, width_multiplier=1.0):
        super().__init__()
        smoke = width_multiplier < 0.5
        net = resnet18(weights=None) if smoke else resnet50(weights=None)
        channels = (128, 256, 512) if smoke else (512, 1024, 2048)
        self.backbone_name = "resnet18_smoke" if smoke else "resnet50"
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList((net.layer1, net.layer2, net.layer3, net.layer4))
        self.exits = nn.ModuleList(ExitHead(c, num_classes) for c in channels)
    def forward(self, x):
        x = self.stem(x)
        logits = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx >= 1:
                logits.append(self.exits[idx - 1](x))
        return {"logits": logits}

def kd_loss(outputs, labels, temperature=4.0, exit_weight=0.5, distill_weight=0.7):
    logits = outputs["logits"]
    if len(logits) != 3:
        raise ValueError("expected three exits")
    hard_losses = [nn.functional.cross_entropy(item, labels) for item in logits]
    hard = hard_losses[-1] + exit_weight * sum(hard_losses[:-1]) / 2.0
    teacher_prob = nn.functional.softmax(logits[-1].detach() / temperature, dim=1)
    terms = [
        nn.functional.kl_div(nn.functional.log_softmax(item / temperature, dim=1), teacher_prob, reduction="batchmean")
        * (temperature * temperature)
        for item in logits[:-1]
    ]
    distill = sum(terms) / len(terms)
    return {
        "loss": hard + distill_weight * distill,
        "hard_loss": hard,
        "distill_loss": distill,
        "final_loss": hard_losses[-1],
        "early_loss": sum(hard_losses[:-1]) / 2.0,
    }
