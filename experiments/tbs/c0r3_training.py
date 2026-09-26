"""Training primitives for C0-R3."""

from __future__ import annotations

from experiments.tbs.c0r2f_training import (
    evaluate_c0r2f,
    make_loader,
    seed_everything,
    train_epoch,
)
from experiments.tbs.c0r3_protocol import LOCKED_C0R3_CONFIG


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.c0r3_model import build_tbs_c0r3_model
    from experiments.tbs.optimization import build_discriminative_parameter_groups

    epochs = LOCKED_C0R3_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C0R3_CONFIG["epochs"]:
        raise ValueError("C0-R3 epochs are locked to 30")
    model = build_tbs_c0r3_model(
        model_name=LOCKED_C0R3_CONFIG["model_name"],
        pretrained=LOCKED_C0R3_CONFIG["pretrained"],
        local_translation_bound=LOCKED_C0R3_CONFIG["local_translation_bound"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C0R3_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C0R3_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C0R3_CONFIG["lr"],
        weight_decay=LOCKED_C0R3_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


__all__ = (
    "build_model_optimizer_scheduler", "evaluate_c0r2f", "make_loader",
    "seed_everything", "train_epoch",
)
