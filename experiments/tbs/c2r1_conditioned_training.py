"""Training primitives for the independent C2-R1 conditioned residual."""

from __future__ import annotations

from experiments.tbs.c0r2_training import make_loader, seed_everything
from experiments.tbs.c2_dualprototype_training import (
    evaluate_c2,
    initialize_prototypes,
)
from experiments.tbs.c2r1_conditioned_protocol import LOCKED_C2R1_CONFIG


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.tbs_fv_c2r1_model import build_tbs_c2r1_conditioned_model

    epochs = LOCKED_C2R1_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C2R1_CONFIG["epochs"]:
        raise ValueError("C2-R1 epochs are locked to 30")
    model = build_tbs_c2r1_conditioned_model(
        model_name=LOCKED_C2R1_CONFIG["model_name"],
        pretrained=LOCKED_C2R1_CONFIG["pretrained"],
        semantic_dim=LOCKED_C2R1_CONFIG["semantic_dim"],
        residual_logit_bound=LOCKED_C2R1_CONFIG["residual_logit_bound"],
        prototype_temperature=LOCKED_C2R1_CONFIG["prototype_temperature"],
        prototype_residual_bound=LOCKED_C2R1_CONFIG["prototype_residual_bound"],
        prototype_context_scale=LOCKED_C2R1_CONFIG["prototype_context_scale"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C2R1_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C2R1_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C2R1_CONFIG["lr"],
        weight_decay=LOCKED_C2R1_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def train_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.tbs.tbs_fv_factorized_loss import factorized_tbs_loss

    model.train()
    keys = (
        "loss", "diagnosis_loss", "base_anchor_loss", "screen_loss",
        "morph_loss", "evidence_loss", "decorr_loss", "prototype_loss",
        "residual_loss",
    )
    totals = {key: 0.0 for key in keys}
    sample_count = 0
    amp_enabled = bool(scaler.is_enabled())
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["diagnosis_label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            losses = factorized_tbs_loss(
                output,
                labels,
                lambda_screen=LOCKED_C2R1_CONFIG["lambda_screen"],
                lambda_morph=LOCKED_C2R1_CONFIG["lambda_morph"],
                lambda_evidence=LOCKED_C2R1_CONFIG["lambda_evidence"],
                lambda_decorr=LOCKED_C2R1_CONFIG["lambda_decorr"],
                lambda_prototype=LOCKED_C2R1_CONFIG["lambda_prototype"],
                lambda_base_anchor=LOCKED_C2R1_CONFIG["lambda_base_anchor"],
                lambda_residual=LOCKED_C2R1_CONFIG["lambda_residual"],
            )
        if amp_enabled:
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()
        count = int(labels.shape[0])
        sample_count += count
        for key in keys:
            totals[key] += float(losses[key].detach().item()) * count
    if sample_count <= 0:
        raise ValueError("C2-R1 training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


__all__ = (
    "build_model_optimizer_scheduler",
    "evaluate_c2",
    "initialize_prototypes",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
