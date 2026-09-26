"""Training primitives for C0-R2F; architecture and diagnostics are unchanged."""

from __future__ import annotations

from experiments.tbs.c0r2_training import (
    build_model_optimizer_scheduler,
    evaluate_c0r2 as evaluate_c0r2f,
    make_loader,
    seed_everything,
)
from experiments.tbs.c0r2f_protocol import LOCKED_C0R2F_CONFIG


def train_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.tbs.c0r2f_loss import c0r2f_loss

    model.train()
    keys = (
        "loss", "diagnosis_loss", "full_anchor_loss", "low_grade_pair_loss",
        "high_grade_mass_loss", "geometry_loss", "residual_loss",
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
            losses = c0r2f_loss(
                output,
                labels,
                lambda_geometry=LOCKED_C0R2F_CONFIG["lambda_geometry"],
                lambda_residual=LOCKED_C0R2F_CONFIG["lambda_residual"],
                lambda_full_anchor=LOCKED_C0R2F_CONFIG["lambda_full_anchor"],
                lambda_low_grade_pair=LOCKED_C0R2F_CONFIG["lambda_low_grade_pair"],
                lambda_high_grade_mass=LOCKED_C0R2F_CONFIG["lambda_high_grade_mass"],
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
        raise ValueError("C0-R2F training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


__all__ = (
    "build_model_optimizer_scheduler", "evaluate_c0r2f", "make_loader",
    "seed_everything", "train_epoch",
)
