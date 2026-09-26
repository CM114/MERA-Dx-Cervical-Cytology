"""Training primitives for the independent full-view C1 factorization."""

from __future__ import annotations

import numpy as np

from experiments.tbs.c1_factorized_protocol import LOCKED_C1_FACTORIZED_CONFIG
from experiments.tbs.c0r2_training import make_loader, seed_everything


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.tbs_fv_factorized_model import build_tbs_fv_factorized_model

    epochs = LOCKED_C1_FACTORIZED_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C1_FACTORIZED_CONFIG["epochs"]:
        raise ValueError("C1 epochs are locked to 30")
    model = build_tbs_fv_factorized_model(
        model_name=LOCKED_C1_FACTORIZED_CONFIG["model_name"],
        pretrained=LOCKED_C1_FACTORIZED_CONFIG["pretrained"],
        semantic_dim=LOCKED_C1_FACTORIZED_CONFIG["semantic_dim"],
        residual_logit_bound=LOCKED_C1_FACTORIZED_CONFIG["residual_logit_bound"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C1_FACTORIZED_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C1_FACTORIZED_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C1_FACTORIZED_CONFIG["lr"],
        weight_decay=LOCKED_C1_FACTORIZED_CONFIG["weight_decay"],
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
                lambda_screen=LOCKED_C1_FACTORIZED_CONFIG["lambda_screen"],
                lambda_morph=LOCKED_C1_FACTORIZED_CONFIG["lambda_morph"],
                lambda_evidence=LOCKED_C1_FACTORIZED_CONFIG["lambda_evidence"],
                lambda_decorr=LOCKED_C1_FACTORIZED_CONFIG["lambda_decorr"],
                lambda_base_anchor=LOCKED_C1_FACTORIZED_CONFIG["lambda_base_anchor"],
                lambda_residual=LOCKED_C1_FACTORIZED_CONFIG["lambda_residual"],
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
        raise ValueError("C1 training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


def evaluate_c1(model, loader, device):
    import torch

    from experiments.tbs.tbs_branch_metrics import compute_tbs_branch_metrics
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    labels, fused, base = [], [], []
    morph_probs, evidence_probs, residual_logits = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            labels.append(batch["diagnosis_label"].detach().cpu().numpy())
            fused.append(output["diagnosis_probs"].detach().float().cpu().numpy())
            base.append(output["base_diagnosis_probs"].detach().float().cpu().numpy())
            morph_probs.append(output["morph_probs"].detach().float().cpu().numpy())
            evidence_probs.append(output["evidence_probs"].detach().float().cpu().numpy())
            residual_logits.append(output["residual_logits"].detach().float().cpu().numpy())
    if not labels:
        raise ValueError("C1 evaluation loader is empty")
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_prob = np.concatenate(fused)
    base_prob = np.concatenate(base)
    audits = {
        "morph_probs": np.concatenate(morph_probs),
        "evidence_probs": np.concatenate(evidence_probs),
        "residual_logits": np.concatenate(residual_logits),
    }
    metrics = compute_tbs_branch_metrics(y_true, y_prob, audits)
    metrics["base_macro_f1"] = float(compute_locked_candidate_metrics(y_true, base_prob)["macro_f1"])
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "y_true": y_true,
        "y_prob": y_prob,
        "full_y_prob": base_prob,
        "audits": audits,
    }


__all__ = (
    "build_model_optimizer_scheduler",
    "evaluate_c1",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
