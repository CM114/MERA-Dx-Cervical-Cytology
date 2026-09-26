"""Training primitives for the independent full-view C2 dual prototypes."""

from __future__ import annotations

import numpy as np

from experiments.tbs.c0r2_training import make_loader, seed_everything
from experiments.tbs.c2_dualprototype_protocol import LOCKED_C2_DUALPROTOTYPE_CONFIG


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.tbs_fv_factorized_model import build_tbs_dualprototype_model

    epochs = LOCKED_C2_DUALPROTOTYPE_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C2_DUALPROTOTYPE_CONFIG["epochs"]:
        raise ValueError("C2 epochs are locked to 30")
    model = build_tbs_dualprototype_model(
        model_name=LOCKED_C2_DUALPROTOTYPE_CONFIG["model_name"],
        pretrained=LOCKED_C2_DUALPROTOTYPE_CONFIG["pretrained"],
        semantic_dim=LOCKED_C2_DUALPROTOTYPE_CONFIG["semantic_dim"],
        residual_logit_bound=LOCKED_C2_DUALPROTOTYPE_CONFIG["residual_logit_bound"],
        prototype_temperature=LOCKED_C2_DUALPROTOTYPE_CONFIG["prototype_temperature"],
        prototype_residual_bound=LOCKED_C2_DUALPROTOTYPE_CONFIG["prototype_residual_bound"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C2_DUALPROTOTYPE_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C2_DUALPROTOTYPE_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C2_DUALPROTOTYPE_CONFIG["lr"],
        weight_decay=LOCKED_C2_DUALPROTOTYPE_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def _structural_factor_labels(labels):
    morph = ((labels == 3) | (labels == 4)).long()
    evidence = ((labels == 2) | (labels == 4)).long()
    return morph, evidence, labels > 0


def initialize_prototypes(model, loader, device):
    """Initialize both prototype spaces from the current training fold only."""
    import torch

    model.eval()
    morph_features, morph_labels = [], []
    evidence_features, evidence_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["diagnosis_label"].to(device, non_blocking=True).long()
            output = model(images)
            morph, evidence, semantic_mask = _structural_factor_labels(labels)
            if bool(semantic_mask.any()):
                morph_features.append(output["morph_features"][semantic_mask].detach())
                evidence_features.append(output["evidence_features"][semantic_mask].detach())
                morph_labels.append(morph[semantic_mask].detach())
                evidence_labels.append(evidence[semantic_mask].detach())
    if not morph_features:
        raise ValueError("C2 prototype initialization found no abnormal training samples")
    model.initialize_prototypes_from_train_features(
        torch.cat(morph_features, dim=0),
        torch.cat(morph_labels, dim=0),
        torch.cat(evidence_features, dim=0),
        torch.cat(evidence_labels, dim=0),
    )


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
                lambda_screen=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_screen"],
                lambda_morph=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_morph"],
                lambda_evidence=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_evidence"],
                lambda_decorr=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_decorr"],
                lambda_prototype=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_prototype"],
                lambda_base_anchor=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_base_anchor"],
                lambda_residual=LOCKED_C2_DUALPROTOTYPE_CONFIG["lambda_residual"],
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
        raise ValueError("C2 training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


def _prototype_margins(output, labels):
    import torch

    morph_targets, evidence_targets, semantic_mask = _structural_factor_labels(labels)
    if not bool(semantic_mask.any()):
        return torch.zeros(0, device=labels.device)
    morph_logits = output["morph_prototype_logits"][semantic_mask]
    evidence_logits = output["evidence_prototype_logits"][semantic_mask]
    morph_targets = morph_targets[semantic_mask]
    evidence_targets = evidence_targets[semantic_mask]
    morph_true = morph_logits.gather(1, morph_targets[:, None]).squeeze(1)
    evidence_true = evidence_logits.gather(1, evidence_targets[:, None]).squeeze(1)
    morph_other = morph_logits.gather(1, (1 - morph_targets)[:, None]).squeeze(1)
    evidence_other = evidence_logits.gather(1, (1 - evidence_targets)[:, None]).squeeze(1)
    return (morph_true - morph_other + evidence_true - evidence_other) / 2.0


def evaluate_c2(model, loader, device):
    import torch

    from experiments.tbs.tbs_branch_metrics import compute_tbs_branch_metrics
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    labels, fused, base = [], [], []
    morph_probs, evidence_probs, residual_logits, prototype_margins = [], [], [], []
    prototype_context_means = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch["diagnosis_label"].to(device, non_blocking=True).long()
            output = model(images)
            labels.append(batch_labels.detach().cpu().numpy())
            fused.append(output["diagnosis_probs"].detach().float().cpu().numpy())
            base.append(output["base_diagnosis_probs"].detach().float().cpu().numpy())
            morph_probs.append(output["morph_probs"].detach().float().cpu().numpy())
            evidence_probs.append(output["evidence_probs"].detach().float().cpu().numpy())
            residual_logits.append(output["residual_logits"].detach().float().cpu().numpy())
            if "prototype_context_abs_mean" in output:
                prototype_context_means.append(
                    float(output["prototype_context_abs_mean"].detach().float().cpu())
                )
            margins = _prototype_margins(output, batch_labels)
            if margins.numel():
                prototype_margins.append(margins.detach().float().cpu().numpy())
    if not labels:
        raise ValueError("C2 evaluation loader is empty")
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_prob = np.concatenate(fused)
    base_prob = np.concatenate(base)
    audits = {
        "morph_probs": np.concatenate(morph_probs),
        "evidence_probs": np.concatenate(evidence_probs),
        "residual_logits": np.concatenate(residual_logits),
        "prototype_margin": np.concatenate(prototype_margins)
        if prototype_margins else np.zeros(1, dtype=float),
    }
    metrics = compute_tbs_branch_metrics(y_true, y_prob, audits)
    if prototype_context_means:
        metrics["prototype_context_abs_mean"] = float(np.mean(prototype_context_means))
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
    "evaluate_c2",
    "initialize_prototypes",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
