"""Training primitives for locked S1-R3 high-grade risk CV."""

from __future__ import annotations

import numpy as np

from experiments.tbs.s1r3_protocol import LOCKED_S1R3_CONFIG
from experiments.tbs.s1r_training import (
    LOSS_NAMES as BASE_LOSS_NAMES,
    make_loader,
    seed_everything,
)


LOSS_NAMES = (*BASE_LOSS_NAMES, "high_grade_risk_loss")
COMPONENT_LOSS_NAMES = tuple(name for name in BASE_LOSS_NAMES if name != "loss")


def finalize_loss_metrics(
    component_totals,
    base_loss_total,
    risk_loss_total,
    high_grade_count,
    sample_count,
):
    sample_count = int(sample_count)
    high_grade_count = int(high_grade_count)
    metrics = {
        name: float(component_totals[name]) / max(sample_count, 1)
        for name in COMPONENT_LOSS_NAMES
    }
    risk_mean = float(risk_loss_total) / max(high_grade_count, 1)
    metrics["high_grade_risk_loss"] = risk_mean
    metrics["loss"] = (
        float(base_loss_total) / max(sample_count, 1)
        + LOCKED_S1R3_CONFIG["lambda_high_grade_risk"] * risk_mean
    )
    return metrics


def build_model_optimizer_scheduler(device):
    import torch

    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.s1r2_model import build_tbs_s1r2_model

    model = build_tbs_s1r2_model(
        model_name=LOCKED_S1R3_CONFIG["model_name"],
        pretrained=LOCKED_S1R3_CONFIG["pretrained"],
        semantic_dim=LOCKED_S1R3_CONFIG["semantic_dim"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_S1R3_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_S1R3_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_S1R3_CONFIG["lr"],
        weight_decay=LOCKED_S1R3_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LOCKED_S1R3_CONFIG["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def _targets(batch, device):
    return {
        "diagnosis_labels": batch["diagnosis_label"].to(device, non_blocking=True),
        "screen_labels": batch["screen_label"].to(device, non_blocking=True),
        "morph_labels": batch["morph_label"].to(device, non_blocking=True),
        "evidence_labels": batch["evidence_label"].to(device, non_blocking=True),
        "semantic_mask": batch["semantic_mask"].to(
            device, non_blocking=True
        ).bool(),
    }


def train_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.tbs.s1r3_loss import singleview_s1r3_loss

    model.train()
    component_totals = {name: 0.0 for name in COMPONENT_LOSS_NAMES}
    base_loss_total = 0.0
    risk_loss_total = 0.0
    high_grade_count = 0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            losses = singleview_s1r3_loss(model(images), targets)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        size = len(images)
        count += size
        for name in COMPONENT_LOSS_NAMES:
            component_totals[name] += float(losses[name].detach()) * size
        risk = float(losses["high_grade_risk_loss"].detach())
        batch_high_grade_count = int(
            ((targets["diagnosis_labels"] == 3) | (targets["diagnosis_labels"] == 4))
            .sum()
            .item()
        )
        high_grade_count += batch_high_grade_count
        risk_loss_total += risk * batch_high_grade_count
        base_loss = (
            losses["loss"]
            - LOCKED_S1R3_CONFIG["lambda_high_grade_risk"]
            * losses["high_grade_risk_loss"]
        )
        base_loss_total += float(base_loss.detach()) * size
    return finalize_loss_metrics(
        component_totals,
        base_loss_total,
        risk_loss_total,
        high_grade_count,
        count,
    )


def evaluate_s1r3(model, loader, device):
    import torch

    from experiments.tbs.metrics import compute_semantic_metrics
    from experiments.tbs.s1r3_loss import singleview_s1r3_loss
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    component_totals = {name: 0.0 for name in COMPONENT_LOSS_NAMES}
    base_loss_total = 0.0
    risk_loss_total = 0.0
    high_grade_count = 0
    residual_abs_total = 0.0
    count = 0
    labels, probabilities, paths = [], [], []
    masks, morph_true, morph_prob = [], [], []
    evidence_true, evidence_prob = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = _targets(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
                losses = singleview_s1r3_loss(output, targets)
            size = len(images)
            count += size
            for name in COMPONENT_LOSS_NAMES:
                component_totals[name] += float(losses[name].detach()) * size
            risk = float(losses["high_grade_risk_loss"].detach())
            batch_high_grade_count = int(
                (
                    (targets["diagnosis_labels"] == 3)
                    | (targets["diagnosis_labels"] == 4)
                )
                .sum()
                .item()
            )
            high_grade_count += batch_high_grade_count
            risk_loss_total += risk * batch_high_grade_count
            base_loss = (
                losses["loss"]
                - LOCKED_S1R3_CONFIG["lambda_high_grade_risk"]
                * losses["high_grade_risk_loss"]
            )
            base_loss_total += float(base_loss.detach()) * size
            residual_abs_total += (
                float(output["residual_logit_abs_mean"].detach()) * size
            )
            labels.append(targets["diagnosis_labels"].cpu().numpy())
            probabilities.append(output["diagnosis_probs"].cpu().numpy())
            masks.append(targets["semantic_mask"].cpu().numpy())
            morph_true.append(targets["morph_labels"].cpu().numpy())
            morph_prob.append(output["morph_probs"][:, 1].cpu().numpy())
            evidence_true.append(targets["evidence_labels"].cpu().numpy())
            evidence_prob.append(output["evidence_probs"][:, 1].cpu().numpy())
            paths.extend(batch["image_path"])
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    semantic_mask = np.concatenate(masks).astype(bool)
    audit = {
        "morph_true": np.concatenate(morph_true),
        "morph_prob_high": np.concatenate(morph_prob),
        "evidence_true": np.concatenate(evidence_true),
        "evidence_prob_definitive": np.concatenate(evidence_prob),
        "semantic_mask": semantic_mask,
    }
    metrics = compute_locked_candidate_metrics(y_true, y_prob)
    metrics.update(
        compute_semantic_metrics(
            audit["morph_true"],
            audit["morph_prob_high"],
            audit["evidence_true"],
            audit["evidence_prob_definitive"],
            semantic_mask,
            y_prob,
        )
    )
    metrics.update(
        finalize_loss_metrics(
            component_totals,
            base_loss_total,
            risk_loss_total,
            high_grade_count,
            count,
        )
    )
    metrics["residual_adapter_weight_norm"] = float(
        model.residual_adapter.weight.detach().float().norm()
    )
    metrics["residual_logit_abs_mean"] = residual_abs_total / max(count, 1)
    return metrics, y_true, y_prob, paths, audit


__all__ = (
    "LOSS_NAMES",
    "build_model_optimizer_scheduler",
    "evaluate_s1r3",
    "finalize_loss_metrics",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
