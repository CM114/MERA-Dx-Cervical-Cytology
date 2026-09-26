"""Training primitives for the locked S1-R2 semantic residual model."""

from __future__ import annotations

import numpy as np

from experiments.tbs.s1r2_protocol import LOCKED_S1R2_CONFIG
from experiments.tbs.s1r_training import (
    LOSS_NAMES,
    make_loader,
    seed_everything,
    train_epoch,
)


def build_model_optimizer_scheduler(device):
    import torch
    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.s1r2_model import build_tbs_s1r2_model

    model = build_tbs_s1r2_model(
        model_name=LOCKED_S1R2_CONFIG["model_name"],
        pretrained=LOCKED_S1R2_CONFIG["pretrained"],
        semantic_dim=LOCKED_S1R2_CONFIG["semantic_dim"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_S1R2_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_S1R2_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_S1R2_CONFIG["lr"],
        weight_decay=LOCKED_S1R2_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LOCKED_S1R2_CONFIG["epochs"]
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


def _loss_kwargs():
    return {
        "stage": "s1",
        "lambda_screen": LOCKED_S1R2_CONFIG["lambda_screen"],
        "lambda_morph": LOCKED_S1R2_CONFIG["lambda_morph"],
        "lambda_evidence": LOCKED_S1R2_CONFIG["lambda_evidence"],
        "lambda_decorr": LOCKED_S1R2_CONFIG["lambda_decorr"],
    }


def evaluate_s1r2(model, loader, device):
    import torch
    from experiments.tbs.metrics import compute_semantic_metrics
    from experiments.tbs.singleview_model import singleview_tbs_loss
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
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
                losses = singleview_tbs_loss(output, targets, **_loss_kwargs())
            size = len(images)
            count += size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * size
            residual_abs_total += float(output["residual_logit_abs_mean"].detach()) * size
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
    metrics.update({name: totals[name] / max(count, 1) for name in LOSS_NAMES})
    metrics["residual_adapter_weight_norm"] = float(
        model.residual_adapter.weight.detach().float().norm()
    )
    metrics["residual_logit_abs_mean"] = residual_abs_total / max(count, 1)
    return metrics, y_true, y_prob, paths, audit


__all__ = (
    "build_model_optimizer_scheduler",
    "evaluate_s1r2",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
