"""Training/evaluation primitives for the locked C0-R2 boundary control."""

from __future__ import annotations

import random

import numpy as np

from experiments.tbs.c0r2_protocol import LOCKED_C0R2_CONFIG


def seed_everything(seed):
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, shuffle, num_workers, device, seed):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=LOCKED_C0R2_CONFIG["batch_size"],
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.c0r2_model import build_tbs_c0r2_model
    from experiments.tbs.optimization import build_discriminative_parameter_groups

    epochs = LOCKED_C0R2_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C0R2_CONFIG["epochs"]:
        raise ValueError("C0-R2 epochs are locked to 30")
    model = build_tbs_c0r2_model(
        model_name=LOCKED_C0R2_CONFIG["model_name"],
        pretrained=LOCKED_C0R2_CONFIG["pretrained"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C0R2_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C0R2_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C0R2_CONFIG["lr"],
        weight_decay=LOCKED_C0R2_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def train_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.tbs.c0r2_loss import c0r2_loss

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
            losses = c0r2_loss(
                output,
                labels,
                lambda_geometry=LOCKED_C0R2_CONFIG["lambda_geometry"],
                lambda_residual=LOCKED_C0R2_CONFIG["lambda_residual"],
                lambda_full_anchor=LOCKED_C0R2_CONFIG["lambda_full_anchor"],
                lambda_low_grade_pair=LOCKED_C0R2_CONFIG["lambda_low_grade_pair"],
                lambda_high_grade_mass=LOCKED_C0R2_CONFIG["lambda_high_grade_mass"],
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
        raise ValueError("C0-R2 training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


def evaluate_c0r2(model, loader, device):
    import torch

    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    labels, fused, full = [], [], []
    residual_means, residual_maxima = [], []
    areas, translations, similarities, geometry_losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            labels.append(batch["diagnosis_label"].detach().cpu().numpy())
            fused.append(output["diagnosis_probs"].detach().float().cpu().numpy())
            full.append(output["full_diagnosis_probs"].detach().float().cpu().numpy())
            residual_means.append(float(output["residual_logit_abs_mean"].detach().cpu()))
            residual_maxima.append(float(output["residual_logit_max_abs"].detach().cpu()))
            areas.append(float(output["local_area"].detach().float().cpu().mean()))
            translations.append(float(output["local_translation_abs"].detach().float().cpu().mean()))
            similarities.append(float(output["local_cosine_similarity"].detach().float().cpu().mean()))
            geometry_losses.append(float(output["geometry_loss"].detach().float().cpu()))
    if not labels:
        raise ValueError("C0-R2 evaluation loader is empty")
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_fused = np.concatenate(fused)
    y_full = np.concatenate(full)
    metrics = compute_locked_candidate_metrics(y_true, y_fused)
    full_metrics = compute_locked_candidate_metrics(y_true, y_full)
    metrics.update({
        "full_macro_f1": float(full_metrics["macro_f1"]),
        "residual_logit_abs_mean": float(np.mean(residual_means)),
        "residual_logit_max_abs": float(np.max(residual_maxima)),
        "local_area_mean": float(np.mean(areas)),
        "local_translation_abs_mean": float(np.mean(translations)),
        "local_cosine_similarity_mean": float(np.mean(similarities)),
        "geometry_loss": float(np.mean(geometry_losses)),
    })
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "y_true": y_true,
        "y_prob": y_fused,
        "full_y_prob": y_full,
    }


__all__ = (
    "build_model_optimizer_scheduler", "evaluate_c0r2", "make_loader",
    "seed_everything", "train_epoch",
)
