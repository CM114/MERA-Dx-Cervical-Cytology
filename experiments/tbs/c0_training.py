"""Training and evaluation primitives for the locked C0 dual-view control."""

from __future__ import annotations

import random

import numpy as np

from experiments.tbs.c0_protocol import LOCKED_C0_CONFIG


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
        batch_size=LOCKED_C0_CONFIG["batch_size"],
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model_optimizer_scheduler(device, total_epochs=None):
    import torch

    from experiments.tbs.c0_model import build_tbs_c0_model
    from experiments.tbs.optimization import build_discriminative_parameter_groups

    epochs = LOCKED_C0_CONFIG["epochs"] if total_epochs is None else int(total_epochs)
    if epochs != LOCKED_C0_CONFIG["epochs"]:
        raise ValueError("C0 epochs are locked to 30")
    model = build_tbs_c0_model(
        model_name=LOCKED_C0_CONFIG["model_name"],
        pretrained=LOCKED_C0_CONFIG["pretrained"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_C0_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_C0_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_C0_CONFIG["lr"],
        weight_decay=LOCKED_C0_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def train_epoch(model, loader, optimizer, scaler, device):
    import torch

    from experiments.tbs.c0_loss import c0_loss

    model.train()
    totals = {"loss": 0.0, "diagnosis_loss": 0.0, "geometry_loss": 0.0}
    sample_count = 0
    amp_enabled = bool(scaler.is_enabled())
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["diagnosis_label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            losses = c0_loss(
                output,
                labels,
                lambda_geometry=LOCKED_C0_CONFIG["lambda_geometry"],
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
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * count
    if sample_count <= 0:
        raise ValueError("C0 training loader is empty")
    return {key: value / sample_count for key, value in totals.items()}


def _metric_aliases(labels, probabilities):
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    return compute_locked_candidate_metrics(labels, probabilities)


def evaluate_c0(model, loader, device):
    """Evaluate fused, full-only, local-only outputs and view diagnostics."""

    import torch

    model.eval()
    labels = []
    fused = []
    full = []
    local = []
    gates = []
    areas = []
    translations = []
    similarities = []
    geometry_losses = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            labels.append(batch["diagnosis_label"].detach().cpu().numpy())
            fused.append(output["diagnosis_probs"].detach().float().cpu().numpy())
            full.append(output["full_diagnosis_probs"].detach().float().cpu().numpy())
            local.append(output["local_diagnosis_probs"].detach().float().cpu().numpy())
            gates.append(output["view_gate"].detach().float().cpu().reshape(-1).numpy())
            areas.append(output["local_area"].detach().float().cpu().reshape(-1).numpy())
            translations.append(
                output["local_translation_abs"].detach().float().cpu().reshape(-1).numpy()
            )
            similarities.append(
                output["local_cosine_similarity"].detach().float().cpu().reshape(-1).numpy()
            )
            geometry_losses.append(float(output["geometry_loss"].detach().float().cpu().item()))
    if not labels:
        raise ValueError("C0 evaluation loader is empty")
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    y_fused = np.concatenate(fused)
    y_full = np.concatenate(full)
    y_local = np.concatenate(local)
    metrics = _metric_aliases(y_true, y_fused)
    full_metrics = _metric_aliases(y_true, y_full)
    local_metrics = _metric_aliases(y_true, y_local)
    metrics.update(
        {
            "full_macro_f1": full_metrics["macro_f1"],
            "local_macro_f1": local_metrics["macro_f1"],
            "view_gate_mean": float(np.concatenate(gates).mean()),
            "local_area_mean": float(np.concatenate(areas).mean()),
            "local_translation_abs_mean": float(np.concatenate(translations).mean()),
            "local_cosine_similarity_mean": float(np.concatenate(similarities).mean()),
            "geometry_loss": float(np.mean(geometry_losses)),
        }
    )
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "y_true": y_true,
        "y_prob": y_fused,
        "full_y_prob": y_full,
        "local_y_prob": y_local,
    }


__all__ = (
    "build_model_optimizer_scheduler",
    "evaluate_c0",
    "make_loader",
    "seed_everything",
    "train_epoch",
)
