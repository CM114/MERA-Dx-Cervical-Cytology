"""Training and evaluation primitives for the locked S1-R protocol."""

from __future__ import annotations

import random

import numpy as np

from experiments.tbs.s1r_protocol import LOCKED_S1R_CONFIG


LOSS_NAMES = ("loss", "diagnosis_loss", "screen_loss", "morph_loss", "evidence_loss", "decorr_loss")


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        batch_size=LOCKED_S1R_CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _targets(batch, device):
    import torch

    return {
        "diagnosis_labels": batch["diagnosis_label"].to(device, non_blocking=True),
        "screen_labels": batch["screen_label"].to(device, non_blocking=True),
        "morph_labels": batch["morph_label"].to(device, non_blocking=True),
        "evidence_labels": batch["evidence_label"].to(device, non_blocking=True),
        "semantic_mask": batch["semantic_mask"].to(device, non_blocking=True).bool(),
    }


def build_model_optimizer_scheduler(device):
    import torch
    from experiments.tbs.optimization import build_discriminative_parameter_groups
    from experiments.tbs.singleview_model import build_tbs_singleview_model

    model = build_tbs_singleview_model(
        "s1",
        model_name=LOCKED_S1R_CONFIG["model_name"],
        pretrained=LOCKED_S1R_CONFIG["pretrained"],
        semantic_dim=LOCKED_S1R_CONFIG["semantic_dim"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_S1R_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_S1R_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_S1R_CONFIG["lr"],
        weight_decay=LOCKED_S1R_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LOCKED_S1R_CONFIG["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def _loss_kwargs():
    return {
        "stage": "s1",
        "lambda_screen": LOCKED_S1R_CONFIG["lambda_screen"],
        "lambda_morph": LOCKED_S1R_CONFIG["lambda_morph"],
        "lambda_evidence": LOCKED_S1R_CONFIG["lambda_evidence"],
        "lambda_decorr": LOCKED_S1R_CONFIG["lambda_decorr"],
    }


def train_epoch(model, loader, optimizer, scaler, device):
    import torch
    from experiments.tbs.singleview_model import singleview_tbs_loss

    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            losses = singleview_tbs_loss(model(images), targets, **_loss_kwargs())
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        size = len(images)
        count += size
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach()) * size
    return {name: value / max(count, 1) for name, value in totals.items()}


def evaluate_s1(model, loader, device):
    import torch
    from experiments.tbs.metrics import compute_semantic_metrics
    from experiments.tbs.singleview_model import singleview_tbs_loss
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    labels, probabilities = [], []
    morph_true, morph_prob, evidence_true, evidence_prob, masks = [], [], [], [], []
    paths = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = _targets(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(images)
                losses = singleview_tbs_loss(output, targets, **_loss_kwargs())
            size = len(images)
            count += size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * size
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
    metrics.update(compute_semantic_metrics(
        audit["morph_true"], audit["morph_prob_high"],
        audit["evidence_true"], audit["evidence_prob_definitive"],
        semantic_mask, y_prob,
    ))
    metrics.update({name: totals[name] / max(count, 1) for name in LOSS_NAMES})
    return metrics, y_true, y_prob, paths, audit
