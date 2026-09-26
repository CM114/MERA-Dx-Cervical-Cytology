"""Shared historical-M0-compatible training primitives for S0-R."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG


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
        batch_size=LOCKED_S0R_CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model_optimizer_scheduler(device, total_epochs):
    import torch
    from experiments.tbs.models import build_stage1_model
    from experiments.tbs.optimization import build_discriminative_parameter_groups

    model = build_stage1_model(
        "m0",
        model_name=LOCKED_S0R_CONFIG["model_name"],
        pretrained=LOCKED_S0R_CONFIG["pretrained"],
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_S0R_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_S0R_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_S0R_CONFIG["lr"],
        weight_decay=LOCKED_S0R_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(total_epochs)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def _train_args(device):
    return SimpleNamespace(
        variant="m0",
        lambda_screen=0.0,
        label_smoothing=LOCKED_S0R_CONFIG["label_smoothing"],
        boundary_loss="none",
        temperature=0.1,
        lambda_pb=0.0,
        amp=str(device).startswith("cuda"),
    )


def train_epoch(model, loader, optimizer, scaler, device):
    from experiments.train_tbs_stage1 import train_one_epoch

    return train_one_epoch(
        model, loader, optimizer, scaler, device, _train_args(device)
    )


def evaluate_m0(model, loader, device):
    from experiments.train_tbs_stage1 import evaluate

    return evaluate(model, loader, device, _train_args(device))
