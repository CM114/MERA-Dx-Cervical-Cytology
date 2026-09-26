"""M1 gradient conflict audit.

Uses torch.autograd.grad on representative backbone weights to compute
cos(g_L5, g_Lscreen) at each batch without multiple backward passes.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
from experiments.tbs.models import build_stage1_model
from experiments.tbs.metrics import compute_stage1_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="M1 gradient conflict audit")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--lambda_screen", type=float, default=0.3)
    return p.parse_args(argv)


def _stage_from_name(name):
    m = re.search(r'stages\.(\d+)', name)
    if m: return f"stage{m.group(1)}"
    m = re.search(r'blocks\.(\d+)', name)
    if m: return f"stage{m.group(1)}"
    return "other"


def _representative_params(model, n_per_stage=6):
    """Pick a small set of backbone weight tensors for gradient analysis."""
    params = []
    stage_counts = {}
    for name, p in model.named_parameters():
        if "backbone" not in name or p.requires_grad is False:
            continue
        if p.ndim < 2:
            continue
        stage = _stage_from_name(name)
        if stage_counts.get(stage, 0) < n_per_stage:
            params.append((name, p))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    return params


def gradient_audit_epoch(model, loader, optimizer, device, amp_enabled, lambda_screen):
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rep = _representative_params(model)
    rep_tensors = [p for _, p in rep]
    rep_names = [n for n, _ in rep]

    total_loss_val = 0.0
    cos_sum = 0.0
    cos_n = 0
    conflict_batches = 0
    total_batches = 0
    stage_cos_sum = {}
    stage_cos_n = {}
    asc_h_conflict = 0
    asc_h_batches = 0
    g5_norm_sum = 0.0
    gs_norm_sum = 0.0

    model.train()
    for images, labels, screen_labels, _, _ in _iterate_m1(loader, device):
        optimizer.zero_grad(set_to_none=True)

        # Single forward pass
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            L5 = nn.functional.cross_entropy(output["diagnosis_logits"], labels)
            Ls = lambda_screen * nn.functional.binary_cross_entropy_with_logits(
                output["screen_logits"], screen_labels.float()
            )
            total_loss = L5 + Ls

        # Gradient decomposition via autograd.grad
        # Both need retain_graph=True so the graph survives for the training backward
        grads_5 = torch.autograd.grad(
            L5, rep_tensors, retain_graph=True, allow_unused=True,
        )
        grads_s = torch.autograd.grad(
            Ls, rep_tensors, retain_graph=True, allow_unused=True,
        )

        batch_cos = []
        for i, (name, g5, gs) in enumerate(zip(rep_names, grads_5, grads_s)):
            if g5 is None or gs is None:
                batch_cos.append(0.0)
                continue
            g5f = g5.detach().flatten().float()
            gsf = gs.detach().flatten().float()
            n5, ns = g5f.norm(), gsf.norm()
            if n5 > 1e-12 and ns > 1e-12:
                cv = float((g5f @ gsf) / (n5 * ns))
            else:
                cv = 0.0
            batch_cos.append(cv)
            g5_norm_sum += float(n5)
            gs_norm_sum += float(ns)

            stage = _stage_from_name(name)
            stage_cos_sum[stage] = stage_cos_sum.get(stage, 0.0) + cv
            stage_cos_n[stage] = stage_cos_n.get(stage, 0) + 1

        mean_cos = float(np.mean(batch_cos)) if batch_cos else 0.0
        cos_sum += mean_cos
        cos_n += 1
        if mean_cos < 0:
            conflict_batches += 1
        total_batches += 1

        if bool((labels == 3).any()):  # ASC-H in batch
            asc_h_batches += 1
            if mean_cos < 0:
                asc_h_conflict += 1

        # Normal training step
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_val += total_loss.item() * images.size(0)

    n = max(len(loader.dataset), 1)
    tb = max(total_batches, 1)
    return {
        "train_loss": total_loss_val / n,
        "mean_cos_g5_gscreen": cos_sum / max(cos_n, 1),
        "conflict_batch_rate": conflict_batches / tb,
        "total_batches": total_batches,
        "asc_h_conflict_rate": asc_h_conflict / max(asc_h_batches, 1) if asc_h_batches > 0 else float("nan"),
        "mean_g5_norm": g5_norm_sum / max(cos_n, 1),
        "mean_gs_norm": gs_norm_sum / max(cos_n, 1),
        "per_stage_cosine": {s: stage_cos_sum[s] / max(stage_cos_n[s], 1) for s in stage_cos_sum},
    }


def _iterate_m1(loader, device):
    for batch in loader:
        yield (
            batch["image"].to(device),
            batch["diagnosis_label"].to(device),
            batch["screen_label"].to(device),
            (batch["diagnosis_label"] == 3),
            None,
        )


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    probs, labels = [], []
    for images, lbls, _, _, _ in _iterate_m1(loader, device):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
        probs.append(output["diagnosis_probs"].cpu())
        labels.append(lbls.cpu())
    return compute_stage1_metrics(
        torch.cat(labels).numpy(), torch.cat(probs).numpy()
    )


def main(argv=None):
    args = parse_args(argv)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_stage1_model(variant="m1", model_name="caformer_s18", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    train_loader = DataLoader(
        XUDataTBS5Dataset(args.train_csv, eval_tf),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        XUDataTBS5Dataset(args.dev_csv, eval_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    best_f1 = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = gradient_audit_epoch(model, train_loader, optimizer, device, amp_enabled, args.lambda_screen)
        scheduler.step()
        dev = evaluate(model, dev_loader, device, amp_enabled)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": stats["train_loss"],
            "dev_macro_f1": float(dev["macro_f1"]),
            "mean_cos": stats["mean_cos_g5_gscreen"],
            "conflict_rate": stats["conflict_batch_rate"],
            "asc_h_conflict_rate": stats["asc_h_conflict_rate"],
        }
        for s, v in stats["per_stage_cosine"].items():
            row[f"cos_{s}"] = v
        rows.append(row)
        print(f"Epoch {epoch:2d}: loss={stats['train_loss']:.4f} f1={row['dev_macro_f1']:.4f} "
              f"cos={row['mean_cos']:.4f} conflict={row['conflict_rate']:.2%} ({dt:.1f}s)",
              flush=True)
        if row["dev_macro_f1"] > best_f1:
            best_f1 = row["dev_macro_f1"]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gradient_audit.csv", index=False)

    mean_conflict = float(df["conflict_rate"].mean())
    late_conflict = float(df[df["epoch"] > 15]["conflict_rate"].mean())
    print(f"\nMean conflict rate: {mean_conflict:.2%}", flush=True)
    print(f"Late (16-30) conflict rate: {late_conflict:.2%}", flush=True)
    decision = "M1_GP_ELIGIBLE" if mean_conflict > 0.20 else "CLOSE_M1_AUXILIARY"
    print(f"Decision: {decision}", flush=True)

    json.dump({
        "mean_conflict_rate": mean_conflict,
        "late_conflict_rate": late_conflict,
        "decision": decision,
        "best_dev_macro_f1": best_f1,
    }, open(out_dir / "audit_summary.json", "w"), indent=2)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
