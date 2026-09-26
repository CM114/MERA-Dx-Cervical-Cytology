"""Train the xudata-only RGB full/local dual-view classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.xudata_gain_common import (
    compute_locked_candidate_metrics,
    make_internal_train_split,
    reject_forbidden_data_path,
    validate_train_dev_paths,
    write_safety_artifacts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0_checkpoint", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        validate_train_dev_paths(args.train_csv, args.dev_csv)
        reject_forbidden_data_path(args.m0_checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, args, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _train_epoch(model, loader, optimizer, scaler, device):
    import torch
    import torch.nn.functional as F

    model.train()
    total, count = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["diagnosis_label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(images)
            loss = F.cross_entropy(output["logits"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


def _evaluate(model, loader, device):
    import torch

    model.eval()
    labels, probabilities, image_paths, theta_values = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
            labels.append(batch["diagnosis_label"].numpy())
            probabilities.append(torch.softmax(output["logits"].float(), dim=1).cpu().numpy())
            image_paths.extend(batch["image_path"])
            theta_values.append(output["theta"].float().cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    theta = np.concatenate(theta_values)
    metrics = compute_locked_candidate_metrics(y_true, y_prob)
    theta_matrix = theta.reshape(len(theta), 2, 3)
    metrics.update(
        {
            "theta_center_x_mean": float(theta_matrix[:, 0, 2].mean()),
            "theta_center_y_mean": float(theta_matrix[:, 1, 2].mean()),
            "theta_area_scale_mean": float(
                np.abs(
                    theta_matrix[:, 0, 0] * theta_matrix[:, 1, 1]
                    - theta_matrix[:, 0, 1] * theta_matrix[:, 1, 0]
                ).mean()
            ),
        }
    )
    return metrics, y_true, y_prob, image_paths


def run_training(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.xudata_dualview_model import build_xudata_dualview_from_m0

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required unless --device cpu is specified")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    fit_csv, select_csv = make_internal_train_split(
        args.train_csv, out_dir / "internal_split", select_fraction=0.1
    )
    model = build_xudata_dualview_from_m0(
        args.m0_checkpoint, device, model_name=args.model_name
    )
    train_transform, eval_transform = build_transforms(
        args.img_size, input_mode="letterbox"
    )
    fit_loader = _loader(XUDataTBS5Dataset(fit_csv, train_transform), args, True)
    select_loader = _loader(XUDataTBS5Dataset(select_csv, eval_transform), args, False)
    dev_loader = _loader(XUDataTBS5Dataset(args.dev_csv, eval_transform), args, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_select = -1.0
    best_path = out_dir / "best_model.pth"
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(model, fit_loader, optimizer, scaler, device)
        select_metrics, _, _, _ = _evaluate(model, select_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "select_macro_f1": select_metrics["macro_f1"],
            "select_theta_area_scale_mean": select_metrics["theta_area_scale_mean"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["select_macro_f1"] > best_select:
            best_select = row["select_macro_f1"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dev_metrics, y_true, y_prob, image_paths = _evaluate(model, dev_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true,
            "pred_label": y_prob.argmax(axis=1),
        }
    )
    for class_index in range(5):
        predictions[f"prob_{class_index}"] = y_prob[:, class_index]
    predictions.to_csv(out_dir / "dev_predictions.csv", index=False, lineterminator="\n")
    (out_dir / "dev_metrics.json").write_text(
        json.dumps(dev_metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        vars(args),
        {
            "route": "XUDATA_ONLY_RGB_DUALVIEW",
            "selection_split": "train_select",
            "dev_evaluation_count": 1,
            "model_trained": True,
            "external_masks_used": False,
        },
    )
    (out_dir / "completed.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dev_evaluation_count": 1,
                "calibration_used": False,
                "test_used": False,
                "external_masks_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dev_metrics


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_training(args), indent=2))


if __name__ == "__main__":
    main()
