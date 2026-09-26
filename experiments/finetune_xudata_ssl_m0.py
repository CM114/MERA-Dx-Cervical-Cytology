"""Fine-tune an SSL backbone with train-only selection and one-shot dev evaluation."""

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
    parser.add_argument("--ssl_checkpoint", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        validate_train_dev_paths(args.train_csv, args.dev_csv)
        reject_forbidden_data_path(args.ssl_checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def prepare_internal_manifests(train_csv, output_dir):
    return make_internal_train_split(train_csv, output_dir, select_fraction=0.1)


def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_ssl_backbone(model, checkpoint, device):
    import torch

    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("backbone_state")
    if state is None:
        raise ValueError("SSL checkpoint is missing backbone_state")
    model.backbone.load_state_dict(state, strict=True)


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
    total = 0.0
    count = 0
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
            loss = F.cross_entropy(output["diagnosis_logits"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


@__import__("contextlib").contextmanager
def _inference_context():
    import torch

    with torch.no_grad():
        yield


def _evaluate(model, loader, device):
    import torch

    model.eval()
    labels, probabilities, image_paths = [], [], []
    with _inference_context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
            labels.append(batch["diagnosis_label"].numpy())
            probabilities.append(output["diagnosis_probs"].float().cpu().numpy())
            image_paths.extend(batch["image_path"])
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    return compute_locked_candidate_metrics(y_true, y_prob), y_true, y_prob, image_paths


def run_finetune(args):
    import torch
    from experiments.tbs.models import build_stage1_model
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required unless --device cpu is specified")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    fit_csv, select_csv = prepare_internal_manifests(
        args.train_csv, out_dir / "internal_split"
    )
    model = build_stage1_model(
        "m0", model_name=args.model_name, pretrained=False
    ).to(device)
    _load_ssl_backbone(model, args.ssl_checkpoint, device)
    train_transform, eval_transform = build_transforms(
        args.img_size, input_mode="letterbox"
    )
    fit_ds = XUDataTBS5Dataset(fit_csv, train_transform)
    select_ds = XUDataTBS5Dataset(select_csv, eval_transform)
    dev_ds = XUDataTBS5Dataset(args.dev_csv, eval_transform)
    fit_loader = _loader(fit_ds, args, True)
    select_loader = _loader(select_ds, args, False)
    dev_loader = _loader(dev_ds, args, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_select = -1.0
    best_path = out_dir / "best_model.pth"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, fit_loader, optimizer, scaler, device)
        select_metrics, _, _, _ = _evaluate(model, select_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "select_macro_f1": select_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["select_macro_f1"] > best_select:
            best_select = row["select_macro_f1"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch},
                best_path,
            )

    payload = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"], strict=True)
    dev_metrics, y_true, y_prob, image_paths = _evaluate(model, dev_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true,
            "pred_label": y_prob.argmax(axis=1),
        }
    )
    for class_index in range(5):
        prediction_frame[f"prob_{class_index}"] = y_prob[:, class_index]
    prediction_frame.to_csv(
        out_dir / "dev_predictions.csv", index=False, lineterminator="\n"
    )
    (out_dir / "dev_metrics.json").write_text(
        json.dumps(dev_metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        vars(args),
        {
            "route": "XUDATA_ONLY_SSL_FINETUNE",
            "selection_split": "train_select",
            "dev_evaluation_count": 1,
            "model_trained": True,
        },
    )
    (out_dir / "completed.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dev_evaluation_count": 1,
                "calibration_used": False,
                "test_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dev_metrics


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_finetune(args), indent=2))


if __name__ == "__main__":
    main()
