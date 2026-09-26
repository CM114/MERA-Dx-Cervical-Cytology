"""Pretrain a CaFormer backbone with xudata train images only."""

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

from experiments.tbs.xudata_ssl import SimSiamModel, build_ssl_backbone
from experiments.xudata_gain_common import (
    reject_forbidden_data_path,
    write_safety_artifacts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args(argv)
    try:
        reject_forbidden_data_path(args.train_csv)
    except ValueError as exc:
        parser.error(str(exc))
    return args


class _TwoViewDataset:
    def __init__(self, csv_path, transform):
        self.frame = pd.read_csv(csv_path)
        if "image_path" not in self.frame.columns:
            raise ValueError("train CSV must contain image_path")
        if self.frame.empty:
            raise ValueError("train CSV is empty")
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from PIL import Image

        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.transform(image)


def _build_transform(img_size):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_pretraining(args):
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required unless --device cpu is specified")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    backbone, feature_dim = build_ssl_backbone(
        args.model_name, pretrained=not args.no_pretrained
    )
    model = SimSiamModel(
        backbone,
        feature_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    dataset = _TwoViewDataset(args.train_csv, _build_transform(args.img_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for first, second in loader:
            first, second = first.to(device), second.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss, _, _ = model(first, second)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * first.size(0)
            sample_count += first.size(0)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total_loss / max(sample_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state": model.backbone.state_dict(),
            "model_name": args.model_name,
            "feature_dim": feature_dim,
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        out_dir / "backbone_checkpoint.pth",
    )
    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
    write_safety_artifacts(
        out_dir,
        vars(args),
        {
            "route": "XUDATA_ONLY_SIMSIAM_PRETRAIN",
            "train_image_count": len(dataset),
            "model_trained": True,
        },
    )
    return history[-1]


def main(argv=None):
    args = parse_args(argv)
    result = run_pretraining(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
