"""Extract frozen M0 features for any split (calibration/test).

Minimal script — loads M0 checkpoint, forward pass, saves .npy features.
"""

import sys, argparse
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.tbs.models import build_stage1_model
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract M0 features")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out_features", type=Path, required=True)
    p.add_argument("--out_labels", type=Path, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda:1")
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=True)
    model.to(device)
    model.eval()

    _, eval_tf = build_transforms(args.img_size, input_mode="letterbox")
    ds = XUDataTBS5Dataset(args.csv, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_feat, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"]
        with torch.cuda.amp.autocast(enabled=True):
            output = model(images)
        all_feat.append(output["features"].cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_feat)
    labels = np.concatenate(all_labels)
    np.save(args.out_features, features)
    np.save(args.out_labels, labels)
    print(f"Saved: {args.out_features} ({features.shape}), {args.out_labels} ({labels.shape})", flush=True)


if __name__ == "__main__":
    main()
