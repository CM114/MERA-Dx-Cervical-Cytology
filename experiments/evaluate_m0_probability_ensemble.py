"""Evaluate a frozen three-seed M0 probability ensemble on clean_v2 dev."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.xudata_gain_common import (
    compute_locked_candidate_metrics,
    reject_forbidden_data_path,
    write_safety_artifacts,
)


def average_probability_arrays(probability_arrays):
    arrays = [np.asarray(values, dtype=np.float64) for values in probability_arrays]
    if not arrays:
        raise ValueError("at least one probability array is required")
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != 5:
        raise ValueError("probability arrays must have shape [samples, 5]")
    if any(values.shape != shape for values in arrays[1:]):
        raise ValueError("all probability arrays must have equal shapes")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("probability arrays must be finite")
    if any((values < 0).any() for values in arrays):
        raise ValueError("probability arrays must be non-negative")
    result = np.mean(arrays, axis=0)
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have positive sums")
    return result / row_sums


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if len(args.checkpoint) != 3:
        parser.error("exactly three --checkpoint arguments are required")
    try:
        reject_forbidden_data_path(args.dev_csv)
        for checkpoint in args.checkpoint:
            reject_forbidden_data_path(checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _load_checkpoint(path, device):
    import torch

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload.get("model_state", payload.get("state_dict", payload))
    return state


def _predict(checkpoint, dataset, args, device):
    import torch
    from torch.utils.data import DataLoader

    from experiments.tbs.models import build_stage1_model

    model = build_stage1_model(
        variant="m0",
        model_name=args.model_name,
        pretrained=False,
    ).to(device)
    model.load_state_dict(_load_checkpoint(checkpoint, device), strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    labels = []
    probabilities = []
    image_paths = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
        labels.append(batch["diagnosis_label"].numpy())
        probabilities.append(output["diagnosis_probs"].float().cpu().numpy())
        image_paths.extend(batch["image_path"])
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        image_paths,
    )


def _checkpoint_tag(path, index):
    match = re.search(r"seed(\d+)", str(path))
    return f"seed{match.group(1)}" if match else f"checkpoint{index + 1}"


def run_ensemble(args):
    import torch

    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required unless --device cpu is specified")
    device = torch.device(args.device)
    _, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    dataset = XUDataTBS5Dataset(args.dev_csv, eval_transform)
    all_labels = None
    all_paths = None
    probabilities = []
    model_metrics = {}
    for index, checkpoint in enumerate(args.checkpoint):
        labels, values, paths = _predict(checkpoint, dataset, args, device)
        if all_labels is None:
            all_labels, all_paths = labels, paths
        elif not np.array_equal(all_labels, labels) or all_paths != paths:
            raise RuntimeError("checkpoint evaluations changed dev ordering")
        probabilities.append(values)
        model_metrics[_checkpoint_tag(checkpoint, index)] = compute_locked_candidate_metrics(
            labels, values
        )

    ensemble_probabilities = average_probability_arrays(probabilities)
    ensemble_metrics = compute_locked_candidate_metrics(all_labels, ensemble_probabilities)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "image_path": all_paths,
            "true_label": all_labels,
            "ensemble_pred_label": ensemble_probabilities.argmax(axis=1),
        }
    )
    for class_index in range(5):
        rows[f"ensemble_prob_{class_index}"] = ensemble_probabilities[:, class_index]
    rows.to_csv(out_dir / "dev_predictions.csv", index=False, lineterminator="\n")
    (out_dir / "individual_metrics.json").write_text(
        json.dumps(model_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for tag, metrics in model_metrics.items():
        (out_dir / f"individual_{tag}_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (out_dir / "ensemble_metrics.json").write_text(
        json.dumps(ensemble_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_safety_artifacts(
        out_dir,
        vars(args),
        {
            "route": "FROZEN_M0_PROBABILITY_ENSEMBLE",
            "checkpoint_count": 3,
            "dev_evaluation_count": 1,
        },
    )
    return ensemble_metrics


def main(argv=None):
    args = parse_args(argv)
    metrics = run_ensemble(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
