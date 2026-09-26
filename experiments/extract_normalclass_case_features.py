"""Extract frozen patch features for normalClass case-level MIL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save_feature_npz(path, features, labels, case_ids, image_paths):
    path = Path(path)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    case_ids = np.asarray(case_ids).astype(str)
    image_paths = np.asarray(image_paths).astype(str)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [patches, dimensions]")
    if any(array.ndim != 1 for array in (labels, case_ids, image_paths)):
        raise ValueError("metadata arrays must be one-dimensional")
    if len({features.shape[0], labels.size, case_ids.size, image_paths.size}) != 1:
        raise ValueError("feature and metadata arrays must have equal length")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        case_ids=case_ids,
        image_paths=image_paths,
    )


def load_feature_npz(path):
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "labels", "case_ids", "image_paths"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Feature NPZ is missing arrays: {missing}")
        values = {name: payload[name].copy() for name in required}
    features = np.asarray(values["features"], dtype=np.float32)
    labels = np.asarray(values["labels"], dtype=np.int64)
    case_ids = np.asarray(values["case_ids"]).astype(str)
    image_paths = np.asarray(values["image_paths"]).astype(str)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [patches, dimensions]")
    if any(array.ndim != 1 for array in (labels, case_ids, image_paths)):
        raise ValueError("metadata arrays must be one-dimensional")
    if len({features.shape[0], labels.size, case_ids.size, image_paths.size}) != 1:
        raise ValueError("feature and metadata arrays must have equal length")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    values = {
        "features": features,
        "labels": labels,
        "case_ids": case_ids,
        "image_paths": image_paths,
    }
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_features(checkpoint, manifest, out_npz, out_summary, model_name, variant, img_size, batch_size, num_workers, device, amp):
    import torch
    from torch.utils.data import DataLoader

    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.models import build_stage1_model

    device = torch.device(device)
    model = build_stage1_model(variant, model_name=model_name, pretrained=False)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _, eval_transform = build_transforms(img_size, input_mode="letterbox")
    dataset = XUDataTBS5Dataset(manifest, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    by_path = {str(row["image_path"]): row for row in dataset.records}
    all_features, all_labels, all_case_ids, all_paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(images)
            features = output["features"].float().cpu().numpy()
            paths = [str(path) for path in batch["image_path"]]
            all_features.append(features)
            all_paths.extend(paths)
            all_labels.extend(int(batch["diagnosis_label"][index]) for index in range(len(paths)))
            try:
                all_case_ids.extend(str(by_path[path]["case_id"]) for path in paths)
            except KeyError as exc:
                raise ValueError("Manifest must contain case_id for case MIL extraction") from exc
    features = np.concatenate(all_features, axis=0)
    save_feature_npz(out_npz, features, all_labels, all_case_ids, all_paths)
    summary = {
        "schema_version": "xudata-replacement-normalclass-case-features-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": _sha256(manifest),
        "feature_npz": str(Path(out_npz).resolve()),
        "patch_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "case_count": int(len(set(all_case_ids))),
        "variant": variant,
        "model_name": model_name,
        "heldout_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "raw_data_modified": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    Path(out_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--out_summary", type=Path, required=True)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--variant", choices=("m0", "m1", "m2"), default="m2")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args(argv)
    extract_features(
        args.checkpoint,
        args.manifest,
        args.out_npz,
        args.out_summary,
        args.model_name,
        args.variant,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.device,
        not args.no_amp,
    )


if __name__ == "__main__":
    main()
