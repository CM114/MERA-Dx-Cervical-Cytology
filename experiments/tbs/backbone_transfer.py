import hashlib
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone_checkpoint(model, checkpoint_path):
    """Load a complete compatible backbone while preserving target heads."""
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        source_state = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        source_state = payload["state_dict"]
    else:
        source_state = payload
    if not isinstance(source_state, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")

    target_state = model.state_dict()
    target_backbone_keys = {
        key for key in target_state if key.startswith("backbone.")
    }
    target_head_before = {
        key: value.detach().cpu().clone()
        for key, value in target_state.items()
        if not key.startswith("backbone.")
    }
    compatible = {}
    skipped = []
    shape_mismatches = []
    for raw_key, value in source_state.items():
        key = str(raw_key)
        if not key.startswith("backbone."):
            skipped.append(key)
            continue
        if key not in target_state:
            skipped.append(key)
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(key)
            continue
        compatible[key] = value

    missing_backbone_keys = sorted(target_backbone_keys - set(compatible))
    if missing_backbone_keys or shape_mismatches:
        raise RuntimeError(
            "Incomplete backbone transfer; refusing partial initialization. "
            f"missing={missing_backbone_keys}, "
            f"shape_mismatches={sorted(shape_mismatches)}"
        )
    if not compatible:
        raise RuntimeError(
            f"No compatible backbone weights found in checkpoint: {checkpoint_path}"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading backbone: {unexpected}")
    missing_after_load = sorted(
        key for key in missing if key.startswith("backbone.")
    )
    if missing_after_load:
        raise RuntimeError(
            f"Backbone keys remained missing after load: {missing_after_load}"
        )

    target_state_after = model.state_dict()
    changed_head_keys = [
        key
        for key, before in target_head_before.items()
        if not torch.equal(before, target_state_after[key].detach().cpu())
    ]
    if changed_head_keys:
        raise RuntimeError(
            "Target head parameters changed during backbone-only transfer: "
            f"{changed_head_keys}"
        )

    return {
        "schema_version": "stage1-backbone-transfer-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "loaded_key_count": len(compatible),
        "loaded_keys": sorted(compatible),
        "skipped_nonbackbone_or_unknown_keys": sorted(skipped),
        "shape_mismatch_keys": sorted(shape_mismatches),
        "missing_backbone_keys": missing_after_load,
        "target_head_keys": sorted(target_head_before),
        "target_head_parameters_verified_unchanged": True,
        "complete_target_backbone_loaded": True,
        "policy": "backbone_only; target five-class heads remain freshly initialized",
    }
