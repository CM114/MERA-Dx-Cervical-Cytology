"""Run the existing Stage-1 trainer with audited backbone initialization."""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.train_tbs_stage1 as stage1
from experiments.tbs.backbone_transfer import (
    file_sha256,
    load_backbone_checkpoint,
)


def _extract_transfer_argument(argv):
    remaining = []
    checkpoint = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--init_backbone_checkpoint":
            if checkpoint is not None or index + 1 >= len(argv):
                raise ValueError("--init_backbone_checkpoint must appear once with a path")
            checkpoint = Path(argv[index + 1]).resolve(strict=True)
            index += 2
            continue
        if token.startswith("--init_backbone_checkpoint="):
            if checkpoint is not None:
                raise ValueError("--init_backbone_checkpoint must appear once")
            checkpoint = Path(token.split("=", 1)[1]).resolve(strict=True)
            index += 1
            continue
        remaining.append(token)
        index += 1
    if checkpoint is None:
        raise ValueError("--init_backbone_checkpoint is required")
    return checkpoint, remaining


def _resolve_output_directory(argv):
    for index, token in enumerate(argv):
        if token == "--out_dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve(strict=False)
        if token.startswith("--out_dir="):
            return Path(token.split("=", 1)[1]).resolve(strict=False)
    raise ValueError("The transfer launcher requires an explicit --out_dir")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    checkpoint, remaining = _extract_transfer_argument(argv)
    out_dir = _resolve_output_directory(remaining)
    original_builder = stage1.build_stage1_model
    original_environment = stage1._environment_payload

    def build_with_transfer(*args, **kwargs):
        model = original_builder(*args, **kwargs)
        summary = load_backbone_checkpoint(model, checkpoint)
        with (out_dir / "transfer_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return model

    def environment_with_transfer(args):
        payload = original_environment(args)
        payload.update(
            {
                "init_backbone_checkpoint": str(checkpoint),
                "init_backbone_checkpoint_sha256": file_sha256(checkpoint),
                "transfer_policy": "complete_backbone_only",
                "target_heldout_opened": False,
                "target_calibration_opened": False,
                "target_sealed_test_opened": False,
            }
        )
        return payload

    stage1.build_stage1_model = build_with_transfer
    stage1._environment_payload = environment_with_transfer
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        return stage1.main()
    finally:
        sys.argv = old_argv
        stage1.build_stage1_model = original_builder
        stage1._environment_payload = original_environment


if __name__ == "__main__":
    raise SystemExit(main())
