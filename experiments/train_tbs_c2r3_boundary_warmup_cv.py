"""Run the sealed five-fold C2-R3 boundary-warmup candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.c0_protocol import sha256_file
from experiments.tbs.c2r3_boundary_warmup_protocol import (
    C2R3W_CV_COMPLETE_ROUTE,
    C2R3W_CV_SCHEMA,
    LOCKED_C2R3W_CONFIG,
    boundary_loss_scale,
    validate_complete_c2r3w_cv_artifacts,
)
from experiments.tbs.s0r_protocol import validate_s0r_folds
from experiments.tbs.s1r3_protocol import validate_sealed_source_path
from experiments.train_tbs_c0r2_cv import _read_s0r_summary, _write_epoch12_predictions
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import write_safety_artifacts


def parse_args(argv=None, validate_paths=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not validate_paths:
        return args
    try:
        for path in (
            args.train_fit_csv, args.select_s0_csv, args.select_s1_csv,
            args.fold_dir, args.s0r_cv_dir,
        ):
            validate_sealed_source_path(path, require_exists=True)
        validate_sealed_source_path(args.out_dir)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite C2-R3W CV output: {args.out_dir}")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        _read_s0r_summary(args.s0r_cv_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def run_cv(args):
    import torch

    from experiments.tbs.c2r3_conditional_boundary_training import (
        build_model_optimizer_scheduler,
        evaluate_c2,
        initialize_prototypes,
        make_loader,
        seed_everything,
        train_epoch,
    )
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms

    for path in (
        args.train_fit_csv, args.select_s0_csv, args.select_s1_csv,
        args.fold_dir, args.s0r_cv_dir,
    ):
        validate_sealed_source_path(path, require_exists=True)
    validate_sealed_source_path(args.out_dir)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite C2-R3W CV output: {args.out_dir}")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    s0r_summary_path, s0r_summary = _read_s0r_summary(args.s0r_cv_dir)
    baseline_completion = validate_complete_cv_artifacts(args.s0r_cv_dir)
    if s0r_summary.get("history_sha256") != baseline_completion.get("history_sha256"):
        raise ValueError("S0-R history checksum mismatch")
    fold_audit = validate_s0r_folds(args.train_fit_csv, args.select_s0_csv, args.fold_dir)
    if s0r_summary.get("fold_metadata_sha256") != fold_audit["fold_metadata_sha256"]:
        raise ValueError("S0-R fold metadata does not match supplied folds")
    if s0r_summary.get("pool_sha256") != fold_audit["pool_sha256"]:
        raise ValueError("S0-R pool does not match supplied folds")
    sealed_select_s1_sha256 = sha256_file(args.select_s1_csv)
    if s0r_summary.get("select_s1_sha256") != sealed_select_s1_sha256:
        raise ValueError("select_s1 checksum does not match S0-R baseline")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = build_transforms(
        LOCKED_C2R3W_CONFIG["img_size"], LOCKED_C2R3W_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    for fold in range(LOCKED_C2R3W_CONFIG["fold_count"]):
        fold_seed = LOCKED_C2R3W_CONFIG["seed"] + fold
        seed_everything(fold_seed)
        fold_output = args.out_dir / f"fold_{fold}"
        fold_output.mkdir()
        train_csv = args.fold_dir / f"fold_{fold}" / "train.csv"
        val_csv = args.fold_dir / f"fold_{fold}" / "val.csv"
        train_loader = make_loader(
            XUDataTBS5Dataset(train_csv, train_transform),
            True, args.num_workers, args.device, fold_seed,
        )
        prototype_loader = make_loader(
            XUDataTBS5Dataset(train_csv, eval_transform),
            False, args.num_workers, args.device, fold_seed,
        )
        val_loader = make_loader(
            XUDataTBS5Dataset(val_csv, eval_transform),
            False, args.num_workers, args.device, fold_seed,
        )
        model, optimizer, scheduler, scaler = build_model_optimizer_scheduler(
            device, LOCKED_C2R3W_CONFIG["epochs"]
        )
        initialize_prototypes(model, prototype_loader, device)
        history = []
        for epoch in range(1, LOCKED_C2R3W_CONFIG["epochs"] + 1):
            scale = boundary_loss_scale(
                epoch, LOCKED_C2R3W_CONFIG["boundary_warmup_epochs"]
            )
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                hard_example_fraction=LOCKED_C2R3W_CONFIG["hard_example_fraction"],
                boundary_hard_example_fraction=LOCKED_C2R3W_CONFIG[
                    "boundary_hard_example_fraction"
                ],
                boundary_loss_scale=scale,
            )
            evaluation = evaluate_c2(model, val_loader, device)
            row = {
                "fold": int(fold),
                "epoch": int(epoch),
                "train_boundary_loss_scale": float(scale),
                **{f"train_{key}": float(value) for key, value in train_metrics.items()},
                **{key: float(value) for key, value in evaluation["metrics"].items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(
                fold_output / "metrics.csv", index=False, lineterminator="\n"
            )
            if epoch == 12:
                _write_epoch12_predictions(fold_output / "epoch12_predictions.csv", evaluation)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()
        del model, optimizer, scheduler, scaler, train_loader, prototype_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = validate_complete_c2r3w_cv_artifacts(args.out_dir)
    summary.update({
        "route": C2R3W_CV_COMPLETE_ROUTE,
        "locked_training_config": LOCKED_C2R3W_CONFIG,
        "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
        "pool_sha256": fold_audit["pool_sha256"],
        "s0r_cv_summary_sha256": sha256_file(s0r_summary_path),
        "select_s1_sha256": sealed_select_s1_sha256,
        "select_s1_access": "byte_hash_only_not_parsed",
        "dev_accessed": False,
        "c1_checkpoint_used": False,
        "c2_checkpoint_used": False,
        "c2r1_checkpoint_used": False,
        "c2r2_checkpoint_used": False,
    })
    (args.out_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": C2R3W_CV_SCHEMA,
            "route": C2R3W_CV_COMPLETE_ROUTE,
            "select_s1_access": "byte_hash_only_not_parsed",
            "dev_opened": False,
            "model_trained": True,
            "checkpoints_written": False,
            "c1_checkpoint_used": False,
            "c2_checkpoint_used": False,
            "c2r1_checkpoint_used": False,
            "c2r2_checkpoint_used": False,
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_cv(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
