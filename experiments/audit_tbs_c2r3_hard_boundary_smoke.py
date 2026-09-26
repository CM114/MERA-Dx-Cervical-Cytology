"""Run a short, non-promotable C2-R3 hard-boundary mechanism preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.c0_protocol import sha256_file
from experiments.tbs.c2r3_conditional_boundary_protocol import LOCKED_C2R3_CONFIG
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, validate_s0r_folds
from experiments.tbs.s1r3_protocol import validate_sealed_source_path
from experiments.train_tbs_c0r2_cv import _read_s0r_summary
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts


SMOKE_SCHEMA = "xudata-tbs-c2r3-hard-boundary-smoke-v2"
SMOKE_ROUTE = "C2R3_HARD_BOUNDARY_SMOKE_COMPLETE_NOT_FOR_SELECTION"


def _boundary_loss_scale(epoch, warmup_epochs):
    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / float(warmup_epochs))


def parse_args(argv=None, validate_paths=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--hard_example_fraction",
        type=float,
        default=1.0,
        help="hard fraction for both pair CE terms; 1.0 keeps pair CE on all samples",
    )
    parser.add_argument(
        "--boundary_hard_example_fraction",
        type=float,
        default=0.8,
        help="hard fraction for high-grade boundary only",
    )
    parser.add_argument(
        "--boundary_warmup_epochs",
        type=int,
        default=0,
        help="linearly ramp boundary loss weight over this many epochs; 0 disables warm-up",
    )
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
            raise FileExistsError(f"refusing to overwrite smoke output: {args.out_dir}")
        if not 0 <= args.fold < LOCKED_C2R3_CONFIG["fold_count"]:
            raise ValueError("fold must be in [0, 4]")
        if not 1 <= args.epochs <= 6:
            raise ValueError("smoke epochs must be in [1, 6]")
        if not 0.0 < args.hard_example_fraction <= 1.0:
            raise ValueError("hard_example_fraction must be within (0, 1]")
        if not 0.0 < args.boundary_hard_example_fraction <= 1.0:
            raise ValueError("boundary_hard_example_fraction must be within (0, 1]")
        if not 0 <= args.boundary_warmup_epochs <= args.epochs:
            raise ValueError("boundary_warmup_epochs must be within [0, epochs]")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        _read_s0r_summary(args.s0r_cv_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _baseline_abnormal(row):
    if "abnormal_macro_f1" in row.index:
        return float(row["abnormal_macro_f1"])
    components = ["asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1"]
    missing = [key for key in components if key not in row.index]
    if missing:
        return None
    return sum(float(row[key]) for key in components) / len(components)


def run_smoke(args):
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
        raise FileExistsError(f"refusing to overwrite smoke output: {args.out_dir}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")

    s0r_summary_path, s0r_summary = _read_s0r_summary(args.s0r_cv_dir)
    baseline_completion = validate_complete_cv_artifacts(args.s0r_cv_dir)
    if s0r_summary.get("history_sha256") != baseline_completion.get("history_sha256"):
        raise ValueError("S0-R history checksum mismatch")
    fold_audit = validate_s0r_folds(args.train_fit_csv, args.select_s0_csv, args.fold_dir)
    if s0r_summary.get("fold_metadata_sha256") != fold_audit["fold_metadata_sha256"]:
        raise ValueError("S0-R fold metadata does not match supplied folds")
    if s0r_summary.get("pool_sha256") != fold_audit["pool_sha256"]:
        raise ValueError("S0-R pool does not match supplied folds")
    if s0r_summary.get("select_s1_sha256") != sha256_file(args.select_s1_csv):
        raise ValueError("select_s1 checksum does not match S0-R baseline")

    baseline_path = args.s0r_cv_dir / f"fold_{args.fold}" / "metrics.csv"
    baseline = pd.read_csv(baseline_path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = build_transforms(
        LOCKED_C2R3_CONFIG["img_size"], LOCKED_C2R3_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    fold_seed = LOCKED_C2R3_CONFIG["seed"] + args.fold
    seed_everything(fold_seed)
    fold_output = args.out_dir / f"fold_{args.fold}"
    fold_output.mkdir()
    train_csv = args.fold_dir / f"fold_{args.fold}" / "train.csv"
    val_csv = args.fold_dir / f"fold_{args.fold}" / "val.csv"
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
        device, LOCKED_C2R3_CONFIG["epochs"]
    )
    initialize_prototypes(model, prototype_loader, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        boundary_loss_scale = _boundary_loss_scale(
            epoch, args.boundary_warmup_epochs
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            hard_example_fraction=args.hard_example_fraction,
            boundary_hard_example_fraction=args.boundary_hard_example_fraction,
            boundary_loss_scale=boundary_loss_scale,
        )
        evaluation = evaluate_c2(model, val_loader, device)
        candidate = {key: float(value) for key, value in evaluation["metrics"].items()}
        base_row = baseline.loc[baseline["epoch"] == epoch].iloc[0]
        row = {
            "fold": int(args.fold),
            "epoch": int(epoch),
            "train_boundary_loss_scale": float(boundary_loss_scale),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **candidate,
            "s0_macro_f1": float(base_row["macro_f1"]),
            "delta_macro_f1": candidate["macro_f1"] - float(base_row["macro_f1"]),
            "s0_low_grade_pair_macro_f1": float(base_row["low_grade_pair_macro_f1"]),
            "delta_low_grade_pair_macro_f1": candidate["low_grade_pair_macro_f1"] - float(base_row["low_grade_pair_macro_f1"]),
            "s0_high_grade_pair_macro_f1": float(base_row["high_grade_pair_macro_f1"]),
            "delta_high_grade_pair_macro_f1": candidate["high_grade_pair_macro_f1"] - float(base_row["high_grade_pair_macro_f1"]),
            "s0_screen_sensitivity": float(base_row["screen_sensitivity"]),
            "delta_screen_sensitivity": candidate["screen_sensitivity"] - float(base_row["screen_sensitivity"]),
            "s0_high_grade_undercall": float(base_row["asc_h_hsil_to_normal_lowgrade_rate"]),
            "delta_high_grade_undercall": candidate["asc_h_hsil_to_normal_lowgrade_rate"] - float(base_row["asc_h_hsil_to_normal_lowgrade_rate"]),
        }
        baseline_abnormal = _baseline_abnormal(base_row)
        if baseline_abnormal is not None and "abnormal_macro_f1" in candidate:
            row["s0_abnormal_macro_f1"] = baseline_abnormal
            row["delta_abnormal_macro_f1"] = candidate["abnormal_macro_f1"] - baseline_abnormal
        history.append(row)
        pd.DataFrame(history).to_csv(
            fold_output / "metrics.csv", index=False, lineterminator="\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        scheduler.step()
    del model, optimizer, scheduler, scaler, train_loader, prototype_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "schema_version": SMOKE_SCHEMA,
        "route": SMOKE_ROUTE,
        "not_for_selection": True,
        "fold": int(args.fold),
        "epochs": int(args.epochs),
        "hard_example_fraction": float(args.hard_example_fraction),
        "boundary_hard_example_fraction": float(args.boundary_hard_example_fraction),
        "boundary_warmup_epochs": int(args.boundary_warmup_epochs),
        "locked_training_config": LOCKED_C2R3_CONFIG,
        "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
        "pool_sha256": fold_audit["pool_sha256"],
        "s0r_cv_summary_sha256": sha256_file(s0r_summary_path),
        "select_s1_sha256": sha256_file(args.select_s1_csv),
        "dev_accessed": False,
        "test_accessed": False,
        "checkpoints_written": False,
        "c1_checkpoint_used": False,
        "c2_checkpoint_used": False,
        "c2r1_checkpoint_used": False,
        "c2r2_checkpoint_used": False,
    }
    (args.out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_smoke(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
