"""Run locked five-fold S0-R CV without clean_v2 dev or model checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import (
    LOCKED_S0R_CONFIG,
    build_s0r_folds,
    sha256_file,
    validate_s0r_folds,
)
from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts


CV_OUTPUT_SCHEMA = "xudata-tbs-s0r-cv-v1"


def add_locked_pair_metric_aliases(metrics, y_true, y_prob):
    from experiments.xudata_gain_common import compute_locked_candidate_metrics

    metrics = dict(metrics)
    locked = compute_locked_candidate_metrics(y_true, y_prob)
    metrics["low_grade_pair_macro_f1"] = locked["low_grade_pair_macro_f1"]
    metrics["high_grade_pair_macro_f1"] = locked["high_grade_pair_macro_f1"]
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        for path in (args.train_fit_csv, args.select_s0_csv, args.select_s1_csv):
            reject_forbidden_data_path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (args.fold_dir, args.out_dir):
            reject_forbidden_data_path(path)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite CV output directory: {args.out_dir}")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def validate_complete_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    histories = {}
    forbidden = list(out_dir.rglob("*.pth")) + list(out_dir.rglob("*.pt"))
    if forbidden:
        raise ValueError("CV output must not contain model checkpoints")
    expected_epochs = list(range(1, LOCKED_S0R_CONFIG["epochs"] + 1))
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.get("epoch") is None or frame["epoch"].astype(int).tolist() != expected_epochs:
            raise ValueError(f"fold {fold} history is incomplete or unordered")
        histories[str(fold)] = sha256_file(path)
    return {
        "schema_version": CV_OUTPUT_SCHEMA,
        "fold_count": LOCKED_S0R_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_S0R_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


def run_cv(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.s0r_training import (
        build_model_optimizer_scheduler,
        evaluate_m0,
        make_loader,
        seed_everything,
        train_epoch,
    )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")
    if args.fold_dir.exists():
        fold_audit = validate_s0r_folds(
            args.train_fit_csv, args.select_s0_csv, args.fold_dir
        )
    else:
        build_s0r_folds(args.train_fit_csv, args.select_s0_csv, args.fold_dir)
        fold_audit = validate_s0r_folds(
            args.train_fit_csv, args.select_s0_csv, args.fold_dir
        )
    sealed_select_s1_sha256 = sha256_file(args.select_s1_csv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    _, eval_transform = build_transforms(
        LOCKED_S0R_CONFIG["img_size"], LOCKED_S0R_CONFIG["input_mode"]
    )
    train_transform, _ = build_transforms(
        LOCKED_S0R_CONFIG["img_size"], LOCKED_S0R_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        fold_seed = LOCKED_S0R_CONFIG["seed"] + fold
        seed_everything(fold_seed)
        fold_output = args.out_dir / f"fold_{fold}"
        fold_output.mkdir()
        train_csv = args.fold_dir / f"fold_{fold}" / "train.csv"
        val_csv = args.fold_dir / f"fold_{fold}" / "val.csv"
        train_loader = make_loader(
            XUDataTBS5Dataset(train_csv, train_transform), True,
            args.num_workers, args.device, fold_seed,
        )
        val_loader = make_loader(
            XUDataTBS5Dataset(val_csv, eval_transform), False,
            args.num_workers, args.device, fold_seed,
        )
        model, optimizer, scheduler, scaler = build_model_optimizer_scheduler(
            device, LOCKED_S0R_CONFIG["epochs"]
        )
        history = []
        for epoch in range(1, LOCKED_S0R_CONFIG["epochs"] + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
            evaluation = evaluate_m0(model, val_loader, device)
            validation_metrics = add_locked_pair_metric_aliases(
                evaluation["metrics"], evaluation["y_true"], evaluation["y_prob"]
            )
            row = {
                "fold": fold,
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **validation_metrics,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(
                fold_output / "metrics.csv", index=False, lineterminator="\n"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = validate_complete_cv_artifacts(args.out_dir)
    summary.update(
        {
            "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
            "locked_training_config": LOCKED_S0R_CONFIG,
            "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
            "pool_sha256": fold_audit["pool_sha256"],
            "select_s1_sha256": sealed_select_s1_sha256,
            "select_s1_access": "byte_hash_only_not_parsed",
            "dev_accessed": False,
        }
    )
    (args.out_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": CV_OUTPUT_SCHEMA,
            "route": summary["route"],
            "select_s1_access": "byte_hash_only_not_parsed",
            "dev_opened": False,
            "model_trained": True,
            "checkpoints_written": False,
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_cv(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
