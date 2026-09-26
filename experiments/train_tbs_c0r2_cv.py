"""Run the sealed five-fold C0-R2 boundary-stability control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.c0_protocol import sha256_file
from experiments.tbs.c0r2_protocol import (
    C0R2_CV_COMPLETE_ROUTE,
    C0R2_CV_SCHEMA,
    LOCKED_C0R2_CONFIG,
    validate_complete_c0r2_cv_artifacts,
)
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, validate_s0r_folds
from experiments.tbs.s1r3_protocol import validate_sealed_source_path
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import write_safety_artifacts


def _read_s0r_summary(s0r_cv_dir):
    summary_path = Path(s0r_cv_dir) / "cv_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "xudata-tbs-s0r-cv-v1":
        raise ValueError("S0-R summary schema is not locked")
    if payload.get("locked_training_config") != LOCKED_S0R_CONFIG:
        raise ValueError("S0-R training config does not match the locked baseline")
    if payload.get("route") != "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION":
        raise ValueError("S0-R CV is not complete")
    if payload.get("checkpoints_written") is not False:
        raise ValueError("S0-R summary permits checkpoints")
    if payload.get("dev_accessed") is not False:
        raise ValueError("S0-R summary indicates dev access")
    if payload.get("select_s1_access") != "byte_hash_only_not_parsed":
        raise ValueError("S0-R summary does not seal select_s1 access")
    return summary_path, payload


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
            raise FileExistsError(f"refusing to overwrite C0-R2 CV output: {args.out_dir}")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        _read_s0r_summary(args.s0r_cv_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _write_epoch12_predictions(path, evaluation):
    frame = pd.DataFrame({
        "label": evaluation["y_true"].astype(int),
        "pred": evaluation["y_prob"].argmax(axis=1).astype(int),
        "full_pred": evaluation["full_y_prob"].argmax(axis=1).astype(int),
    })
    for index in range(5):
        frame[f"p_{index}"] = evaluation["y_prob"][:, index]
        frame[f"full_p_{index}"] = evaluation["full_y_prob"][:, index]
    frame.to_csv(path, index=False, lineterminator="\n")


def run_cv(args):
    import torch

    from experiments.tbs.c0r2_training import (
        build_model_optimizer_scheduler, evaluate_c0r2, make_loader,
        seed_everything, train_epoch,
    )
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms

    for path in (
        args.train_fit_csv, args.select_s0_csv, args.select_s1_csv,
        args.fold_dir, args.s0r_cv_dir,
    ):
        validate_sealed_source_path(path, require_exists=True)
    validate_sealed_source_path(args.out_dir)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite C0-R2 CV output: {args.out_dir}")
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
        LOCKED_C0R2_CONFIG["img_size"], LOCKED_C0R2_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    for fold in range(LOCKED_C0R2_CONFIG["fold_count"]):
        fold_seed = LOCKED_C0R2_CONFIG["seed"] + fold
        seed_everything(fold_seed)
        fold_output = args.out_dir / f"fold_{fold}"
        fold_output.mkdir()
        train_csv = args.fold_dir / f"fold_{fold}" / "train.csv"
        val_csv = args.fold_dir / f"fold_{fold}" / "val.csv"
        train_loader = make_loader(XUDataTBS5Dataset(train_csv, train_transform), True, args.num_workers, args.device, fold_seed)
        val_loader = make_loader(XUDataTBS5Dataset(val_csv, eval_transform), False, args.num_workers, args.device, fold_seed)
        model, optimizer, scheduler, scaler = build_model_optimizer_scheduler(device, LOCKED_C0R2_CONFIG["epochs"])
        history = []
        for epoch in range(1, LOCKED_C0R2_CONFIG["epochs"] + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
            evaluation = evaluate_c0r2(model, val_loader, device)
            row = {
                "fold": int(fold), "epoch": int(epoch),
                **{f"train_{key}": float(value) for key, value in train_metrics.items()},
                **{key: float(value) for key, value in evaluation["metrics"].items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(fold_output / "metrics.csv", index=False, lineterminator="\n")
            if epoch == 12:
                _write_epoch12_predictions(fold_output / "epoch12_predictions.csv", evaluation)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = validate_complete_c0r2_cv_artifacts(args.out_dir)
    summary.update({
        "route": C0R2_CV_COMPLETE_ROUTE,
        "locked_training_config": LOCKED_C0R2_CONFIG,
        "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
        "pool_sha256": fold_audit["pool_sha256"],
        "s0r_cv_summary_sha256": sha256_file(s0r_summary_path),
        "select_s1_sha256": sealed_select_s1_sha256,
        "select_s1_access": "byte_hash_only_not_parsed",
        "dev_accessed": False,
    })
    (args.out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_safety_artifacts(
        args.out_dir, vars(args), {
            "schema_version": C0R2_CV_SCHEMA,
            "route": C0R2_CV_COMPLETE_ROUTE,
            "select_s1_access": "byte_hash_only_not_parsed",
            "dev_opened": False, "model_trained": True,
            "checkpoints_written": False,
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_cv(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
