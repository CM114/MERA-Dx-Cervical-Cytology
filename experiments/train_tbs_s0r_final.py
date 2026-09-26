"""Retrain S0-R at one fixed CV-selected epoch and evaluate dev exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import (
    LOCKED_EPOCH_RULE,
    LOCKED_S0R_CONFIG,
    sha256_file,
    validate_s0r_folds,
)
from experiments.xudata_gain_common import (
    reject_forbidden_data_path,
    validate_train_dev_paths,
    write_safety_artifacts,
)


FINAL_CHECKPOINT_SCHEMA = "xudata-tbs-s0r-final-checkpoint-v1"


def validate_final_training_contract(selected_epoch):
    selected_epoch = int(selected_epoch)
    if not 1 <= selected_epoch <= LOCKED_S0R_CONFIG["epochs"]:
        raise ValueError("selected_epoch must be within the completed CV epoch range")
    return {
        "epochs": selected_epoch,
        "scheduler_t_max": LOCKED_S0R_CONFIG["epochs"],
        "internal_validation_enabled": False,
        "checkpoint_selection_enabled": False,
        "dev_evaluation_count": 1,
        "objective": LOCKED_S0R_CONFIG["objective"],
    }


def validate_final_authorization(selector_json, cv_summary_json):
    selector_json, cv_summary_json = Path(selector_json), Path(cv_summary_json)
    decision = json.loads(selector_json.read_text(encoding="utf-8"))
    if decision.get("schema_version") != "xudata-tbs-s0r-epoch-selection-v1":
        raise ValueError("unexpected S0-R selector schema")
    if not decision.get("final_retrain_authorized") or decision.get("route") != "FINAL_RETRAIN_AUTHORIZED":
        raise ValueError("selector does not authorize final retraining")
    if decision.get("locked_training_config") != LOCKED_S0R_CONFIG:
        raise ValueError("selector locked training config does not match S0-R")
    if decision.get("locked_epoch_rule") != LOCKED_EPOCH_RULE:
        raise ValueError("selector locked epoch rule does not match S0-R")
    if decision.get("cv_summary_sha256") != sha256_file(cv_summary_json):
        raise ValueError("selector CV summary checksum mismatch")
    epoch = decision.get("selected_epoch")
    validate_final_training_contract(epoch)
    return int(epoch)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--cv_dir", type=Path, required=True)
    parser.add_argument("--selection_json", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        validate_train_dev_paths(args.train_fit_csv, args.dev_csv)
        for path in (
            args.train_fit_csv, args.select_s0_csv, args.select_s1_csv,
            args.selection_json, args.dev_csv,
        ):
            reject_forbidden_data_path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (args.fold_dir, args.cv_dir, args.out_dir):
            reject_forbidden_data_path(path)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite final output directory: {args.out_dir}")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _validate_final_inputs(args):
    cv_summary_path = args.cv_dir / "cv_summary.json"
    selected_epoch = validate_final_authorization(args.selection_json, cv_summary_path)
    selector = json.loads(args.selection_json.read_text(encoding="utf-8"))
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        history = args.cv_dir / f"fold_{fold}" / "metrics.csv"
        if selector.get("cv_history_sha256", {}).get(str(fold)) != sha256_file(history):
            raise ValueError(f"selector fold {fold} history checksum mismatch")
    fold_audit = validate_s0r_folds(
        args.train_fit_csv, args.select_s0_csv, args.fold_dir
    )
    cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    if cv_summary.get("fold_metadata_sha256") != fold_audit["fold_metadata_sha256"]:
        raise ValueError("CV fold metadata does not match final S0-R inputs")
    if cv_summary.get("pool_sha256") != fold_audit["pool_sha256"]:
        raise ValueError("CV pool does not match final S0-R inputs")
    sealed_sha = sha256_file(args.select_s1_csv)
    if cv_summary.get("select_s1_sha256") != sealed_sha:
        raise ValueError("sealed select_s1 checksum changed since CV")
    return selected_epoch, fold_audit, sealed_sha


def validate_pool_dev_independence(pool_csv, dev_csv):
    """Parse dev only after final training, before its one authorized evaluation."""
    from experiments.train_tbs_singleview import validate_manifest_independence

    validate_manifest_independence(pool_csv, dev_csv)


def run_final(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.metrics import save_evaluation_artifacts
    from experiments.tbs.s0r_training import (
        build_model_optimizer_scheduler,
        evaluate_m0,
        make_loader,
        seed_everything,
        train_epoch,
    )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")
    selected_epoch, fold_audit, sealed_sha = _validate_final_inputs(args)
    contract = validate_final_training_contract(selected_epoch)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = build_transforms(
        LOCKED_S0R_CONFIG["img_size"], LOCKED_S0R_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    seed_everything(LOCKED_S0R_CONFIG["seed"])
    pool_csv = args.fold_dir / "pool.csv"
    train_loader = make_loader(
        XUDataTBS5Dataset(pool_csv, train_transform), True,
        args.num_workers, args.device, LOCKED_S0R_CONFIG["seed"],
    )
    model, optimizer, scheduler, scaler = build_model_optimizer_scheduler(
        device, LOCKED_S0R_CONFIG["epochs"]
    )
    history = []
    for epoch in range(1, selected_epoch + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(
            args.out_dir / "train_metrics.csv", index=False, lineterminator="\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        scheduler.step()

    selector_sha = sha256_file(args.selection_json)
    cv_summary_sha = sha256_file(args.cv_dir / "cv_summary.json")
    torch.save(
        {
            "schema_version": FINAL_CHECKPOINT_SCHEMA,
            "stage": "s0r",
            "model_name": LOCKED_S0R_CONFIG["model_name"],
            "epoch": selected_epoch,
            "model_state": model.state_dict(),
            "locked_training_config": LOCKED_S0R_CONFIG,
            "selector_sha256": selector_sha,
            "cv_summary_sha256": cv_summary_sha,
            "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
            "pool_sha256": fold_audit["pool_sha256"],
            "train_fit_sha256": sha256_file(args.train_fit_csv),
            "select_s0_sha256": sha256_file(args.select_s0_csv),
            "select_s1_sha256": sealed_sha,
            "selection_role": "fivefold_fixed_epoch",
        },
        args.out_dir / "final_model.pth",
    )

    # Dev is parsed and instantiated only after fixed-epoch training and final checkpoint save.
    validate_pool_dev_independence(pool_csv, args.dev_csv)
    dev_loader = make_loader(
        XUDataTBS5Dataset(args.dev_csv, eval_transform), False,
        args.num_workers, args.device, LOCKED_S0R_CONFIG["seed"],
    )
    evaluation = evaluate_m0(model, dev_loader, device)
    metrics = evaluation["metrics"]
    save_evaluation_artifacts(
        args.out_dir,
        "dev",
        evaluation["y_true"],
        evaluation["y_prob"],
        evaluation["image_paths"],
        evaluation["maturity_labels"],
        evaluation["maturity_names"],
    )
    summary = {
        "schema_version": "xudata-tbs-s0r-final-v1",
        "route": "S0R_FINAL_DEV_COMPLETE_GATE_REQUIRED",
        "selected_epoch": selected_epoch,
        "training_contract": contract,
        "dev_evaluation_count": 1,
        "dev_metrics": metrics,
        "select_s1_access": "byte_hash_only_not_parsed",
        "select_s1_sha256": sealed_sha,
        "s1_authorized": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-s0r-final-safety-v1",
            "route": summary["route"],
            "model_trained": True,
            "internal_validation_used": False,
            "checkpoint_selection_used": False,
            "dev_evaluation_count": 1,
            "select_s1_access": "byte_hash_only_not_parsed",
            "s1_authorized": False,
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_final(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
