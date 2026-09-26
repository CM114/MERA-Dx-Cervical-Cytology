"""Train S1 once at the locked S1-R epoch, then evaluate dev once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import sha256_file, validate_s0r_folds
from experiments.tbs.s1r_protocol import (
    LOCKED_S1R_CONFIG,
    LOCKED_S1R_RULE,
    select_s1r_epoch,
    validate_cv_provenance,
    validate_complete_s1r_cv_artifacts,
)
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import (
    reject_forbidden_data_path,
    validate_train_dev_paths,
    write_safety_artifacts,
)


FINAL_CHECKPOINT_SCHEMA = "xudata-tbs-s1r-final-checkpoint-v1"


def validate_s1r_final_training_contract(selected_epoch):
    selected_epoch = int(selected_epoch)
    if not 1 <= selected_epoch <= LOCKED_S1R_CONFIG["epochs"]:
        raise ValueError("selected_epoch must be within the completed S1-R epoch range")
    return {
        "epochs": selected_epoch,
        "scheduler_t_max": LOCKED_S1R_CONFIG["epochs"],
        "internal_validation_enabled": False,
        "checkpoint_selection_enabled": False,
        "dev_evaluation_count": 1,
        "objective": LOCKED_S1R_CONFIG["objective"],
    }


def validate_s1r_final_authorization(selector_json, s1r_cv_dir, s0r_cv_dir=None):
    selector_json, s1r_cv_dir = Path(selector_json), Path(s1r_cv_dir)
    decision = json.loads(selector_json.read_text(encoding="utf-8"))
    if decision.get("schema_version") != "xudata-tbs-s1r-epoch-selection-v1":
        raise ValueError("unexpected S1-R selector schema")
    if not decision.get("final_retrain_authorized") or decision.get("route") != "S1R_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED":
        raise ValueError("selector does not authorize final retraining")
    if s0r_cv_dir is None:
        raise ValueError("paired S0-R CV directory is required for authorization")
    s0r_cv_dir = Path(s0r_cv_dir)
    s1_summary_path = s1r_cv_dir / "cv_summary.json"
    s0_summary_path = s0r_cv_dir / "cv_summary.json"
    if not s1_summary_path.is_file() or not s0_summary_path.is_file():
        raise FileNotFoundError("paired S0-R and S1-R CV summaries are required")
    s1_summary = json.loads(s1_summary_path.read_text(encoding="utf-8"))
    s0_summary = json.loads(s0_summary_path.read_text(encoding="utf-8"))
    validate_cv_provenance(s1_summary, "S1R_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION")
    validate_cv_provenance(s0_summary, "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION")
    s1_completion = validate_complete_s1r_cv_artifacts(s1r_cv_dir)
    s0_completion = validate_complete_cv_artifacts(s0r_cv_dir)
    if s1_summary.get("history_sha256") != s1_completion["history_sha256"]:
        raise ValueError("S1-R history checksum mismatch")
    if s0_summary.get("history_sha256") != s0_completion["history_sha256"]:
        raise ValueError("S0-R history checksum mismatch")
    s1_histories = [s1r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    s0_histories = [s0r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    recomputed = select_s1r_epoch(s1_histories, s0_histories)
    if recomputed.get("route") != decision.get("route") or recomputed.get("selected_epoch") != decision.get("selected_epoch") or recomputed.get("selected_score") != decision.get("selected_score"):
        raise ValueError("selector does not match recomputed paired S0-R/S1-R decision")
    if decision.get("s1r_cv_summary_sha256") != sha256_file(s1_summary_path):
        raise ValueError("selector S1-R summary checksum mismatch")
    if decision.get("s0r_cv_summary_sha256") != sha256_file(s0_summary_path):
        raise ValueError("selector S0-R summary checksum mismatch")
    if decision.get("locked_training_config") != LOCKED_S1R_CONFIG:
        raise ValueError("selector locked training config does not match S1-R")
    if decision.get("locked_rule") != LOCKED_S1R_RULE:
        raise ValueError("selector locked rule does not match S1-R")
    epoch = decision.get("selected_epoch")
    validate_s1r_final_training_contract(epoch)
    return int(epoch)


def validate_pool_dev_independence(pool_csv, dev_csv):
    from experiments.train_tbs_singleview import validate_manifest_independence

    validate_manifest_independence(pool_csv, dev_csv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--s1r_cv_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
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
        for path in (args.fold_dir, args.s1r_cv_dir, args.s0r_cv_dir, args.out_dir):
            reject_forbidden_data_path(path)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite final output directory: {args.out_dir}")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _validate_final_inputs(args):
    cv_summary_path = args.s1r_cv_dir / "cv_summary.json"
    if not cv_summary_path.is_file():
        raise FileNotFoundError(cv_summary_path)
    selected_epoch = validate_s1r_final_authorization(args.selection_json, args.s1r_cv_dir, args.s0r_cv_dir)
    selector = json.loads(args.selection_json.read_text(encoding="utf-8"))
    completion = validate_complete_s1r_cv_artifacts(args.s1r_cv_dir)
    if selector.get("s1r_cv_history_sha256") != completion["history_sha256"]:
        raise ValueError("selector S1-R history checksum mismatch")
    fold_audit = validate_s0r_folds(args.train_fit_csv, args.select_s0_csv, args.fold_dir)
    cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    if cv_summary.get("fold_metadata_sha256") != fold_audit["fold_metadata_sha256"]:
        raise ValueError("S1-R fold metadata does not match final inputs")
    if cv_summary.get("pool_sha256") != fold_audit["pool_sha256"]:
        raise ValueError("S1-R pool does not match final inputs")
    sealed_sha = sha256_file(args.select_s1_csv)
    if cv_summary.get("select_s1_sha256") != sealed_sha:
        raise ValueError("sealed select_s1 checksum changed since S1-R CV")
    return selected_epoch, fold_audit, sealed_sha


def _write_dev_artifacts(out_dir, metrics, y_true, y_prob, paths, audit):
    from sklearn.metrics import classification_report, confusion_matrix
    from experiments.tbs.labels import DIAGNOSIS_NAMES

    out_dir = Path(out_dir)
    y_pred = y_prob.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": paths,
            "true_label": y_true,
            "pred_label": y_pred,
            "semantic_mask": audit["semantic_mask"].astype(int),
            "morph_true": audit["morph_true"],
            "morph_prob_high": audit["morph_prob_high"],
            "evidence_true": audit["evidence_true"],
            "evidence_prob_definitive": audit["evidence_prob_definitive"],
        }
    )
    for index, name in enumerate(DIAGNOSIS_NAMES):
        frame[f"prob_{name}"] = y_prob[:, index]
    frame.to_csv(out_dir / "dev_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(5))), index=DIAGNOSIS_NAMES, columns=DIAGNOSIS_NAMES).to_csv(out_dir / "dev_confusion_matrix.csv")
    pd.DataFrame(classification_report(y_true, y_pred, labels=list(range(5)), target_names=DIAGNOSIS_NAMES, output_dict=True, zero_division=0)).transpose().to_csv(out_dir / "dev_classification_report.csv")
    (out_dir / "dev_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_final(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.s1r_training import (
        build_model_optimizer_scheduler,
        evaluate_s1,
        make_loader,
        seed_everything,
        train_epoch,
    )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")
    pool_csv = args.fold_dir / "pool.csv"
    validate_pool_dev_independence(pool_csv, args.dev_csv)
    selected_epoch, fold_audit, sealed_sha = _validate_final_inputs(args)
    contract = validate_s1r_final_training_contract(selected_epoch)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_transform, eval_transform = build_transforms(LOCKED_S1R_CONFIG["img_size"], LOCKED_S1R_CONFIG["input_mode"])
    device = torch.device(args.device)
    seed_everything(LOCKED_S1R_CONFIG["seed"])
    train_loader = make_loader(XUDataTBS5Dataset(pool_csv, train_transform), True, args.num_workers, args.device, LOCKED_S1R_CONFIG["seed"])
    model, optimizer, scheduler, scaler = build_model_optimizer_scheduler(device)
    history = []
    for epoch in range(1, selected_epoch + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / "train_metrics.csv", index=False, lineterminator="\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        scheduler.step()
    selector_sha = sha256_file(args.selection_json)
    cv_summary_sha = sha256_file(args.s1r_cv_dir / "cv_summary.json")
    torch.save(
        {
            "schema_version": FINAL_CHECKPOINT_SCHEMA,
            "stage": "s1r",
            "model_name": LOCKED_S1R_CONFIG["model_name"],
            "epoch": selected_epoch,
            "model_state": model.state_dict(),
            "locked_training_config": LOCKED_S1R_CONFIG,
            "locked_rule": LOCKED_S1R_RULE,
            "selector_sha256": selector_sha,
            "cv_summary_sha256": cv_summary_sha,
            "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
            "pool_sha256": fold_audit["pool_sha256"],
            "train_fit_sha256": sha256_file(args.train_fit_csv),
            "select_s0_sha256": sha256_file(args.select_s0_csv),
            "select_s1_sha256": sealed_sha,
            "selection_role": "fivefold_paired_fixed_epoch",
        },
        args.out_dir / "final_model.pth",
    )
    dev_loader = make_loader(XUDataTBS5Dataset(args.dev_csv, eval_transform), False, args.num_workers, args.device, LOCKED_S1R_CONFIG["seed"])
    metrics, y_true, y_prob, paths, audit = evaluate_s1(model, dev_loader, device)
    _write_dev_artifacts(args.out_dir, metrics, y_true, y_prob, paths, audit)
    summary = {
        "schema_version": "xudata-tbs-s1r-final-v1",
        "route": "S1R_FINAL_DEV_COMPLETE_GATE_REQUIRED",
        "selected_epoch": selected_epoch,
        "training_contract": contract,
        "dev_evaluation_count": 1,
        "dev_metrics": metrics,
        "select_s1_access": "byte_hash_only_not_parsed",
        "select_s1_sha256": sealed_sha,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-s1r-final-safety-v1",
            "route": summary["route"],
            "model_trained": True,
            "internal_validation_used": False,
            "checkpoint_selection_used": False,
            "dev_evaluation_count": 1,
            "select_s1_access": "byte_hash_only_not_parsed",
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_final(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
