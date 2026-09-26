"""Run a locked, five-fold XUData backbone comparison.

All ordinary backbones use the same direct five-class head, transforms,
optimizer, epochs, seed and fold manifests.  The optional proposed-method
directory is read as an already-produced XUData candidate and is never
retrained here. Values read from a comparison workbook are written as external
references and are explicitly marked as not directly comparable to this
benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import (
    LOCKED_S0R_CONFIG,
    sha256_file,
    validate_s0r_folds,
)
from experiments.tbs.s0r_training import make_loader, seed_everything


DEFAULT_MODELS = (
    "resnet50",
    "resnet101",
    "densenet201",
    "efficientnet_b3",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "maxvit_tiny_rw_224",
    "coatnet_0_rw_224",
    "regnety_016",
)
MODEL_CATEGORIES = {
    "resnet50": "CNN/Conv",
    "resnet101": "CNN/Conv",
    "densenet201": "CNN/Conv",
    "efficientnet_b3": "CNN/Conv",
    "convnext_tiny": "CNN/Conv",
    "swin_tiny_patch4_window7_224": "Transformer",
    "swin_small_patch4_window7_224": "Transformer",
    "vit_base_patch16_224": "Transformer",
    "deit_small_patch16_224": "Transformer",
    "maxvit_tiny_rw_224": "Hybrid/Attention",
    "coatnet_0_rw_224": "Hybrid/Attention",
    "regnety_016": "CNN/Conv",
    "ours_c2r3w": "Proposed",
}
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "densenet201": "DenseNet-201",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "swin_small_patch4_window7_224": "Swin-Small",
    "vit_base_patch16_224": "ViT-B/16",
    "deit_small_patch16_224": "DeiT-Small",
    "maxvit_tiny_rw_224": "MaxViT-Tiny",
    "coatnet_0_rw_224": "CoAtNet-0",
    "regnety_016": "RegNetY-0.16GF",
    "ours_c2r3w": "MERA-Dx (proposed)",
}
PAPER_MODEL_ORDER = DEFAULT_MODELS + ("ours_c2r3w",)
PAPER_METRICS = (
    "accuracy",
    "macro_f1",
    "macro_sensitivity",
    "macro_specificity",
    "macro_auc",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
)
PAPER_COLUMNS = ("category", "model", *PAPER_METRICS)
BENCHMARK_METRICS = (
    "accuracy",
    "macro_sensitivity",
    "macro_specificity",
    "macro_f1",
    "macro_auc",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
    "nll",
    "brier",
    "ece",
)
REFERENCE_COLUMNS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "macro_f1",
    "macro_auc",
)
BENCHMARK_SCHEMA = "xudata-backbone-benchmark-v2"


def _parse_model_list(value):
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [part.strip() for part in str(value).split(",")]
    values = [value for value in values if value]
    if not values:
        raise ValueError("at least one model name is required")
    if len(set(values)) != len(values):
        raise ValueError("model names must be unique")
    return tuple(values)


def parse_args(argv=None, validate_paths=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fit_csv", type=Path, required=True)
    parser.add_argument("--select_s0_csv", type=Path, required=True)
    parser.add_argument("--select_s1_csv", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated timm model identifiers",
    )
    parser.add_argument("--selection_epoch", type=int, default=30)
    parser.add_argument("--reference_xlsx", type=Path, default=None)
    parser.add_argument("--our_cv_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.models = _parse_model_list(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    if not validate_paths:
        return args
    try:
        for path in (
            args.train_fit_csv,
            args.select_s0_csv,
            args.select_s1_csv,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not args.fold_dir.is_dir():
            raise FileNotFoundError(args.fold_dir)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite benchmark output: {args.out_dir}")
        if args.reference_xlsx is not None and not args.reference_xlsx.is_file():
            raise FileNotFoundError(args.reference_xlsx)
        if args.our_cv_dir is not None and not args.our_cv_dir.is_dir():
            raise FileNotFoundError(args.our_cv_dir)
        if not 1 <= args.selection_epoch <= LOCKED_S0R_CONFIG["epochs"]:
            raise ValueError("selection_epoch must be within the locked 1..30 epoch range")
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _training_args(device):
    return SimpleNamespace(
        variant="m0",
        lambda_screen=0.0,
        label_smoothing=LOCKED_S0R_CONFIG["label_smoothing"],
        boundary_loss="none",
        temperature=0.1,
        lambda_pb=0.0,
        amp=str(device).startswith("cuda"),
    )


def _build_model_optimizer_scheduler(model_name, device, pretrained=True):
    import torch
    from experiments.tbs.models import build_stage1_model
    from experiments.tbs.optimization import build_discriminative_parameter_groups

    model = build_stage1_model(
        "m0",
        model_name=model_name,
        pretrained=pretrained,
    ).to(device)
    groups = build_discriminative_parameter_groups(
        model,
        base_lr=LOCKED_S0R_CONFIG["lr"],
        backbone_lr_multiplier=LOCKED_S0R_CONFIG["backbone_lr_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=LOCKED_S0R_CONFIG["lr"],
        weight_decay=LOCKED_S0R_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LOCKED_S0R_CONFIG["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    return model, optimizer, scheduler, scaler


def _build_preflight_model(model_name, pretrained):
    from experiments.tbs.models import build_stage1_model

    model = build_stage1_model("m0", model_name=model_name, pretrained=pretrained)
    params = sum(parameter.numel() for parameter in model.parameters())
    feature_dim = int(model.backbone.num_features)
    return model, params, feature_dim


def preflight_models(model_names, pretrained):
    registry = []
    failures = []
    for model_name in model_names:
        try:
            model, params, feature_dim = _build_preflight_model(model_name, pretrained)
            registry.append(
                {
                    "model": model_name,
                    "category": MODEL_CATEGORIES.get(model_name, "Uncategorized"),
                    "paper_label": MODEL_LABELS.get(model_name, model_name),
                    "status": "available",
                    "parameters": int(params),
                    "feature_dim": feature_dim,
                }
            )
            del model
        except Exception as exc:  # pragma: no cover - depends on server timm/HF state
            item = {"model": model_name, "status": "unavailable", "error": repr(exc)}
            registry.append(item)
            failures.append(item)
    if failures:
        names = ", ".join(item["model"] for item in failures)
        raise RuntimeError(
            f"model preflight failed for: {names}; inspect model_registry.json before training"
        )
    return registry


def read_reference_xlsx(path):
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("openpyxl is required for --reference_xlsx") from exc
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheet = workbook.active
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        model = row[1] if len(row) > 1 else None
        values = row[2:7] if len(row) >= 7 else ()
        if not model or len(values) != 5:
            continue
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        rows.append(
            {
                "model": str(model).strip(),
                "source": "reference_excel_not_directly_comparable",
                "accuracy": float(values[0]),
                "sensitivity": float(values[1]),
                "specificity": float(values[2]),
                "macro_f1": float(values[3]),
                "macro_auc": float(values[4]),
            }
        )
    if not rows:
        raise ValueError(f"no numeric model rows found in workbook: {path}")
    return pd.DataFrame(rows)


def summarize_metrics(frame, selection_epoch):
    frame = pd.DataFrame(frame).copy()
    selected = frame[frame["epoch"].astype(int) == int(selection_epoch)]
    rows = []
    for model, group in selected.groupby("model", sort=False):
        row = {
            "model": str(model),
            "source": "xudata_reproduced_fivefold",
            "selection_epoch": int(selection_epoch),
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in BENCHMARK_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_mean_only_table(comparison):
    """Build the paper-facing mean-only table from reproduced XUData rows."""

    frame = pd.DataFrame(comparison).copy()
    required = {"model", "source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"comparison is missing required columns: {sorted(missing)}")

    reproduced_sources = {"xudata_reproduced_fivefold", "xudata_ours_c2r3w"}
    rows = []
    for model_name in PAPER_MODEL_ORDER:
        matches = frame[
            (frame["model"].astype(str) == model_name)
            & frame["source"].isin(reproduced_sources)
        ]
        if matches.empty:
            continue
        if len(matches) != 1:
            raise ValueError(f"comparison must contain one row for model {model_name!r}")
        source_row = matches.iloc[0]
        row = {
            "category": MODEL_CATEGORIES[model_name],
            "model": MODEL_LABELS[model_name],
        }
        for metric in PAPER_METRICS:
            column = f"mean_{metric}"
            if column not in source_row.index:
                raise ValueError(f"comparison row for {model_name!r} lacks {column}")
            value = pd.to_numeric(source_row[column], errors="coerce")
            if not np.isfinite(value):
                raise ValueError(f"comparison row for {model_name!r} has invalid {column}")
            row[metric] = float(value)
        rows.append(row)
    if not rows:
        raise ValueError("comparison contains no reproduced XUData rows for the paper table")
    return pd.DataFrame(rows, columns=PAPER_COLUMNS)


def _latex_escape(value):
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_mean_only_exports(comparison, out_dir):
    """Write a single wide IEEE table with means only and return both paths."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = build_mean_only_table(comparison)
    csv_path = out_dir / "paper_table_mean.csv"
    tex_path = out_dir / "paper_table_mean.tex"
    paper.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.4f")

    headers = (
        "Category",
        "Model",
        "Acc.",
        "mF1",
        "mSen.",
        "mSpec.",
        "mAUROC",
        "Low-pair F1",
        "High-pair F1",
        "Screen Sen.",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean performance of representative backbones on the XUData TBS5 development cohort. All baselines use the matched 30-epoch protocol; values are means across five fixed development folds. Fold-wise uncertainty is retained in the accompanying audit files.}",
        r"\label{tab:backbone-comparison-mean}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in paper.itertuples(index=False):
        values = [row.category, row.model]
        values.extend(f"{float(getattr(row, metric)):.4f}" for metric in PAPER_METRICS)
        if row.model == MODEL_LABELS["ours_c2r3w"]:
            values = [rf"\textbf{{{_latex_escape(value)}}}" for value in values]
        else:
            values = [_latex_escape(value) for value in values]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, tex_path


def _read_our_cv(our_cv_dir, selection_epoch):
    rows = []
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        path = Path(our_cv_dir) / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        selected = frame[frame["epoch"].astype(int) == int(selection_epoch)]
        if len(selected) != 1:
            raise ValueError(f"our CV history must have one selection epoch row: {path}")
        row = selected.iloc[0].to_dict()
        row["model"] = "ours_c2r3w"
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(args):
    import torch

    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.train_tbs_stage1 import evaluate, train_one_epoch
    from experiments.train_tbs_s0r_cv import add_locked_pair_metric_aliases
    from experiments.xudata_gain_common import write_safety_artifacts

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")
    fold_audit = validate_s0r_folds(args.train_fit_csv, args.select_s0_csv, args.fold_dir)
    registry = preflight_models(args.models, pretrained=not args.no_pretrained)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.preflight_only:
        summary = {
            "schema_version": BENCHMARK_SCHEMA,
            "route": "BACKBONE_PREFLIGHT_COMPLETE_NOT_TRAINED",
            "models": list(args.models),
            "model_registry": registry,
            "pretrained": not args.no_pretrained,
            "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
            "pool_sha256": fold_audit["pool_sha256"],
            "checkpoints_written": False,
        }
        (args.out_dir / "benchmark_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary

    train_transform, eval_transform = build_transforms(
        LOCKED_S0R_CONFIG["img_size"], LOCKED_S0R_CONFIG["input_mode"]
    )
    device = torch.device(args.device)
    train_args = _training_args(device)
    all_rows = []
    for model_name in args.models:
        model_slug = model_name.replace("/", "__")
        model_output = args.out_dir / model_slug
        model_output.mkdir()
        registry_item = next(item for item in registry if item["model"] == model_name)
        for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
            fold_seed = LOCKED_S0R_CONFIG["seed"] + fold
            seed_everything(fold_seed)
            fold_output = model_output / f"fold_{fold}"
            fold_output.mkdir()
            train_csv = args.fold_dir / f"fold_{fold}" / "train.csv"
            val_csv = args.fold_dir / f"fold_{fold}" / "val.csv"
            train_loader = make_loader(
                XUDataTBS5Dataset(train_csv, train_transform),
                True,
                args.num_workers,
                args.device,
                fold_seed,
            )
            val_loader = make_loader(
                XUDataTBS5Dataset(val_csv, eval_transform),
                False,
                args.num_workers,
                args.device,
                fold_seed,
            )
            model, optimizer, scheduler, scaler = _build_model_optimizer_scheduler(
                model_name, device, pretrained=not args.no_pretrained
            )
            history = []
            for epoch in range(1, LOCKED_S0R_CONFIG["epochs"] + 1):
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, scaler, device, train_args
                )
                evaluation = evaluate(model, val_loader, device, train_args)
                metrics = add_locked_pair_metric_aliases(
                    evaluation["metrics"], evaluation["y_true"], evaluation["y_prob"]
                )
                row = {
                    "model": model_name,
                    "model_slug": model_slug,
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "parameters": int(registry_item["parameters"]),
                    **{f"train_{key}": float(value) for key, value in train_metrics.items()},
                    **{key: float(value) for key, value in metrics.items()},
                }
                history.append(row)
                all_rows.append(row)
                pd.DataFrame(history).to_csv(
                    fold_output / "metrics.csv", index=False, lineterminator="\n"
                )
                print(json.dumps(row, ensure_ascii=False), flush=True)
                scheduler.step()
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_metrics = pd.DataFrame(all_rows)
    fold_metrics.to_csv(args.out_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    summary = summarize_metrics(fold_metrics, args.selection_epoch)
    summary.to_csv(args.out_dir / "summary.csv", index=False, lineterminator="\n")

    comparison_parts = [summary]
    if args.our_cv_dir is not None:
        ours = summarize_metrics(
            _read_our_cv(args.our_cv_dir, args.selection_epoch), args.selection_epoch
        )
        ours["source"] = "xudata_ours_c2r3w"
        comparison_parts.append(ours)
    if args.reference_xlsx is not None:
        comparison_parts.append(read_reference_xlsx(args.reference_xlsx))
    comparison = pd.concat(comparison_parts, ignore_index=True, sort=False)
    comparison.to_csv(
        args.out_dir / "comparison_table.csv", index=False, lineterminator="\n"
    )
    paper_csv, paper_tex = write_mean_only_exports(comparison, args.out_dir)

    run_summary = {
        "schema_version": BENCHMARK_SCHEMA,
        "route": "BACKBONE_BENCHMARK_COMPLETE_NOT_FOR_SELECTION",
        "models": list(args.models),
        "selection_epoch": int(args.selection_epoch),
        "locked_training_config": {
            **LOCKED_S0R_CONFIG,
            "head": "stage1_m0_shared_five_class_head",
            "pretrained": not args.no_pretrained,
            "metric_protocol": list(BENCHMARK_METRICS),
        },
        "fold_metadata_sha256": fold_audit["fold_metadata_sha256"],
        "pool_sha256": fold_audit["pool_sha256"],
        "select_s1_sha256": sha256_file(args.select_s1_csv),
        "reference_xlsx": str(args.reference_xlsx) if args.reference_xlsx else None,
        "our_cv_dir": str(args.our_cv_dir) if args.our_cv_dir else None,
        "paper_exports": {
            "mean_csv": paper_csv.name,
            "mean_tex": paper_tex.name,
        },
        "dev_accessed": False,
        "test_accessed": False,
        "checkpoints_written": False,
    }
    (args.out_dir / "benchmark_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": BENCHMARK_SCHEMA,
            "route": run_summary["route"],
            "model_trained": True,
            "dev_opened": False,
            "select_s1_access": "byte_hash_only_not_parsed",
            "checkpoints_written": False,
        },
    )
    return run_summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_benchmark(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
