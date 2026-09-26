"""Train the C0-C2 xudata TBS factorized dual-space pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.xudata_gain_common import (
    compute_locked_candidate_metrics,
    make_internal_train_split,
    reject_forbidden_data_path,
    validate_train_dev_paths,
    write_safety_artifacts,
)


LOSS_NAMES = (
    "loss",
    "diagnosis_loss",
    "screen_loss",
    "morph_loss",
    "evidence_loss",
    "decorr_loss",
    "prototype_loss",
    "geometry_loss",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("c0", "c1", "c2"), required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--m0_checkpoint", type=Path)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--model_name", default="caformer_s18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--prototype_residual", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--select_fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda_screen", type=float, default=0.2)
    parser.add_argument("--lambda_morph", type=float, default=0.3)
    parser.add_argument("--lambda_evidence", type=float, default=0.3)
    parser.add_argument("--lambda_decorr", type=float, default=0.01)
    parser.add_argument("--lambda_prototype", type=float, default=0.1)
    parser.add_argument("--lambda_geometry", type=float, default=0.02)
    args = parser.parse_args(argv)

    try:
        validate_train_dev_paths(args.train_csv, args.dev_csv)
        reject_forbidden_data_path(args.out_dir)
        if args.stage == "c0":
            if args.m0_checkpoint is None:
                raise ValueError("c0 requires --m0_checkpoint")
            reject_forbidden_data_path(args.m0_checkpoint)
            if not args.m0_checkpoint.is_file():
                raise FileNotFoundError(
                    f"M0 checkpoint does not exist: {args.m0_checkpoint}"
                )
        else:
            if args.init_checkpoint is None:
                raise ValueError(f"{args.stage} requires --init_checkpoint")
            reject_forbidden_data_path(args.init_checkpoint)
            if not args.init_checkpoint.is_file():
                raise FileNotFoundError(
                    f"initial checkpoint does not exist: {args.init_checkpoint}"
                )
        if args.out_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {args.out_dir}"
            )
        if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if not 0.0 < args.select_fraction < 0.5:
            raise ValueError("select_fraction must be between 0 and 0.5")
        if args.semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        if not np.isfinite(args.lr) or args.lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not 0.0 < args.backbone_lr_multiplier <= 1.0:
            raise ValueError("backbone_lr_multiplier must be within (0, 1]")
        if not 0.0 <= args.prototype_residual <= 1.0:
            raise ValueError("prototype_residual must be within [0, 1]")
        for name in (
            "lambda_screen",
            "lambda_morph",
            "lambda_evidence",
            "lambda_decorr",
            "lambda_prototype",
            "lambda_geometry",
        ):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset, batch_size, num_workers, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _targets(batch, device):
    return {
        "diagnosis_labels": batch["diagnosis_label"].to(device, non_blocking=True),
        "screen_labels": batch["screen_label"].to(device, non_blocking=True),
        "morph_labels": batch["morph_label"].to(device, non_blocking=True),
        "evidence_labels": batch["evidence_label"].to(device, non_blocking=True),
        "semantic_mask": batch["semantic_mask"].to(device, non_blocking=True).bool(),
    }


def _loss_kwargs(args):
    return {
        "stage": args.stage,
        "lambda_screen": args.lambda_screen,
        "lambda_morph": args.lambda_morph,
        "lambda_evidence": args.lambda_evidence,
        "lambda_decorr": args.lambda_decorr,
        "lambda_prototype": args.lambda_prototype,
        "lambda_geometry": args.lambda_geometry,
    }


def _build_optimizer(model, args):
    import torch

    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not task_parameters:
        raise RuntimeError("optimizer requires both backbone and task parameters")
    return torch.optim.AdamW(
        [
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": args.lr * args.backbone_lr_multiplier,
            },
            {
                "name": "task",
                "params": task_parameters,
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
    )


def _load_checkpoint(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_stage_transition(target_stage, payload):
    expected_source = {"c1": "c0", "c2": "c1"}.get(target_stage)
    if expected_source is None:
        raise ValueError("stage transition validation applies only to c1 or c2")
    source_stage = payload.get("stage") if isinstance(payload, dict) else None
    if source_stage != expected_source:
        raise ValueError(
            f"{target_stage} requires a {expected_source} checkpoint; "
            f"checkpoint stage is {source_stage!r}"
        )


def _load_initial_weights(model, args, device):
    import torch

    from experiments.tbs.models import build_stage1_model

    if args.stage == "c0":
        m0 = build_stage1_model(
            "m0", model_name=args.model_name, pretrained=False
        ).to(device)
        payload = _load_checkpoint(args.m0_checkpoint, device)
        state = payload.get("model_state", payload.get("state_dict", payload))
        m0.load_state_dict(state, strict=True)
        model.backbone.load_state_dict(m0.backbone.state_dict(), strict=True)
        with torch.no_grad():
            model.c0_head.weight.zero_()
            model.c0_head.weight[:, : model.feature_dim].copy_(
                m0.diagnosis_head.weight
            )
            model.c0_head.bias.copy_(m0.diagnosis_head.bias)
        return

    payload = _load_checkpoint(args.init_checkpoint, device)
    _validate_stage_transition(args.stage, payload)
    state = payload.get("model_state", payload.get("state_dict", payload))
    if args.stage == "c1":
        transferable = {
            key: value
            for key, value in state.items()
            if key.startswith(("backbone.", "local_view.", "view_gate."))
        }
    else:
        transferable = {
            key: value
            for key, value in state.items()
            if not key.startswith(
                (
                    "morph_prototypes",
                    "evidence_prototypes",
                    "log_morph_temperature",
                    "log_evidence_temperature",
                )
            )
        }
    if not transferable:
        raise RuntimeError("checkpoint contains no transferable model parameters")
    result = model.load_state_dict(transferable, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    target_state = model.state_dict()
    expected_loaded = {
        key
        for key in target_state
        if (
            key.startswith(("backbone.", "local_view.", "view_gate."))
            if args.stage == "c1"
            else not key.startswith(
                (
                    "morph_prototypes",
                    "evidence_prototypes",
                    "log_morph_temperature",
                    "log_evidence_temperature",
                )
            )
        )
    }
    if set(transferable) != expected_loaded:
        missing = sorted(expected_loaded - set(transferable))
        extra = sorted(set(transferable) - expected_loaded)
        raise RuntimeError(
            "checkpoint transfer key mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def _initialize_c2_prototypes(model, loader, device):
    import torch

    model.eval()
    morph_features, morph_labels = [], []
    evidence_features, evidence_labels = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            mask = batch["semantic_mask"].bool()
            morph_features.append(output["morph_features"][mask.to(device)].cpu())
            evidence_features.append(output["evidence_features"][mask.to(device)].cpu())
            morph_labels.append(batch["morph_label"][mask])
            evidence_labels.append(batch["evidence_label"][mask])
    model.initialize_factor_prototypes(
        torch.cat(morph_features).to(device),
        torch.cat(morph_labels).to(device),
        torch.cat(evidence_features).to(device),
        torch.cat(evidence_labels).to(device),
    )


def _train_epoch(model, loader, optimizer, scaler, device, args):
    import torch
    from experiments.tbs.factorized_losses import factorized_tbs_loss

    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(images)
            losses = factorized_tbs_loss(
                output, targets, **_loss_kwargs(args)
            )
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(images)
        count += batch_size
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach()) * batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _evaluate(model, loader, device, args):
    import torch
    from experiments.tbs.factorized_losses import factorized_tbs_loss
    from experiments.tbs.metrics import compute_semantic_metrics

    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    labels, probabilities = [], []
    morph_true, morph_prob = [], []
    evidence_true, evidence_prob = [], []
    semantic_masks, image_paths = [], []
    maturity_labels, maturity_names = [], []
    theta_values, gate_values = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = _targets(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
                losses = factorized_tbs_loss(
                    output, targets, **_loss_kwargs(args)
                )
            batch_size = len(images)
            count += batch_size
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].detach()) * batch_size
            labels.append(targets["diagnosis_labels"].cpu().numpy())
            probabilities.append(output["diagnosis_probs"].cpu().numpy())
            semantic_masks.append(targets["semantic_mask"].cpu().numpy())
            image_paths.extend(batch["image_path"])
            maturity_labels.append(batch["maturity_label"].cpu().numpy())
            maturity_names.extend(batch["maturity_name"])
            theta_values.append(output["theta"].float().cpu().numpy())
            gate_values.append(output["view_gate"].float().cpu().numpy())
            if args.stage != "c0":
                morph_true.append(targets["morph_labels"].cpu().numpy())
                morph_prob.append(output["morph_probs"][:, 1].cpu().numpy())
                evidence_true.append(targets["evidence_labels"].cpu().numpy())
                evidence_prob.append(output["evidence_probs"][:, 1].cpu().numpy())

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    metrics = compute_locked_candidate_metrics(y_true, y_prob)
    if args.stage != "c0":
        metrics.update(
            compute_semantic_metrics(
                np.concatenate(morph_true),
                np.concatenate(morph_prob),
                np.concatenate(evidence_true),
                np.concatenate(evidence_prob),
                np.concatenate(semantic_masks).astype(bool),
                y_prob,
            )
        )
    theta = np.concatenate(theta_values)
    determinant = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
    metrics.update(
        {
            "loss": totals["loss"] / max(count, 1),
            "local_area_mean": float(np.abs(determinant).mean()),
            "local_translation_abs_mean": float(np.abs(theta[:, :, 2]).mean()),
            "view_gate_mean": float(np.concatenate(gate_values).mean()),
        }
    )
    audit = {
        "semantic_mask": np.concatenate(semantic_masks).astype(bool),
        "maturity_label": np.concatenate(maturity_labels),
        "maturity_name": maturity_names,
        "theta": theta,
        "view_gate": np.concatenate(gate_values).reshape(-1),
    }
    if args.stage != "c0":
        audit.update(
            {
                "morph_true": np.concatenate(morph_true),
                "morph_prob_high": np.concatenate(morph_prob),
                "evidence_true": np.concatenate(evidence_true),
                "evidence_prob_definitive": np.concatenate(evidence_prob),
            }
        )
    return metrics, y_true, y_prob, image_paths, audit


def _write_dev_artifacts(out_dir, metrics, y_true, y_prob, image_paths, audit, stage):
    from sklearn.metrics import classification_report, confusion_matrix
    from experiments.tbs.labels import DIAGNOSIS_NAMES

    y_pred = y_prob.argmax(axis=1)
    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true,
            "true_name": [DIAGNOSIS_NAMES[index] for index in y_true],
            "pred_label": y_pred,
            "pred_name": [DIAGNOSIS_NAMES[index] for index in y_pred],
            "maturity_label": audit["maturity_label"],
            "maturity_name": audit["maturity_name"],
            "semantic_mask": audit["semantic_mask"].astype(int),
            "screen_prob": y_prob[:, 1:].sum(axis=1),
            "view_gate": audit["view_gate"],
            "local_theta_00": audit["theta"][:, 0, 0],
            "local_theta_01": audit["theta"][:, 0, 1],
            "local_theta_02": audit["theta"][:, 0, 2],
            "local_theta_10": audit["theta"][:, 1, 0],
            "local_theta_11": audit["theta"][:, 1, 1],
            "local_theta_12": audit["theta"][:, 1, 2],
        }
    )
    for index, name in enumerate(DIAGNOSIS_NAMES):
        predictions[f"prob_{name}"] = y_prob[:, index]
    if stage != "c0":
        predictions["morph_true"] = audit["morph_true"]
        predictions["morph_prob_high"] = audit["morph_prob_high"]
        predictions["evidence_true"] = audit["evidence_true"]
        predictions["evidence_prob_definitive"] = audit[
            "evidence_prob_definitive"
        ]
    predictions.to_csv(
        out_dir / "dev_predictions.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(5))),
        index=DIAGNOSIS_NAMES,
        columns=DIAGNOSIS_NAMES,
    ).to_csv(out_dir / "dev_confusion_matrix.csv")
    pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(5)),
            target_names=DIAGNOSIS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(out_dir / "dev_classification_report.csv")
    (out_dir / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_training(args):
    import torch
    from experiments.tbs.dataset import XUDataTBS5Dataset, build_transforms
    from experiments.tbs.factorized_model import build_tbs_factorized_model

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU smoke test")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    fit_csv, select_csv = make_internal_train_split(
        args.train_csv,
        args.out_dir / "internal_split",
        select_fraction=args.select_fraction,
    )
    train_transform, eval_transform = build_transforms(
        args.img_size, input_mode="letterbox"
    )
    fit_loader = _loader(
        XUDataTBS5Dataset(fit_csv, train_transform),
        args.batch_size,
        args.num_workers,
        True,
        args.seed,
    )
    prototype_loader = None
    if args.stage == "c2":
        prototype_loader = _loader(
            XUDataTBS5Dataset(fit_csv, eval_transform),
            args.batch_size,
            args.num_workers,
            False,
            args.seed,
        )
    select_loader = _loader(
        XUDataTBS5Dataset(select_csv, eval_transform),
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
    )
    dev_loader = _loader(
        XUDataTBS5Dataset(args.dev_csv, eval_transform),
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
    )
    model = build_tbs_factorized_model(
        args.stage,
        model_name=args.model_name,
        pretrained=False,
        semantic_dim=args.semantic_dim,
        prototype_residual=args.prototype_residual,
    ).to(device)
    _load_initial_weights(model, args, device)
    if args.stage == "c2":
        _initialize_c2_prototypes(model, prototype_loader, device)

    optimizer = _build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    best_select = -1.0
    stale_epochs = 0
    best_path = args.out_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model, fit_loader, optimizer, scaler, device, args
        )
        select_metrics, _, _, _, _ = _evaluate(
            model, select_loader, device, args
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "task_lr": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"select_{key}": value for key, value in select_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            args.out_dir / "metrics.csv", index=False, lineterminator="\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if select_metrics["macro_f1"] > best_select:
            best_select = select_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "stage": args.stage,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "select_metrics": select_metrics,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    payload = _load_checkpoint(best_path, device)
    model.load_state_dict(payload["model_state"], strict=True)
    dev_metrics, y_true, y_prob, image_paths, audit = _evaluate(
        model, dev_loader, device, args
    )
    _write_dev_artifacts(
        args.out_dir,
        dev_metrics,
        y_true,
        y_prob,
        image_paths,
        audit,
        args.stage,
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-factorized-c0-c2-v1",
            "stage": args.stage,
            "route": "TRAIN_INTERNAL_SELECT_DEV_ONCE",
            "selection_split": "train_select",
            "dev_evaluation_count": 1,
            "model_trained": True,
            "external_data_used": False,
            "labels_are_ontology_derived": True,
        },
    )
    return dev_metrics


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_training(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
