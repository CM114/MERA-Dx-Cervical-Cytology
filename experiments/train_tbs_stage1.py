import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.dataset import (  # noqa: E402
    INPUT_MODES,
    XUDataTBS5Dataset,
    build_transforms,
)
from experiments.tbs.losses import (  # noqa: E402
    compute_stage1_loss,
    pair_boundary_supcon_loss,
    validate_label_smoothing,
)
from experiments.tbs.metrics import (  # noqa: E402
    compute_stage1_metrics,
    save_evaluation_artifacts,
)
from experiments.tbs.pairboundary_evaluation import (  # noqa: E402
    save_pairboundary_evaluation,
)
from experiments.tbs.models import STAGE1_VARIANTS, build_stage1_model  # noqa: E402
from experiments.tbs.optimization import (  # noqa: E402
    build_discriminative_parameter_groups,
    get_named_learning_rates,
    validate_backbone_lr_multiplier,
)
from experiments.tbs.sampling import (  # noqa: E402
    build_case_class_balanced_weights,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")
    return config


def validate_stage1_objective_config(variant, lambda_screen):
    lambda_screen = float(lambda_screen)
    if not np.isfinite(lambda_screen):
        raise ValueError("lambda_screen must be finite")
    if variant == "m2" and lambda_screen != 0.0:
        raise ValueError("M2 lambda_screen must be 0.0")
    return lambda_screen


def validate_pairboundary_objective_config(
    variant, boundary_loss, temperature, lambda_pb
):
    boundary_loss = str(boundary_loss)
    temperature = float(temperature)
    lambda_pb = float(lambda_pb)
    if boundary_loss not in {"none", "pair_boundary_supcon"}:
        raise ValueError(f"Unknown boundary_loss: {boundary_loss}")
    if boundary_loss == "none":
        if not np.isfinite(lambda_pb) or lambda_pb != 0.0:
            raise ValueError("boundary_loss=none requires lambda_pb=0.0")
        return boundary_loss, temperature, lambda_pb
    if variant != "m0":
        raise ValueError("pair_boundary_supcon requires variant=m0")
    if not np.isfinite(temperature) or not np.isclose(
        temperature, 0.1, rtol=0.0, atol=1e-12
    ):
        raise ValueError("M0-PB1 temperature must be 0.1")
    if not np.isfinite(lambda_pb) or not np.isclose(
        lambda_pb, 0.1, rtol=0.0, atol=1e-12
    ):
        raise ValueError("M0-PB1 lambda_pb must be 0.1")
    return boundary_loss, temperature, lambda_pb


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, required=True)
    known, _ = pre_parser.parse_known_args()
    config = _load_config(known.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--variant", choices=STAGE1_VARIANTS, default=config.get("variant"))
    parser.add_argument("--experiment_name", default=config.get("experiment_name"))
    parser.add_argument("--model_name", default=config.get("model_name", "caformer_s18"))
    parser.add_argument("--train_csv", type=Path, default=Path(config["train_csv"]))
    parser.add_argument("--dev_csv", type=Path, default=Path(config["dev_csv"]))
    parser.add_argument("--out_dir", type=Path, default=Path(config["out_dir"]))
    parser.add_argument(
        "--init_backbone_checkpoint",
        type=Path,
        default=(
            Path(config["init_backbone_checkpoint"])
            if config.get("init_backbone_checkpoint")
            else None
        ),
        help="Optional auxiliary checkpoint; only compatible backbone.* weights are loaded.",
    )
    parser.add_argument("--img_size", type=int, default=int(config.get("img_size", 224)))
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default=config.get("input_mode", "crop"),
    )
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 30)))
    parser.add_argument("--batch_size", type=int, default=int(config.get("batch_size", 64)))
    parser.add_argument("--lr", type=float, default=float(config.get("lr", 1e-4)))
    parser.add_argument(
        "--backbone_lr_multiplier",
        type=float,
        default=config.get("backbone_lr_multiplier", 1.0),
    )
    parser.add_argument(
        "--weight_decay", type=float, default=float(config.get("weight_decay", 1e-4))
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=float(config.get("label_smoothing", 0.0)),
    )
    parser.add_argument("--num_workers", type=int, default=int(config.get("num_workers", 8)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 42)))
    parser.add_argument(
        "--sampling_strategy",
        choices=("uniform", "case_class_balanced"),
        default=config.get("sampling_strategy", "uniform"),
    )
    parser.add_argument(
        "--lambda_screen", type=float, default=float(config.get("lambda_screen", 0.3))
    )
    parser.add_argument(
        "--boundary_loss",
        choices=("none", "pair_boundary_supcon"),
        default=config.get("boundary_loss", "none"),
    )
    parser.add_argument(
        "--temperature", type=float, default=float(config.get("temperature", 0.1))
    )
    parser.add_argument(
        "--lambda_pb", type=float, default=float(config.get("lambda_pb", 0.0))
    )

    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=bool(config.get("pretrained", True)))

    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=bool(config.get("amp", True)))
    args = parser.parse_args()
    try:
        args.label_smoothing = validate_label_smoothing(args.label_smoothing)
        args.backbone_lr_multiplier = validate_backbone_lr_multiplier(
            args.backbone_lr_multiplier
        )
        args.lambda_screen = validate_stage1_objective_config(
            args.variant,
            args.lambda_screen,
        )
        (
            args.boundary_loss,
            args.temperature,
            args.lambda_pb,
        ) = validate_pairboundary_objective_config(
            args.variant,
            args.boundary_loss,
            args.temperature,
            args.lambda_pb,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _environment_payload(args):
    try:
        import timm

        timm_version = timm.__version__
    except Exception:
        timm_version = "unavailable"
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "timm": timm_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "train_manifest_sha256": file_sha256(args.train_csv),
        "dev_manifest_sha256": file_sha256(args.dev_csv),
        "code_sha256": {
            "experiments/train_tbs_stage1.py": file_sha256(Path(__file__)),
            "experiments/tbs/losses.py": file_sha256(
                Path(__file__).resolve().parent / "tbs" / "losses.py"
            ),
            "experiments/tbs/pairboundary_evaluation.py": file_sha256(
                Path(__file__).resolve().parent
                / "tbs"
                / "pairboundary_evaluation.py"
            ),
        },
        "init_backbone_checkpoint": (
            str(args.init_backbone_checkpoint.resolve(strict=True))
            if args.init_backbone_checkpoint is not None
            else None
        ),
        "init_backbone_checkpoint_sha256": (
            file_sha256(args.init_backbone_checkpoint)
            if args.init_backbone_checkpoint is not None
            else None
        ),
        "target_heldout_opened": False,
        "target_calibration_opened": False,
        "target_sealed_test_opened": False,
    }
    schema_path = args.train_csv.parent / "label_schema_tbs5.json"
    if schema_path.is_file():
        payload["label_schema_sha256"] = file_sha256(schema_path)
    if torch.cuda.is_available():
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch_arch_list": torch.cuda.get_arch_list(),
            }
        )
    return payload


def load_backbone_checkpoint(model, checkpoint_path):
    """Load only compatible ``backbone.*`` weights into a target Stage-1 model."""
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")

    target_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        key = str(key)
        if not key.startswith("backbone."):
            skipped.append(key)
            continue
        if key not in target_state or tuple(value.shape) != tuple(target_state[key].shape):
            skipped.append(key)
            continue
        compatible[key] = value
    if not compatible:
        raise RuntimeError(
            f"No compatible backbone weights found in checkpoint: {checkpoint_path}"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    missing_backbone = sorted(key for key in missing if key.startswith("backbone."))
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading backbone checkpoint: {unexpected}")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "loaded_keys": sorted(compatible),
        "skipped_keys": sorted(skipped),
        "missing_backbone_keys": missing_backbone,
        "head_keys_untouched": sorted(
            key for key in target_state if not key.startswith("backbone.")
        ),
        "policy": "backbone_only; target diagnosis/screen/abnormal heads remain freshly initialized",
    }


def _make_loader(dataset, args, shuffle):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = None
    if shuffle and args.sampling_strategy == "case_class_balanced":
        sampler = WeightedRandomSampler(
            weights=build_case_class_balanced_weights(dataset.records),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _accumulate_loss_totals(
    totals,
    loss_counts,
    losses,
    batch_size,
    loss_weights=None,
):
    loss_weights = loss_weights or {}
    unknown_weights = set(loss_weights) - set(losses)
    if unknown_weights:
        raise ValueError(f"Unknown loss weights: {sorted(unknown_weights)}")
    loss_keys = set(losses)
    if not totals:
        totals.update({name: 0.0 for name in losses})
        loss_counts.update({name: 0.0 for name in losses})
    elif set(totals) != loss_keys:
        raise ValueError("Loss keys changed between batches")
    elif set(loss_counts) != loss_keys:
        raise ValueError("Loss count keys changed between batches")
    for name, value in losses.items():
        weight = float(loss_weights.get(name, batch_size))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Loss weight for {name} must be finite and nonnegative")
        totals[name] += float(value.detach().item()) * weight
        loss_counts[name] += weight


def _average_loss_totals(totals, loss_counts):
    if not totals or set(totals) != set(loss_counts):
        raise ValueError("Loss totals and counts must have matching nonempty keys")
    averages = {}
    for name, total in totals.items():
        count = loss_counts[name]
        if count <= 0:
            if name in {
                "abnormal_cond_ce",
                "pb_low_loss",
                "pb_high_loss",
            } and total == 0.0:
                averages[name] = 0.0
                continue
            raise ValueError(f"Cannot average {name} over zero samples")
        averages[name] = total / count
    return averages


def _loss_weights_for_batch(losses, diagnosis_labels, boundary_stats=None):
    weights = {}
    if "abnormal_cond_ce" in losses:
        weights["abnormal_cond_ce"] = int((diagnosis_labels > 0).sum().item())
    if "pb_low_loss" in losses:
        if boundary_stats is None:
            raise ValueError("Boundary loss requires boundary anchor statistics")
        weights["pb_low_loss"] = int(boundary_stats["low_valid"])
        weights["pb_high_loss"] = int(boundary_stats["high_valid"])
    return weights or None


def _compute_training_objective(output, diagnosis_labels, screen_labels, args):
    base = compute_stage1_loss(
        output,
        diagnosis_labels,
        screen_labels,
        args.variant,
        args.lambda_screen,
        label_smoothing=args.label_smoothing,
    )
    if args.boundary_loss == "none":
        return base, {}
    pb = pair_boundary_supcon_loss(
        output["features"], diagnosis_labels, args.temperature
    )
    losses = dict(base)
    losses.update(
        {
            "ce_loss": base["diagnosis_loss"],
            "pb_loss": pb.loss,
            "pb_low_loss": pb.low_loss,
            "pb_high_loss": pb.high_loss,
            "loss": base["loss"] + args.lambda_pb * pb.loss,
        }
    )
    stats = {
        "low_candidates": pb.low_candidates,
        "low_valid": pb.low_valid,
        "high_candidates": pb.high_candidates,
        "high_valid": pb.high_valid,
    }
    return losses, stats


def _validate_finite_losses(losses):
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} is nonfinite")


def _validate_boundary_coverage(metrics, minimum=0.95):
    for key in ("pb_low_coverage", "pb_high_coverage"):
        if key not in metrics or float(metrics[key]) < minimum:
            raise RuntimeError(
                f"M0-PB1 anchor coverage failed: {key}={metrics.get(key)!r}, "
                f"required>={minimum}"
            )


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def prepare_stage1_output_directory(out_dir, experiment_name):
    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        if out_dir.is_symlink() or _is_reparse_point(out_dir) or not out_dir.is_dir():
            raise FileExistsError(
                f"refusing to overwrite unsafe Stage-1 output directory: {out_dir}"
            )
        existing_paths = list(out_dir.iterdir())
        if any(path.is_symlink() or _is_reparse_point(path) for path in existing_paths):
            raise FileExistsError(
                f"refusing to overwrite unsafe Stage-1 output directory: {out_dir}"
            )
        if {path.name for path in existing_paths} != {"console.log"}:
            raise FileExistsError(
                f"refusing to overwrite existing Stage-1 artifacts: {out_dir}"
            )
        console_log = out_dir / "console.log"
        if not console_log.is_file() or console_log.stat().st_size != 0:
            raise FileExistsError(
                f"refusing to overwrite existing Stage-1 artifacts: {out_dir}"
            )
    else:
        out_dir.mkdir(parents=True)
    marker = out_dir / ".stage1_training_owner.json"
    marker.write_text(
        json.dumps({"experiment_name": experiment_name}, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker


def _accumulate_boundary_totals(totals, stats):
    for key in ("low_candidates", "low_valid", "high_candidates", "high_valid"):
        totals[key] = totals.get(key, 0) + int(stats[key])


def _finalize_boundary_totals(totals):
    if not totals:
        return {}
    low_candidates = totals["low_candidates"]
    high_candidates = totals["high_candidates"]
    return {
        "pb_low_candidate_anchors": low_candidates,
        "pb_low_valid_anchors": totals["low_valid"],
        "pb_low_coverage": totals["low_valid"] / low_candidates
        if low_candidates else 0.0,
        "pb_high_candidate_anchors": high_candidates,
        "pb_high_valid_anchors": totals["high_valid"],
        "pb_high_coverage": totals["high_valid"] / high_candidates
        if high_candidates else 0.0,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    totals = {}
    loss_counts = {}
    boundary_totals = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        diagnosis_labels = batch["diagnosis_label"].to(device, non_blocking=True)
        screen_labels = batch["screen_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            losses, boundary_stats = _compute_training_objective(
                output, diagnosis_labels, screen_labels, args
            )

        _validate_finite_losses(losses)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        _accumulate_loss_totals(
            totals,
            loss_counts,
            losses,
            batch_size,
            loss_weights=_loss_weights_for_batch(
                losses, diagnosis_labels, boundary_stats
            ),
        )
        if boundary_stats:
            _accumulate_boundary_totals(boundary_totals, boundary_stats)

    averages = _average_loss_totals(totals, loss_counts)
    averages.update(_finalize_boundary_totals(boundary_totals))
    return averages


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {}
    loss_counts = {}
    boundary_totals = {}
    labels = []
    probabilities = []
    auxiliary_screen_probabilities = []
    image_paths = []
    maturity_labels = []
    maturity_names = []
    features = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        diagnosis_labels = batch["diagnosis_label"].to(device, non_blocking=True)
        screen_labels = batch["screen_label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            losses, boundary_stats = _compute_training_objective(
                output, diagnosis_labels, screen_labels, args
            )

        _validate_finite_losses(losses)
        batch_size = images.shape[0]
        _accumulate_loss_totals(
            totals,
            loss_counts,
            losses,
            batch_size,
            loss_weights=_loss_weights_for_batch(
                losses, diagnosis_labels, boundary_stats
            ),
        )
        if boundary_stats:
            _accumulate_boundary_totals(boundary_totals, boundary_stats)

        labels.append(diagnosis_labels.cpu().numpy())
        probabilities.append(output["diagnosis_probs"].cpu().numpy())
        if args.boundary_loss == "pair_boundary_supcon":
            features.append(output["features"].float().cpu().numpy())
        if args.variant == "m1":
            auxiliary_screen_probabilities.append(output["screen_probs"].cpu().numpy())
        image_paths.extend(batch["image_path"])
        maturity_labels.extend(batch["maturity_label"].cpu().numpy().tolist())
        maturity_names.extend(batch["maturity_name"])

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    auxiliary_screen_prob = (
        np.concatenate(auxiliary_screen_probabilities)
        if auxiliary_screen_probabilities
        else None
    )
    metrics = compute_stage1_metrics(y_true, y_prob, auxiliary_screen_prob)
    averages = _average_loss_totals(totals, loss_counts)
    averages.update(_finalize_boundary_totals(boundary_totals))
    metrics.update({f"val_{name}": value for name, value in averages.items()})
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "auxiliary_screen_prob": auxiliary_screen_prob,
        "image_paths": image_paths,
        "maturity_labels": maturity_labels,
        "maturity_names": maturity_names,
        "features": np.concatenate(features) if features else None,
    }


def _print_dataset_summary(name, csv_path):
    frame = pd.read_csv(csv_path)
    print(f"\n{name}: {len(frame)} images")
    print(
        frame.groupby(["diagnosis_label", "diagnosis_name"], sort=True)
        .size()
        .to_string()
    )


def main():
    args = parse_args()
    if args.variant is None:
        raise ValueError("variant must be defined in the config or CLI")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-1 training")

    seed_everything(args.seed)
    prepare_stage1_output_directory(args.out_dir, args.experiment_name)
    resolved_args = vars(args).copy()
    resolved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in resolved_args.items()
    }
    with (args.out_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_args, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    environment = _environment_payload(args)
    with (args.out_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(environment, indent=2, ensure_ascii=False))

    _print_dataset_summary("train", args.train_csv)
    _print_dataset_summary("dev", args.dev_csv)

    train_transform, eval_transform = build_transforms(
        args.img_size,
        input_mode=args.input_mode,
    )
    train_dataset = XUDataTBS5Dataset(args.train_csv, train_transform)
    dev_dataset = XUDataTBS5Dataset(args.dev_csv, eval_transform)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    dev_loader = _make_loader(dev_dataset, args, shuffle=False)

    device = torch.device("cuda:0")
    model = build_stage1_model(
        args.variant,
        model_name=args.model_name,
        pretrained=args.pretrained,
    ).to(device)
    transfer_summary = None
    if args.init_backbone_checkpoint is not None:
        transfer_summary = load_backbone_checkpoint(
            model, args.init_backbone_checkpoint
        )
        with (args.out_dir / "transfer_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(transfer_summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        environment["transfer_summary"] = transfer_summary
        with (args.out_dir / "environment.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(environment, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    parameter_groups = build_discriminative_parameter_groups(
        model,
        base_lr=args.lr,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = []
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        learning_rates = get_named_learning_rates(optimizer.param_groups)
        head_learning_rate = learning_rates["head"]
        backbone_learning_rate = learning_rates["backbone"]
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        if args.boundary_loss == "pair_boundary_supcon":
            _validate_boundary_coverage(train_metrics)
        evaluation = evaluate(model, dev_loader, device, args)
        dev_metrics = evaluation["metrics"]
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "lr": head_learning_rate,
            "head_lr": head_learning_rate,
            "backbone_lr": backbone_learning_rate,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **dev_metrics,
        }
        history.append(epoch_metrics)
        pd.DataFrame(history).to_csv(args.out_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"head_lr={head_learning_rate:.3e} | "
            f"backbone_lr={backbone_learning_rate:.3e} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={dev_metrics['val_loss']:.4f} | "
            f"macro_f1={dev_metrics['macro_f1']:.4f} | "
            f"balanced_acc={dev_metrics['balanced_accuracy']:.4f} | "
            f"macro_auc={dev_metrics['macro_auc']:.4f} | "
            f"screen_sens={dev_metrics['screen_sensitivity']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            checkpoint = {
                "epoch": epoch,
                "variant": args.variant,
                "model_name": args.model_name,
                "model_state": model.state_dict(),
                "metrics": dev_metrics,
                "args": resolved_args,
            }
            torch.save(checkpoint, args.out_dir / "best_model.pth")
            best_payload = {"epoch": epoch, **dev_metrics}
            with (args.out_dir / "best_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(best_payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            save_evaluation_artifacts(
                args.out_dir,
                "dev",
                evaluation["y_true"],
                evaluation["y_prob"],
                evaluation["image_paths"],
                evaluation["maturity_labels"],
                evaluation["maturity_names"],
                evaluation["auxiliary_screen_prob"],
            )
            if args.boundary_loss == "pair_boundary_supcon":
                boundary_summary = save_pairboundary_evaluation(
                    args.out_dir,
                    evaluation["y_true"],
                    evaluation["y_prob"],
                    evaluation["features"],
                )
                best_payload.update(boundary_summary)
                with (args.out_dir / "best_metrics.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(best_payload, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")

    torch.save(
        {
            "epoch": args.epochs,
            "variant": args.variant,
            "model_name": args.model_name,
            "model_state": model.state_dict(),
            "args": resolved_args,
        },
        args.out_dir / "last_model.pth",
    )
    print(f"\nBest dev Macro F1: {best_macro_f1:.6f}")
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
