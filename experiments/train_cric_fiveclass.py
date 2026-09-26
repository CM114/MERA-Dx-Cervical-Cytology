"""Train the CRIC five-class external benchmark.

All models consume the same slide-grouped fold manifests. ``mera_dx`` is the
paper-facing name of the proposed Swin-Tiny factorized model; it is not a
separate class label and it is evaluated with the same five-class metrics as
the direct baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CLASS_NAMES = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
MODEL_ORDER = (
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
    "mera_dx",
)
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
    "mera_dx": "MERA-Dx (proposed)",
}
MODEL_CATEGORIES = {
    **{name: "CNN/Conv" for name in (
        "resnet50", "resnet101", "densenet201", "efficientnet_b3", "convnext_tiny", "regnety_016"
    )},
    **{name: "Transformer" for name in (
        "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "vit_base_patch16_224", "deit_small_patch16_224"
    )},
    "maxvit_tiny_rw_224": "Hybrid/Attention",
    "coatnet_0_rw_224": "Hybrid/Attention",
    "mera_dx": "Proposed",
}


class CRICCellDataset:
    """Small Dataset wrapper kept local to avoid changing the XUData schema."""

    def __new__(cls, frame, transform=None):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, records, item_transform):
                self.records = records.reset_index(drop=True).to_dict("records")
                self.transform = item_transform

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                row = self.records[index]
                path = Path(row["image_path"])
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                except Exception as exc:
                    raise RuntimeError(f"failed to load CRIC patch: {path}") from exc
                if self.transform is not None:
                    image = self.transform(image)
                return {
                    "image": image,
                    "label": int(row["label"]),
                    "image_id": int(row["image_id"]),
                    "cell_key": str(row["cell_key"]),
                    "image_path": str(path),
                }

        return _Dataset(frame, transform)


def build_transforms(img_size=224):
    from torchvision import transforms

    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train, evaluate


def resolve_manifest_paths(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Resolve portable patch paths against the local copy of ``data_dir``."""
    result = frame.copy()
    resolved = []
    for value in result["image_path"].astype(str):
        candidate = Path(value)
        if candidate.is_file():
            resolved.append(str(candidate))
            continue
        candidate = data_dir / value
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resolved.append(str(candidate.resolve()))
    result["image_path"] = resolved
    return result


class DirectFiveClassModel:
    """Adapter around any timm feature backbone with a five-class head."""

    def __new__(cls, model_name, pretrained=False):
        import torch.nn as nn

        import timm

        backbone = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = int(getattr(backbone, "num_features"))

        class _Model(nn.Module):
            def __init__(self, feature_backbone, dim):
                super().__init__()
                self.backbone = feature_backbone
                self.feature_dim = int(dim)
                self.head = nn.Linear(self.feature_dim, 5)

            def forward(self, images):
                features = self.backbone(images)
                if features.ndim > 2:
                    features = features.flatten(1)
                return self.head(features.float())

        return _Model(backbone, feature_dim)


def build_direct_model(model_name, pretrained=False):
    """Build a direct five-class model; kept as a testable public helper."""
    return DirectFiveClassModel(model_name, pretrained=pretrained)


def build_model(model_name, pretrained=False):
    if model_name != "mera_dx":
        return build_direct_model(model_name, pretrained=pretrained)
    from experiments.tbs.tbs_fv_factorized_model import build_tbs_dualprototype_model

    return build_tbs_dualprototype_model(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        semantic_dim=128,
        residual_logit_bound=0.10,
        prototype_temperature=10.0,
        prototype_residual_bound=0.10,
    )


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != 5:
        raise ValueError("y_prob must have shape [n, 5]")
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(5))
    sensitivity = []
    specificity = []
    for index in range(5):
        tp = float(cm[index, index])
        fn = float(cm[index, :].sum() - tp)
        fp = float(cm[:, index].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        sensitivity.append(tp / (tp + fn) if tp + fn else float("nan"))
        specificity.append(tn / (tn + fp) if tn + fp else float("nan"))
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=np.arange(5),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(5), average="macro", zero_division=0)),
        "macro_sensitivity": float(np.nanmean(sensitivity)),
        "macro_specificity": float(np.nanmean(specificity)),
        "macro_auc": auc,
        "n_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }


def _factor_targets(labels):
    import torch

    semantic = labels > 0
    morph = ((labels == 3) | (labels == 4)).long()
    evidence = ((labels == 2) | (labels == 4)).long()
    return semantic, morph, evidence


def _loss_for_output(model_name, output, labels, objective="full"):
    import torch.nn.functional as F

    if objective not in {"full", "direct_ce"}:
        raise ValueError(f"unknown objective: {objective}")
    if model_name != "mera_dx":
        return F.cross_entropy(output, labels), {"loss": float(F.cross_entropy(output, labels).detach())}
    if objective == "direct_ce":
        diagnosis_loss = F.nll_loss(output["diagnosis_log_probs"], labels)
        return diagnosis_loss, {"diagnosis_loss": float(diagnosis_loss.detach())}
    from experiments.tbs.tbs_fv_factorized_loss import factorized_tbs_loss

    losses = factorized_tbs_loss(
        output,
        labels,
        lambda_screen=0.20,
        lambda_morph=0.30,
        lambda_evidence=0.30,
        lambda_decorr=0.01,
        lambda_prototype=0.05,
        lambda_base_anchor=0.25,
        lambda_residual=0.01,
    )
    return losses["loss"], {key: float(value.detach()) for key, value in losses.items() if key != "loss"}


def _initialize_meradx_prototypes(model, loader, device):
    import torch

    model.eval()
    morph_features, morph_labels = [], []
    evidence_features, evidence_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).long()
            output = model(images)
            semantic, morph, evidence = _factor_targets(labels)
            if bool(semantic.any()):
                morph_features.append(output["morph_features"][semantic].detach())
                morph_labels.append(morph[semantic].detach())
                evidence_features.append(output["evidence_features"][semantic].detach())
                evidence_labels.append(evidence[semantic].detach())
    if not morph_features:
        raise RuntimeError("MERA-Dx prototype initialization found no abnormal cells")
    model.initialize_prototypes_from_train_features(
        torch.cat(morph_features),
        torch.cat(morph_labels),
        torch.cat(evidence_features),
        torch.cat(evidence_labels),
    )


def _make_loader(frame, transform, batch_size, shuffle, num_workers):
    import torch
    from torch.utils.data import DataLoader

    dataset = CRICCellDataset(frame, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def _train_epoch(model_name, model, loader, optimizer, scaler, device, objective="full"):
    import torch

    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(images)
            loss, _ = _loss_for_output(
                model_name, output, labels, objective=objective
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


def _evaluate(model_name, model, loader, device):
    import torch

    model.eval()
    labels, probabilities, image_ids, cell_keys, paths = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
                if model_name == "mera_dx":
                    probs = output["diagnosis_probs"]
                else:
                    probs = torch.softmax(output.float(), dim=1)
            labels.append(batch["label"].numpy())
            probabilities.append(probs.float().cpu().numpy())
            image_ids.extend(batch["image_id"].numpy().tolist())
            cell_keys.extend(batch["cell_key"])
            paths.extend(batch["image_path"])
    if not labels:
        raise RuntimeError("evaluation loader is empty")
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    return compute_metrics(y_true, y_prob), y_true, y_prob, image_ids, cell_keys, paths


def _write_predictions(path, y_true, y_prob, image_ids, cell_keys, image_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "cell_key": cell_keys,
            "image_path": image_paths,
            "true_label": y_true,
            "pred_label": y_prob.argmax(axis=1),
        }
    )
    for index in range(5):
        frame[f"prob_{index}"] = y_prob[:, index]
    frame.to_csv(path, index=False, lineterminator="\n")


def run(args):
    import torch

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    metadata_path = data_dir / "preparation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    prep_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if prep_metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("CRIC preparation does not use the expected five-class mapping")
    requested = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = set(requested) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    folds = tuple(int(value.strip()) for value in str(args.folds).split(",") if value.strip())
    if not folds or any(fold < 0 or fold >= args.n_splits for fold in folds):
        raise ValueError(f"folds must be a comma-separated subset of 0..{args.n_splits - 1}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    seed_everything(args.seed)
    train_transform, eval_transform = build_transforms(args.img_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = []
    fold_rows = []

    for model_name in requested:
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_registry_entry = {
            "model": model_name,
            "paper_label": MODEL_LABELS[model_name],
            "category": MODEL_CATEGORIES[model_name],
            "pretrained": bool(args.pretrained),
        }
        for fold in folds:
            fold_path = data_dir / f"fold_{fold}.csv"
            frame = pd.read_csv(fold_path)
            frame = resolve_manifest_paths(frame, data_dir)
            train_frame = frame.loc[frame["split"] == "train"].copy()
            val_frame = frame.loc[frame["split"] == "val"].copy()
            train_ids = set(train_frame["image_id"].astype(int))
            val_ids = set(val_frame["image_id"].astype(int))
            if train_ids & val_ids:
                raise ValueError(f"slide leakage detected in fold {fold}")

            train_loader = _make_loader(
                train_frame, train_transform, args.batch_size, True, args.num_workers
            )
            eval_train_loader = _make_loader(
                train_frame, eval_transform, args.batch_size, False, args.num_workers
            )
            val_loader = _make_loader(
                val_frame, eval_transform, args.batch_size, False, args.num_workers
            )
            model = build_model(model_name, pretrained=args.pretrained).to(device)
            if model_name == "mera_dx":
                _initialize_meradx_prototypes(model, eval_train_loader, device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model_registry_entry["parameters"] = int(parameter_count)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            best_score = -float("inf")
            best_path = model_dir / f"fold_{fold}_best.pt"
            history = []
            for epoch in range(1, args.epochs + 1):
                train_loss = _train_epoch(
                    model_name,
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    objective=args.objective,
                )
                val_metrics, *_ = _evaluate(model_name, model, val_loader, device)
                scheduler.step()
                row = {
                    "epoch": epoch,
                    "fold": fold,
                    "model": model_name,
                    "train_loss": train_loss,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                }
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if row["val_macro_f1"] > best_score:
                    best_score = row["val_macro_f1"]
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "epoch": epoch,
                            "model": model_name,
                            "fold": fold,
                            "seed": args.seed,
                        },
                        best_path,
                    )

            payload = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(payload["model_state"], strict=True)
            metrics, y_true, y_prob, image_ids, cell_keys, image_paths = _evaluate(
                model_name, model, val_loader, device
            )
            fold_dir = model_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
            _write_predictions(
                fold_dir / "predictions.csv",
                y_true,
                y_prob,
                image_ids,
                cell_keys,
                image_paths,
            )
            (fold_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            fold_rows.append(
                {
                    "model": model_name,
                    "paper_label": MODEL_LABELS[model_name],
                    "category": MODEL_CATEGORIES[model_name],
                    "fold": fold,
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                }
            )
            del model, optimizer, scheduler, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()

        registry.append(model_registry_entry)

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(out_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    metric_names = (
        "accuracy",
        "macro_f1",
        "macro_sensitivity",
        "macro_specificity",
        "macro_auc",
    )
    summary_rows = []
    for (model, label, category), group in folds_frame.groupby(
        ["model", "paper_label", "category"], sort=False
    ):
        row = {
            "model": model,
            "paper_label": label,
            "category": category,
            "fold_count": int(len(group)),
        }
        for metric in metric_names:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    (out_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "schema_version": "cric-fiveclass-training-v1",
        "data_dir": str(data_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "models": list(requested),
        "class_names": list(CLASS_NAMES),
        "excluded_label": "SCC",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "folds_run": list(folds),
        "pretrained": bool(args.pretrained),
        "objective": args.objective,
        "device": str(device),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--folds", default="0,1,2,3,4", help="folds to run, e.g. 0 or 0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=("full", "direct_ce"),
        default="full",
        help="MERA-Dx objective: full factorized loss or direct five-class CE only",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
