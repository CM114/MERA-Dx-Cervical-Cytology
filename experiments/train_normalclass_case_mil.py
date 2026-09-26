"""Train frozen-feature gated Attention-MIL on normalClass case bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.case_mil import CaseFeatureBags, GatedAttentionMIL, case_mil_collate
from experiments.tbs.metrics import compute_stage1_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_case_class_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, np.arange(5)).all():
        raise ValueError("labels must be a nonempty one-dimensional array in [0, 4]")
    counts = np.bincount(labels, minlength=5)
    if np.any(counts == 0):
        raise ValueError("all five diagnosis classes are required")
    return (1.0 / (5.0 * counts[labels])).astype(np.float64)


class CaseBagDataset:
    def __init__(self, bags, train=False):
        self.bags = bags
        self.train = bool(train)
        self.epoch = 0

    def __len__(self):
        return len(self.bags)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        return self.bags.get_case(index, train=self.train, epoch=self.epoch)


def _loader(dataset, batch_size, seed, weighted):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if weighted:
        sampler = WeightedRandomSampler(
            torch.as_tensor(build_case_class_weights(dataset.bags.case_labels), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=sampler is None and not weighted,
        sampler=sampler,
        num_workers=0,
        collate_fn=case_mil_collate,
        generator=generator,
    )


def write_case_prediction_artifacts(out_dir, case_ids, labels, probabilities, patch_counts, attention_paths):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (labels.size, 5):
        raise ValueError("probabilities must have shape [cases, 5]")
    if len(case_ids) != labels.size or len(patch_counts) != labels.size:
        raise ValueError("case metadata lengths must match labels")
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    names = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
    frame = pd.DataFrame(
        {
            "case_id": [str(value) for value in case_ids],
            "diagnosis_label": labels,
            "diagnosis_name": [names[int(value)] for value in labels],
            "pred_label": predictions,
            "pred_name": [names[int(value)] for value in predictions],
            "patch_count": [int(value) for value in patch_counts],
        }
    )
    for index, name in enumerate(names):
        frame[f"prob_{name}"] = probabilities[:, index]
    frame["screen_prob_from_diagnosis"] = probabilities[:, 1:].sum(axis=1)
    frame.to_csv(out_dir / "case_predictions.csv", index=False, lineterminator="\n")
    metrics = compute_stage1_metrics(labels, probabilities)
    (out_dir / "case_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    matrix = confusion_matrix(labels, predictions, labels=list(range(5)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out_dir / "case_confusion_matrix.csv")
    report = classification_report(labels, predictions, labels=list(range(5)), target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "case_classification_report.csv")
    attention_rows = []
    for case_id, paths in zip(case_ids, attention_paths):
        for rank, path in enumerate(paths, start=1):
            attention_rows.append({"case_id": str(case_id), "attention_rank": rank, "image_path": str(path)})
    pd.DataFrame(attention_rows, columns=("case_id", "attention_rank", "image_path")).to_csv(out_dir / "attention_audit.csv", index=False, lineterminator="\n")
    return metrics


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _evaluate(model, loader, device):
    import torch

    model.eval()
    labels, probabilities, case_ids, patch_counts, attention_paths = [], [], [], [], []
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            target = batch["labels"].to(device)
            output = model(features, mask)
            losses.append((float(criterion(output["logits"], target).cpu()), int(target.numel())))
            labels.extend(target.cpu().numpy().tolist())
            probabilities.append(output["probs"].cpu().numpy())
            case_ids.extend(batch["case_ids"])
            patch_counts.extend(batch["patch_counts"])
            for row, paths in enumerate(batch["image_paths"]):
                valid_count = int(mask[row].sum().item())
                attention = output["attention"][row, :valid_count].cpu().numpy()
                order = np.argsort(-attention)[: min(10, valid_count)]
                attention_paths.append([paths[int(index)] for index in order])
    probabilities = np.concatenate(probabilities, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    metrics = compute_stage1_metrics(labels, probabilities)
    metrics["case_val_loss"] = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    return {"metrics": metrics, "labels": labels, "probabilities": probabilities, "case_ids": case_ids, "patch_counts": patch_counts, "attention_paths": attention_paths}


def train(train_npz, dev_npz, out_dir, epochs=30, batch_size=8, max_patches=128, hidden_dim=256, lr=1e-3, weight_decay=1e-4, seed=42, patience=8):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for case MIL training")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    seed_everything(seed)
    train_bags = CaseFeatureBags.from_npz(train_npz, max_patches=max_patches, seed=seed)
    dev_bags = CaseFeatureBags.from_npz(dev_npz, max_patches=max_patches, seed=seed)
    if train_bags.feature_dim != dev_bags.feature_dim:
        raise ValueError("train and dev feature dimensions must match")
    train_dataset = CaseBagDataset(train_bags, train=True)
    dev_dataset = CaseBagDataset(dev_bags, train=False)
    train_loader = _loader(train_dataset, batch_size, seed, weighted=True)
    dev_loader = _loader(dev_dataset, batch_size, seed, weighted=False)
    device = torch.device("cuda:0")
    model = GatedAttentionMIL(train_bags.feature_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    history, best, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, int(epochs) + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss, total_count = 0.0, 0
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            target = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, mask)
            loss = criterion(output["logits"], target)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("case MIL loss is nonfinite")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(target.numel())
            total_count += int(target.numel())
        evaluation = _evaluate(model, dev_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            **evaluation["metrics"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={row['train_loss']:.4f} | case_val_loss={row['case_val_loss']:.4f} | macro_f1={row['macro_f1']:.4f} | balanced_acc={row['balanced_accuracy']:.4f} | macro_auc={row['macro_auc']:.4f}", flush=True)
        if row["macro_f1"] > best:
            best, best_epoch, stale = row["macro_f1"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "feature_dim": train_bags.feature_dim, "hidden_dim": hidden_dim, "metrics": row}, out_dir / "best_model.pth")
            write_case_prediction_artifacts(out_dir, evaluation["case_ids"], evaluation["labels"], evaluation["probabilities"], evaluation["patch_counts"], evaluation["attention_paths"])
        else:
            stale += 1
        if stale >= int(patience):
            break
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "feature_dim": train_bags.feature_dim, "hidden_dim": hidden_dim}, out_dir / "last_model.pth")
    summary = {
        "schema_version": "xudata-replacement-normalclass-case-attention-mil-v1",
        "train_npz": str(Path(train_npz).resolve()),
        "dev_npz": str(Path(dev_npz).resolve()),
        "train_case_count": len(train_bags),
        "dev_case_count": len(dev_bags),
        "feature_dim": train_bags.feature_dim,
        "train_npz_sha256": file_sha256(train_npz),
        "dev_npz_sha256": file_sha256(dev_npz),
        "max_patches": max_patches,
        "best_epoch": best_epoch,
        "best_case_macro_f1": best,
        "heldout_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "raw_data_modified": False,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "best_metrics.json",
        {"epoch": best_epoch, **history[best_epoch - 1]} if best_epoch else {},
    )
    write_json(
        out_dir / "safety.json",
        {
            "schema_version": "xudata-replacement-normalclass-case-attention-mil-safety-v1",
            "raw_data_modified": False,
            "heldout_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
            "purpose": "fold-0 normalClass train/dev case-level Attention-MIL development only",
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=Path, required=True)
    parser.add_argument("--dev_npz", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_patches", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(**vars(args))


if __name__ == "__main__":
    main()
