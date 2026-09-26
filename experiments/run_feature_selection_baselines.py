"""Fold-safe PCA+GWO and CNN-feature+GA+SVM baselines.

Features are extracted independently for each outer fold. PCA and optimizer
fitness are fitted only on an inner split of the outer training data.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.paper_baselines_xudata import compute_metrics, validate_locked_folds, write_run_artifacts


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _extract(csv_path, model, transform, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from experiments.tbs.dataset import XUDataTBS5Dataset

    loader = DataLoader(XUDataTBS5Dataset(csv_path, transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    feats, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            z = model(x)
            if z.ndim > 2:
                z = z.flatten(1)
            feats.append(z.float().cpu().numpy())
            labels.append(batch["diagnosis_label"].numpy())
            paths.extend(batch["image_path"])
    return np.concatenate(feats), np.concatenate(labels), paths


def _fitness(mask, x_train, y_train, x_inner, y_inner, kind):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return -1.0
    if kind == "gwo":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        clf = SVC(kernel="rbf", C=5.0, gamma="scale")
    clf.fit(x_train[:, idx], y_train)
    return float(accuracy_score(y_inner, clf.predict(x_inner[:, idx])) - 0.001 * len(idx) / mask.size)


def _search(x, y, kind, population=8, iterations=5, seed=42):
    _seed(seed)
    train_x, inner_x, train_y, inner_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)
    d = x.shape[1]
    pop = np.random.randint(0, 2, size=(population, d), dtype=np.int8)
    pop[:, np.random.randint(0, d)] = 1
    best = pop[0].copy()
    best_score = -1.0
    for iteration in range(iterations):
        scores = np.array([_fitness(mask, train_x, train_y, inner_x, inner_y, kind) for mask in pop])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best = pop[order[0]].copy()
        elites = pop[order[: max(2, population // 2)]]
        children = []
        while len(children) < population:
            a, b = elites[np.random.randint(len(elites))], elites[np.random.randint(len(elites))]
            point = np.random.randint(1, d) if d > 1 else 1
            child = np.concatenate([a[:point], b[point:]]).astype(np.int8)
            flip = np.random.rand(d) < (0.05 if kind == "ga" else 0.10)
            child[flip] = 1 - child[flip]
            if child.sum() == 0:
                child[np.random.randint(0, d)] = 1
            children.append(child)
        pop = np.stack(children)
    return best.astype(bool), best_score


def run(args):
    import torch
    import timm
    from experiments.tbs.dataset import build_transforms

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_locked_folds(args.fold_root)
    (args.out_dir / "protocol_audit.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _, transform = build_transforms(224, "crop")
    device = torch.device(args.device)
    extractor = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg").to(device)
    summaries = []
    for fold in args.folds:
        fold_root = args.out_dir / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        train_csv = args.fold_root / f"fold_{fold}" / "train.csv"
        val_csv = args.fold_root / f"fold_{fold}" / "val.csv"
        x_train, y_train, train_paths = _extract(train_csv, extractor, transform, device, args.batch_size, args.num_workers)
        x_val, y_val, val_paths = _extract(val_csv, extractor, transform, device, args.batch_size, args.num_workers)
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)
        pca = PCA(n_components=min(args.pca_components, x_train.shape[1]), random_state=42).fit(x_train)
        x_train = pca.transform(x_train).astype(np.float32)
        x_val = pca.transform(x_val).astype(np.float32)
        for kind, label in (("gwo", "PCA-GWO"), ("ga", "CNN-GA-SVM")):
            started = time.time()
            mask, score = _search(x_train, y_train, kind, args.population, args.iterations, 42 + fold)
            selector = KNeighborsClassifier(n_neighbors=5) if kind == "gwo" else SVC(kernel="rbf", C=5.0, gamma="scale", probability=True)
            selector.fit(x_train[:, mask], y_train)
            pred = selector.predict(x_val[:, mask])
            if kind == "ga":
                proba = selector.predict_proba(x_val[:, mask])
            else:
                proba = None
            metrics = compute_metrics(y_val, pred, proba)
            predictions = pd.DataFrame({"image_path": val_paths, "y_true": y_val, "y_pred": pred, "fold": fold})
            config = {"method": label, "implementation_status": "fold-safe paper-guided adaptation", "fold": fold, "seed": 42 + fold, "pca_components": int(pca.n_components_), "selected_features": int(mask.sum()), "inner_search_score": score, "population": args.population, "iterations": args.iterations, "protocol_pool_sha256": protocol["metadata"]["pool_sha256"], "elapsed_seconds": time.time() - started}
            write_run_artifacts(fold_root / label, config, metrics, predictions)
            summaries.append({"method": label, "fold": fold, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "selected_features": int(mask.sum())})
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(summaries).to_csv(args.out_dir / "summary_folds.csv", index=False, lineterminator="\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=lambda value: [int(x) for x in value.split(",")], default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))

