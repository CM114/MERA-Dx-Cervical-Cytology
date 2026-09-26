"""Train paper-guided baseline adapters on the locked Xudata five-fold pool."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper_baselines_xudata import (
    CLASS_NAMES,
    compute_metrics,
    validate_locked_folds,
    write_run_artifacts,
)
from experiments.paper_baseline_adapters import METHOD_BUILDERS, build_method


METHOD_META = {
    "HCT-Net": {"paper_score_sipakmed": 0.9926, "status": "paper-guided adaptation"},
    "LGPNet": {"paper_score_sipakmed": 0.9753, "status": "paper-guided adaptation"},
    "A2SDNet121": {"paper_score_sipakmed": 0.9922, "status": "paper-guided adaptation"},
    "DeepCervix-HDFF": {"paper_score_sipakmed": 0.9914, "status": "paper-guided adaptation"},
    "MSENet": {"paper_score_sipakmed": 0.9721, "status": "paper-guided adaptation"},
    "MSCCNet": {"paper_score_sipakmed": 0.9790, "status": "paper-guided adaptation"},
    "CerCan-Net": {"paper_score_sipakmed": 0.9770, "status": "paper-guided adaptation"},
    "DIFF": {"paper_score_sipakmed": 0.9602, "status": "paper-guided adaptation"},
    "MTFM": {"paper_score_sipakmed": 0.9867, "status": "paper-guided adaptation"},
    "KnowledgeDistill": {"paper_score_sipakmed": 0.9852, "status": "paper-guided adaptation"},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHOD_BUILDERS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args(argv)


def _loader(csv_path, transform, batch_size, workers, shuffle, seed):
    import random
    import torch
    from torch.utils.data import DataLoader
    from experiments.tbs.dataset import XUDataTBS5Dataset

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        XUDataTBS5Dataset(csv_path, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _seed(seed):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(model, loader, optimizer, device, train, max_batches=0):
    import torch

    model.train(train)
    losses = []
    all_y, all_pred, all_prob, all_paths = [], [], [], []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["diagnosis_label"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            prob = logits.softmax(dim=1).detach().cpu().numpy()
            pred = prob.argmax(axis=1)
            all_y.append(labels.detach().cpu().numpy())
            all_pred.append(pred)
            all_prob.append(prob)
            all_paths.extend(batch["image_path"])
            losses.append(float(loss.detach().cpu()))
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)
    return float(np.mean(losses)), compute_metrics(y, pred, prob), y, pred, prob, all_paths


def run(args):
    import torch
    from experiments.tbs.dataset import build_transforms

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    folds = tuple(int(x) for x in args.folds.split(",") if x.strip())
    unknown = [name for name in methods if name not in METHOD_BUILDERS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_locked_folds(args.fold_root)
    (args.out_dir / "protocol_audit.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    train_transform, eval_transform = build_transforms(224, "crop")
    device = torch.device(args.device)
    all_summary = []
    for method in methods:
        method_root = args.out_dir / method.replace("/", "_")
        method_root.mkdir(exist_ok=True)
        for fold in folds:
            fold_root = method_root / f"fold_{fold}"
            if (fold_root / "metrics.json").is_file():
                print(f"SKIP existing {method} fold {fold}", flush=True)
                continue
            fold_root.mkdir(parents=True, exist_ok=True)
            seed = 42 + fold
            _seed(seed)
            train_csv = args.fold_root / f"fold_{fold}" / "train.csv"
            val_csv = args.fold_root / f"fold_{fold}" / "val.csv"
            train_loader = _loader(train_csv, train_transform, args.batch_size, args.num_workers, True, seed)
            val_loader = _loader(val_csv, eval_transform, args.batch_size, args.num_workers, False, seed)
            model = build_method(method, num_classes=5, pretrained=args.pretrained).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            history = []
            started = time.time()
            for epoch in range(1, args.epochs + 1):
                train_loss, train_metrics, *_ = _run_epoch(model, train_loader, optimizer, device, True, args.max_train_batches)
                val_loss, val_metrics, y, pred, prob, paths = _run_epoch(model, val_loader, optimizer, device, False, args.max_val_batches)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
                history.append(row)
                print(json.dumps({"method": method, "fold": fold, **row}, ensure_ascii=False), flush=True)
                scheduler.step()
                torch.save(
                    {
                        "method": method,
                        "fold": fold,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    },
                    fold_root / "last.pt",
                )
            final_metrics = compute_metrics(y, pred, prob)
            predictions = pd.DataFrame({"image_path": paths, "y_true": y, "y_pred": pred, **{f"proba_{i}": prob[:, i] for i in range(5)}, "fold": fold})
            config = {
                "method": method,
                "implementation_status": METHOD_META[method]["status"],
                "paper_sipakmed_accuracy_reference_only": METHOD_META[method]["paper_score_sipakmed"],
                "fold": fold,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "pretrained": args.pretrained,
                "class_names": list(CLASS_NAMES),
                "protocol_pool_sha256": protocol["metadata"]["pool_sha256"],
                "elapsed_seconds": time.time() - started,
                "python": platform.python_version(),
            }
            write_run_artifacts(fold_root, config, {"final": final_metrics, "history": history}, predictions)
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
            all_summary.append({"method": method, "fold": fold, "macro_f1": final_metrics["macro_f1"], "accuracy": final_metrics["accuracy"]})
    pd.DataFrame(all_summary).to_csv(args.out_dir / "summary_folds.csv", index=False, lineterminator="\n")
    return all_summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
