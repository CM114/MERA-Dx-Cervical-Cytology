#!/usr/bin/env python3
"""Grouped five-fold KD multi-exit/self-distillation adaptation on SIPaKMeD."""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from experiments.sipakmed.paper_baselines_v1.kd_multiexit import KDMultiExit, kd_loss
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import (
    AUGMENTATION_VARIANTS, DEFAULT_MANIFEST, DEFAULT_RESULTS, EXPECTED_CLASS_NAMES,
    atomic_json, classification_metrics, image_tensor, json_safe_args,
    make_internal_split, seed_everything, validate_manifest,
)

class KDDataset(Dataset):
    def __init__(self, frame, augment):
        self.frame = frame.reset_index(drop=True)
        self.variants = AUGMENTATION_VARIANTS if augment else ("identity",)
    def __len__(self):
        return len(self.frame) * len(self.variants)
    def __getitem__(self, index):
        row_idx, variant_idx = divmod(index, len(self.variants))
        row = self.frame.iloc[row_idx]
        return image_tensor(row.image_path, self.variants[variant_idx]), int(row.label)

def loader(frame, augment, batch_size, workers, shuffle, seed):
    return DataLoader(
        KDDataset(frame, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    y, p = [], []
    for images, labels in data:
        outputs = model(images.to(device, non_blocking=True))
        p.extend(outputs["logits"][-1].argmax(1).cpu().tolist())
        y.extend(labels.tolist())
    return classification_metrics(y, p)

def save_ckpt(path, model, optim, epoch, metrics, config):
    torch.save({
        "method": "KD-MultiExit-SelfDistill", "epoch": epoch,
        "selection_metrics": metrics, "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
    }, path)

def train_fold(fold, manifest, run_root, device, args):
    seed = args.seed + fold
    seed_everything(seed)
    root = run_root / f"fold_{fold}"
    root.mkdir(parents=False, exist_ok=False)
    outer = manifest.loc[manifest.fold != fold].copy()
    heldout = manifest.loc[manifest.fold == fold].copy()
    fit, select = make_internal_split(outer, seed)
    frames = {"fit": fit, "select": select, "heldout": heldout}
    groups = {k: set(v.group_id.astype(str)) for k, v in frames.items()}
    intersections = {f"{a}_x_{b}": len(groups[a] & groups[b]) for a, b in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))}
    if any(intersections.values()):
        raise RuntimeError(f"fold {fold} group leakage: {intersections}")
    split = root / "split"
    split.mkdir()
    for name, frame in frames.items():
        frame.to_csv(split / f"{name}.csv", index=False)
    atomic_json(root / "split_audit.json", {
        "fold": fold, "counts": {k: len(v) for k, v in frames.items()},
        "groups": {k: len(v) for k, v in groups.items()},
        "pairwise_group_intersections": intersections,
        "classes": EXPECTED_CLASS_NAMES, "fit_views": len(AUGMENTATION_VARIANTS),
    })
    fit_data = loader(fit, True, args.batch_size, args.workers, True, seed)
    select_data = loader(select, False, args.eval_batch_size, args.workers, False, seed)
    model = KDMultiExit().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    best_score, best_record = (-1.0, -1.0), None
    log_path, status_path = root / "epochs.jsonl", run_root / "status.json"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        sums = {k: 0.0 for k in ("loss", "hard_loss", "distill_loss", "final_loss", "early_loss")}
        seen = 0
        for batch_idx, (images, labels) in enumerate(fit_data, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            losses = kd_loss(model(images), labels, args.temperature, args.exit_weight, args.distill_weight)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"fold {fold} epoch {epoch}: non-finite loss")
            losses["loss"].backward()
            optim.step()
            n = len(labels)
            for name in sums:
                sums[name] += float(losses[name].detach()) * n
            seen += n
            if args.max_batches_per_epoch and batch_idx >= args.max_batches_per_epoch:
                break
        sched.step()
        select_metrics = evaluate(model, select_data, device)
        record = {
            "fold": fold, "epoch": epoch, "learning_rate": optim.param_groups[0]["lr"],
            **{k: v / max(seen, 1) for k, v in sums.items()},
            "fit_samples_seen": seen, "select": select_metrics, "seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = (select_metrics["macro_f1"], select_metrics["accuracy"])
        if score > best_score:
            best_score, best_record = score, record
            save_ckpt(root / "best.pt", model, optim, epoch, select_metrics, json_safe_args(args))
        save_ckpt(root / "latest.pt", model, optim, epoch, select_metrics, json_safe_args(args))
        atomic_json(status_path, {
            "state": "training", "method": "KD-MultiExit-SelfDistill",
            "fold": fold, "epoch": epoch, "total_epochs": args.epochs,
            "latest_select_macro_f1": select_metrics["macro_f1"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"fold={fold} epoch={epoch}/{args.epochs} loss={record['loss']:.4f} select_macro_f1={select_metrics['macro_f1']:.4f} sec={record['seconds']:.1f}", flush=True)
    if best_record is None:
        raise RuntimeError(f"no checkpoint selected for fold {fold}")
    checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    heldout_metrics = evaluate(model, loader(heldout, False, args.eval_batch_size, args.workers, False, seed), device)
    result = {
        "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD",
        "fold": fold, "seed": seed, "n_fit": len(fit), "n_select": len(select), "n_heldout": len(heldout),
        "fit_views": len(AUGMENTATION_VARIANTS), "best_epoch": int(best_record["epoch"]),
        "heldout_metrics": heldout_metrics, "heldout_evaluation_count": 1,
        "class_names": EXPECTED_CLASS_NAMES, "predictions_recorded": True,
    }
    atomic_json(root / "result.json", result)
    atomic_json(status_path, {"state": "fold_complete", "method": "KD-MultiExit-SelfDistill", "fold": fold, "result": heldout_metrics, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} macro_f1={heldout_metrics['macro_f1']:.4f}", flush=True)
    del model, optim, sched, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def write_summary(root, results):
    rows = [{"fold": r["fold"], **r["heldout_metrics"]} for r in results]
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(root / "fold_metrics.csv", index=False)
    names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary = {
        "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD",
        "completed_folds": len(results), "metrics_mean": {n: float(table[n].mean()) for n in names},
        "metrics_sample_sd": {n: float(table[n].std(ddof=1)) for n in names} if len(table) > 1 else {},
        "folds": rows,
    }
    atomic_json(root / "summary.json", summary)
    return summary

def run_smoke(manifest, device, args, root):
    rows = manifest.iloc[[0, 1]].copy()
    images, labels = next(iter(loader(rows, False, 2, 0, False, args.seed)))
    model = KDMultiExit(width_multiplier=0.25).to(device).eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        losses = kd_loss(outputs, labels.to(device), args.temperature, args.exit_weight, args.distill_weight)
    values = list(outputs["logits"]) + list(losses.values())
    if not all(torch.isfinite(v).all() for v in values):
        raise FloatingPointError("KD smoke produced non-finite output")
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "smoke.json", {"state": "passed", "input_batch": list(images.shape), "loss": float(losses["loss"]), "backbone": model.backbone_name, "num_exits": len(outputs["logits"]), "temperature": args.temperature, "updated_utc": datetime.now(timezone.utc).isoformat()})

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-batches-per-epoch", type=int, default=0)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--exit-weight", type=float, default=0.5)
    p.add_argument("--distill-weight", type=float, default=0.7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-only", action="store_true")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    manifest = validate_manifest(args.manifest)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    root = args.results_root / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "source_fidelity.json", {
        "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD", "paper_pmid": "36412723",
        "implementation_status": "paper-guided adaptation; author code and weights unavailable",
        "paper_reported_score_is_not_rerun_result": True,
        "recoverable_mechanisms": ["shared backbone", "three exits", "hard loss at every exit", "temperature-scaled final-exit self-distillation", "final exit for held-out reporting"],
        "explicit_adaptations": ["image-only five-class SIPaKMeD input", "pretrained=False", "verified rotation/flip augmentation", "200 epochs per fold"],
        "run_config": json_safe_args(args) | {"device_resolved": str(device)},
    })
    atomic_json(root / "status.json", {"state": "initialized", "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD", "configured_folds": [int(x) for x in args.folds.split(",") if x.strip()], "updated_utc": datetime.now(timezone.utc).isoformat()})
    run_smoke(manifest, device, args, root / "smoke")
    if args.smoke_only:
        atomic_json(root / "status.json", {"state": "smoke_complete", "method": "KD-MultiExit-SelfDistill", "updated_utc": datetime.now(timezone.utc).isoformat()})
        return
    results = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        results.append(train_fold(fold, manifest, root, device, args))
        write_summary(root, results)
    summary = write_summary(root, results)
    atomic_json(root / "status.json", {"state": "completed", "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD", "completed_folds": len(results), "summary": summary, "updated_utc": datetime.now(timezone.utc).isoformat()})

if __name__ == "__main__":
    main()
