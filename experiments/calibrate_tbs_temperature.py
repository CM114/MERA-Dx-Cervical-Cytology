"""M6-posthoc: Global Temperature Scaling on sealed calibration set.

Loads frozen M0 checkpoints, collects logits on calibration + dev,
fits a single temperature T, and produces calibration diagnostics.
Does NOT access test set. CPU-only.
"""

import argparse
import json
import sys
import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.tbs.models import build_stage1_model
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.temperature_scaling import (
    apply_temperature,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    high_group_brier,
    nll,
    per_class_ece,
    reliability_data,
    stratified_ece_high_grade,
    stratified_ece_screening,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M6-posthoc temperature scaling")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to M0 best_model.pth")
    parser.add_argument("--calibration_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="M0 seed label (42, 7, 2026)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for name, path in [("--calibration_csv", args.calibration_csv), ("--dev_csv", args.dev_csv)]:
        if any(token in str(path).lower() for token in ("test",)):
            parser.error(f"{name} must not contain 'test'")

    return args


@torch.no_grad()
def collect_logits(model, loader, device):
    """Run forward pass and return (logits, labels)."""
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["diagnosis_label"]
        output = model(images)
        logits = output["diagnosis_logits"]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main(argv=None):
    args = parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model from checkpoint -----------------------------------------
    model = build_stage1_model(variant="m0", model_name="caformer_s18", pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # --- Datasets -------------------------------------------------------------
    _, eval_transform = build_transforms(args.img_size, input_mode="letterbox")
    cal_ds = XUDataTBS5Dataset(args.calibration_csv, eval_transform)
    dev_ds = XUDataTBS5Dataset(args.dev_csv, eval_transform)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Collect logits -------------------------------------------------------
    print(f"Collecting logits on calibration ({len(cal_ds)}) ...", flush=True)
    cal_logits, cal_labels = collect_logits(model, cal_loader, device)
    print(f"Collecting logits on dev ({len(dev_ds)}) ...", flush=True)
    dev_logits, dev_labels = collect_logits(model, dev_loader, device)

    # --- Fit temperature ------------------------------------------------------
    print("Fitting temperature on calibration set...", flush=True)
    fit_result = fit_temperature(cal_logits, cal_labels)

    # --- Apply to dev ---------------------------------------------------------
    dev_probs_raw = apply_temperature(dev_logits, 1.0)
    dev_probs_cal = apply_temperature(dev_logits, fit_result["temperature"])

    # --- Dev metrics (before) -------------------------------------------------
    ece_raw_eq = expected_calibration_error(dev_probs_raw, dev_labels, 15, "equal_width")
    ece_raw_em = expected_calibration_error(dev_probs_raw, dev_labels, 15, "equal_mass")
    nll_raw = nll(dev_probs_raw, dev_labels)
    brier_raw = brier_score(dev_probs_raw, dev_labels)
    hg_brier_raw = high_group_brier(dev_probs_raw, dev_labels)

    # --- Dev metrics (after) --------------------------------------------------
    ece_cal_eq = expected_calibration_error(dev_probs_cal, dev_labels, 15, "equal_width")
    ece_cal_em = expected_calibration_error(dev_probs_cal, dev_labels, 15, "equal_mass")
    nll_cal = nll(dev_probs_cal, dev_labels)
    brier_cal = brier_score(dev_probs_cal, dev_labels)
    hg_brier_cal = high_group_brier(dev_probs_cal, dev_labels)

    # --- Stratified ECE (after) -----------------------------------------------
    screen_ece = stratified_ece_screening(dev_probs_cal, dev_labels)
    high_ece = stratified_ece_high_grade(dev_probs_cal, dev_labels)
    classwise_ece = per_class_ece(dev_probs_cal, dev_labels)

    # --- Bootstrap CIs on dev -------------------------------------------------
    nll_ci = bootstrap_ci(dev_probs_cal, dev_labels, nll)
    ece_ci = bootstrap_ci(
        dev_probs_cal, dev_labels,
        lambda p, l: expected_calibration_error(p, l, 15, "equal_width")["ece"],
    )
    hg_brier_ci = bootstrap_ci(dev_probs_cal, dev_labels, high_group_brier)

    # --- Reliability data -----------------------------------------------------
    rel_global = reliability_data(dev_probs_cal, dev_labels)
    p_normal = dev_probs_cal[:, 0]
    p_abnormal = 1.0 - p_normal
    is_abnormal = (dev_labels != 0).astype(np.int64)
    screen_probs = np.column_stack([p_normal, p_abnormal])
    rel_screen = reliability_data(screen_probs, is_abnormal)
    p_high = dev_probs_cal[:, 3] + dev_probs_cal[:, 4]
    is_high = np.isin(dev_labels, [3, 4]).astype(np.int64)
    high_probs = np.column_stack([1.0 - p_high, p_high])
    rel_high = reliability_data(high_probs, is_high)

    # --- Assert argmax invariance ---------------------------------------------
    assert np.array_equal(
        dev_probs_raw.argmax(axis=1), dev_probs_cal.argmax(axis=1)
    ), "Temperature scaling changed argmax predictions"

    # --- Save artifacts -------------------------------------------------------
    T = fit_result["temperature"]

    # temperature.json
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "temperature": T,
                "nll_before": fit_result["nll_before"],
                "nll_after": fit_result["nll_after"],
                "converged": fit_result["converged"],
                "n_fev": fit_result["n_fev"],
                "calibration_nll_before": fit_result["nll_before"],
                "calibration_nll_after": fit_result["nll_after"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Predictions CSVs
    class_names = ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]

    def _make_pred_csv(labels, probs_raw, probs_cal):
        rows = []
        for i in range(len(labels)):
            row = {"sample_idx": i, "true_label": int(labels[i])}
            for k, name in enumerate(class_names):
                row[f"prob_raw_{name}"] = float(probs_raw[i, k])
                row[f"prob_cal_{name}"] = float(probs_cal[i, k])
            row["pred"] = int(probs_cal[i].argmax())
            row["confidence_raw"] = float(probs_raw[i].max())
            row["confidence_cal"] = float(probs_cal[i].max())
            rows.append(row)
        return pd.DataFrame(rows)

    cal_probs_raw = apply_temperature(cal_logits, 1.0)
    cal_probs_cal = apply_temperature(cal_logits, fit_result["temperature"])
    _make_pred_csv(cal_labels, cal_probs_raw, cal_probs_cal).to_csv(
        out_dir / "calibration_predictions.csv", index=False
    )
    _make_pred_csv(dev_labels, dev_probs_raw, dev_probs_cal).to_csv(
        out_dir / "dev_predictions.csv", index=False
    )

    # Dev report
    dev_report = {
        "n_bins": 15,
        "ece_equal_width_before": ece_raw_eq["ece"],
        "ece_equal_width_after": ece_cal_eq["ece"],
        "ece_equal_mass_before": ece_raw_em["ece"],
        "ece_equal_mass_after": ece_cal_em["ece"],
        "nll_before": nll_raw,
        "nll_after": nll_cal,
        "brier_before": brier_raw,
        "brier_after": brier_cal,
        "high_group_brier_before": hg_brier_raw,
        "high_group_brier_after": hg_brier_cal,
        "screening_ece": screen_ece["ece"],
        "high_grade_ece": high_ece["ece"],
        "classwise_ece": {k: v["ece"] for k, v in classwise_ece.items()},
        "nll_bootstrap": nll_ci,
        "ece_bootstrap": ece_ci,
        "high_group_brier_bootstrap": hg_brier_ci,
    }
    (out_dir / "dev_report.json").write_text(
        json.dumps(dev_report, indent=2), encoding="utf-8"
    )

    # Reliability data
    (out_dir / "reliability_data.json").write_text(
        json.dumps(
            {
                "global": {"confidences": rel_global[0], "accuracies": rel_global[1]},
                "screening": {"confidences": rel_screen[0], "accuracies": rel_screen[1]},
                "high_grade": {"confidences": rel_high[0], "accuracies": rel_high[1]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Metadata
    metadata = {
        "module": "M6-posthoc",
        "method": "global_temperature_scaling",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "temperature_fit_split": "calibration",
        "calibration_used": True,
        "dev_used_for_selection": False,
        "test_used": False,
        "optimizer": "L-BFGS-B",
        "objective": "NLL",
        "generated_at": str(datetime.datetime.now()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary print
    print(f"\nTemperature: {T:.4f}", flush=True)
    print(f"  ECE (equal-width): {ece_raw_eq['ece']:.4f} → {ece_cal_eq['ece']:.4f}", flush=True)
    print(f"  NLL:               {nll_raw:.4f} → {nll_cal:.4f}", flush=True)
    print(f"  Brier:             {brier_raw:.4f} → {brier_cal:.4f}", flush=True)
    print(f"  High-group Brier:  {hg_brier_raw:.4f} → {hg_brier_cal:.4f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
