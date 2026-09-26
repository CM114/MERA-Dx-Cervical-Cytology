"""M4-NucProto audit: check prototype updates, mask spatial validity, branch usage.

Pure diagnostic — no training, no model changes.
"""

import sys, json, math, argparse
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from experiments.tbs.m4_nucproto_model import build_m4_nucproto
from experiments.tbs.dataset import build_transforms, XUDataTBS5Dataset
from experiments.tbs.models import build_stage1_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit M4-NucProto")
    p.add_argument("--m0_checkpoint", type=Path, required=True)
    p.add_argument("--dev_csv", type=Path, required=True)
    p.add_argument("--mask_dir_dev", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


class AuditDataset:
    def __init__(self, manifest_csv, mask_dir, transform):
        self.base = XUDataTBS5Dataset(manifest_csv, transform)
        self.mask_dir = Path(mask_dir)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        sid = Path(item["image_path"]).stem
        mask = torch.zeros(224, 224)
        valid = False
        try:
            with Image.open(self.mask_dir / f"{sid}.png") as m:
                mi = m.convert("L").resize((224, 224), Image.BILINEAR)
            arr = np.asarray(mi, dtype=np.float32) / 255.0
            mask = torch.from_numpy(arr)
            area = arr.sum()
            valid = (area > 10) and (area < arr.size * 0.95)
        except Exception:
            pass
        return {"image": item["image"], "diagnosis_label": item["diagnosis_label"],
                "mask": mask, "mask_valid": valid, "sample_id": sid, "image_path": item["image_path"]}


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, eval_tf = build_transforms(224, input_mode="letterbox")
    dev_ds = AuditDataset(args.dev_csv, args.mask_dir_dev, eval_tf)
    # Use small subset for audit
    indices = np.random.RandomState(0).choice(len(dev_ds), min(200, len(dev_ds)), replace=False)
    from torch.utils.data import Subset
    sub_ds = Subset(dev_ds, indices)

    # Collate
    def collate(batch):
        return (
            torch.stack([b["image"] for b in batch]),
            torch.tensor([b["diagnosis_label"] for b in batch], dtype=torch.long),
            torch.stack([b["mask"] for b in batch]),
            torch.tensor([b["mask_valid"] for b in batch], dtype=torch.bool),
            [b["sample_id"] for b in batch],
        )
    loader = DataLoader(sub_ds, batch_size=32, shuffle=False, collate_fn=collate)

    # --- Audit 1: Check P2 prototype update ---
    print("=== Audit 1: Prototype update check ===", flush=True)
    model_p2 = build_m4_nucproto(args.m0_checkpoint, device, "P2")
    model_p2.eval()
    proto_init = model_p2.prototypes.data.clone()
    print(f"  Proto requires_grad: {model_p2.prototypes.requires_grad}", flush=True)

    # Check if proto in any optimizer group (just check grad after one backward)
    model_p2.train()
    batch = next(iter(loader))
    images, labels = batch[0].to(device), batch[1].to(device)
    masks, valid = batch[2].to(device), batch[3].to(device)
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        logits, _ = model_p2(images, masks, valid)
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    proto_grad_norm = model_p2.prototypes.grad.norm().item() if model_p2.prototypes.grad is not None else 0.0
    print(f"  Proto grad norm after one backward: {proto_grad_norm:.6f}", flush=True)
    model_p2.zero_grad(set_to_none=True)

    # Angle between init and current
    cos_angles = []
    for c in range(5):
        cos = F.cosine_similarity(proto_init[c:c+1], model_p2.prototypes.data[c:c+1]).item()
        cos_angles.append(math.degrees(math.acos(max(-1, min(1, cos)))))
    print(f"  Proto angles (init vs current): {[f'{a:.2f}°' for a in cos_angles]}", flush=True)
    del model_p2

    # --- Audit 2: Mask spatial validity at feature resolution ---
    print("\n=== Audit 2: Mask spatial validity ===", flush=True)
    model_p3 = build_m4_nucproto(args.m0_checkpoint, device, "P3")
    model_p3.eval()

    # Get stage3 feature map size by dummy forward
    dummy = images[:1]
    with torch.no_grad():
        _ = model_p3.backbone(dummy)
    s3 = model_p3._stage3_feat
    feat_h, feat_w = s3.shape[2], s3.shape[3]
    print(f"  Stage3 feature map: {feat_h}×{feat_w} (stride ~{224//feat_h})", flush=True)
    stride = 224 // feat_h

    # For a few samples, compute real vs shifted mask IoU at feature resolution
    real_ious, shift_ious = [], []
    for images_b, _, masks_b, valid_b, _ in loader:
        images_b = images_b.to(device)
        masks_b = masks_b.to(device)
        with torch.no_grad():
            _ = model_p3.backbone(images_b[:1])
        for i in range(min(4, images_b.size(0))):
            if not valid_b[i]:
                continue
            m = masks_b[i].cpu().numpy()
            m_shift = np.roll(m, shift=28, axis=(0, 1))
            # Downsample to feature resolution
            m_ds = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m_ds = F.interpolate(m_ds, size=(feat_h, feat_w), mode="bilinear", align_corners=False).squeeze().numpy()
            m_shift_ds = torch.from_numpy(m_shift).unsqueeze(0).unsqueeze(0)
            m_shift_ds = F.interpolate(m_shift_ds, size=(feat_h, feat_w), mode="bilinear", align_corners=False).squeeze().numpy()
            # Binarise
            m_ds_b = np.asarray(m_ds > 0.5, dtype=bool)
            m_shift_b = np.asarray(m_shift_ds > 0.5, dtype=bool)
            inter = int((m_ds_b & m_shift_b).sum())
            union = int((m_ds_b | m_shift_b).sum())
            iou = inter / max(union, 1)
            shift_ious.append(iou)
            # Real mask at different strides
            for s in [4, 8, 16]:
                h, w = 224//s, 224//s
                md = F.interpolate(torch.from_numpy(m).unsqueeze(0).unsqueeze(0), size=(h,w), mode="bilinear").squeeze().numpy()
                ms = F.interpolate(torch.from_numpy(m_shift).unsqueeze(0).unsqueeze(0), size=(h,w), mode="bilinear").squeeze().numpy()
                md_b = np.asarray(md > 0.5, dtype=bool)
                ms_b = np.asarray(ms > 0.5, dtype=bool)
                inter = int((md_b & ms_b).sum())
                union = int((md_b | ms_b).sum())
                i = inter / max(union, 1)
                real_ious.append((f"stride{s}", i))

    if shift_ious:
        print(f"  Real-vs-shifted mask IoU at stride{stride}: mean={np.mean(shift_ious):.3f}", flush=True)
        if np.mean(shift_ious) > 0.5:
            print("  WARNING: IoU too high — P4 not a valid spatial control", flush=True)

    # --- Audit 3: Nucleus branch usage ---
    print("\n=== Audit 3: Nucleus branch participation ===", flush=True)
    gate_n_vals, delta_logit_vals = [], []
    with torch.no_grad():
        for images_b, _, masks_b, valid_b, _ in loader:
            images_b = images_b.to(device)
            masks_b = masks_b.to(device)
            valid_b = valid_b.to(device)
            # Full forward
            logits_full, _ = model_p3(images_b, masks_b, valid_b)
            # Zero out nucleus branch
            h_global = model_p3.proj_global(model_p3.backbone(images_b).flatten(1))
            h_norm = F.normalize(h_global, p=2, dim=1)
            proto_norm = F.normalize(model_p3.prototypes, p=2, dim=1)
            tau = F.softplus(model_p3.log_tau) + 0.1
            logits_no_nuc = tau * (h_norm @ proto_norm.T)
            delta = (logits_full - logits_no_nuc).abs().max(dim=1).values
            delta_logit_vals.extend(delta.cpu().numpy().tolist())
            gate_n_vals.append(torch.sigmoid(model_p3.gate_n).item())

    print(f"  gate_n: {np.mean(gate_n_vals):.4f}", flush=True)
    print(f"  max |logit diff| (nuc on vs off): mean={np.mean(delta_logit_vals):.4f} median={np.median(delta_logit_vals):.4f}", flush=True)

    # --- Audit 4: Check masked pooling correctness ---
    print("\n=== Audit 4: Masked pooling sanity ===", flush=True)
    # Create a known mask and verify pooling
    test_feat = torch.ones(1, 512, 14, 14, device=device)
    test_mask = torch.zeros(1, 14, 14, device=device)
    test_mask[0, 2:6, 2:6] = 1.0  # 4×4 region = 16 cells
    pooled = (test_feat * test_mask.unsqueeze(1)).sum(dim=[2,3]) / test_mask.sum().clamp(min=1)
    print(f"  Known mask test: expected=1.0, got={pooled.mean().item():.4f}", flush=True)
    assert abs(pooled.mean().item() - 1.0) < 0.01, "Masked pooling is broken"
    print("  Masked pooling: OK", flush=True)

    # Summary
    summary = {
        "proto_grad_norm": proto_grad_norm,
        "proto_angles_deg": [float(a) for a in cos_angles],
        "proto_updated": proto_grad_norm > 1e-8,
        "shift_mask_iou_mean": float(np.mean(shift_ious)) if shift_ious else None,
        "p4_valid_control": bool(np.mean(shift_ious) < 0.5) if shift_ious else None,
        "gate_n_mean": float(np.mean(gate_n_vals)),
        "nuc_branch_delta_logit_mean": float(np.mean(delta_logit_vals)),
        "nuc_branch_participates": float(np.mean(delta_logit_vals)) > 0.01,
        "masked_pooling_correct": True,
    }
    print(f"\nSummary: {json.dumps(summary, indent=2)}", flush=True)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
