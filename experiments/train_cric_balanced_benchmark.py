"""Fair CRIC benchmark with inverse-sqrt class-balanced training sampling.

This wrapper keeps the original CRIC training/evaluation protocol and changes
only the training sampler.  It is intended for a fair comparison with the
balanced MERA-Dx run.
"""

from pathlib import Path
import json

import numpy as np

from experiments import train_cric_fiveclass as base


_ORIGINAL_MAKE_LOADER = base._make_loader


def build_sampling_weights(labels, n_classes=5):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("labels contain an invalid class index")

    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f"every class must be present; counts={counts.tolist()}")

    class_weights = 1.0 / np.sqrt(counts.astype(np.float64))
    class_weights /= class_weights.mean()
    sample_weights = class_weights[labels]
    return sample_weights, class_weights, counts


def balanced_make_loader(frame, transform, batch_size, shuffle, num_workers):
    if not shuffle:
        return _ORIGINAL_MAKE_LOADER(
            frame, transform, batch_size, shuffle, num_workers
        )

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = base.CRICCellDataset(frame, transform)
    sample_weights, _, _ = build_sampling_weights(frame["label"].to_numpy())
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def sampling_metadata(data_dir, folds):
    result = {}
    for fold in folds:
        frame = base.pd.read_csv(Path(data_dir) / f"fold_{fold}.csv")
        train = frame.loc[frame["split"] == "train"]
        _, class_weights, counts = build_sampling_weights(train["label"].to_numpy())
        result[str(fold)] = {
            "train_counts": counts.tolist(),
            "inverse_sqrt_class_weights": class_weights.tolist(),
        }
    return result


def main(argv=None):
    args = base.parse_args(argv)
    requested = tuple(x.strip() for x in args.models.split(",") if x.strip())
    allowed = {
        "maxvit_tiny_rw_224",
        "densenet201",
        "swin_tiny_patch4_window7_224",
    }
    unknown = set(requested) - allowed
    if unknown:
        raise ValueError(
            "This balanced benchmark accepts only MaxViT-Tiny, DenseNet-201, "
            f"and direct Swin-Tiny; unknown={sorted(unknown)}"
        )
    if not requested:
        raise ValueError("at least one benchmark model is required")

    folds = tuple(
        int(value.strip()) for value in str(args.folds).split(",") if value.strip()
    )
    base._make_loader = balanced_make_loader
    summary = base.run(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sampling_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "cric-inverse-sqrt-sampler-v1",
                "sampling": "weighted_random_sampler_replacement",
                "weight_rule": "1/sqrt(class_count), normalized to mean 1",
                "folds": sampling_metadata(args.data_dir, folds),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "cric-balanced-baseline-benchmark-v1",
                "models": list(requested),
                "objective": "direct_five_class_cross_entropy_for_baselines",
                "sampling": "inverse_sqrt_class_balanced",
                "data_dir": str(Path(args.data_dir).resolve()),
                "out_dir": str(out_dir.resolve()),
                "epochs": args.epochs,
                "folds": str(args.folds),
                "pretrained": bool(args.pretrained),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
