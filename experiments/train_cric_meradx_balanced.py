"""Run the full MERA-Dx objective with inverse-sqrt class-balanced sampling.

Only the training sampler changes.  The model, loss, transforms, optimizer,
fold manifests, and evaluation remain those of ``train_cric_fiveclass.py``.
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

    # Inverse-sqrt is deliberately milder than inverse-frequency weighting.
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


def _sampling_metadata(data_dir, folds):
    fold_metadata = {}
    for fold in folds:
        frame = base.pd.read_csv(Path(data_dir) / f"fold_{fold}.csv")
        train = frame.loc[frame["split"] == "train"]
        _, class_weights, counts = build_sampling_weights(train["label"].to_numpy())
        fold_metadata[str(fold)] = {
            "train_counts": counts.tolist(),
            "inverse_sqrt_class_weights": class_weights.tolist(),
        }
    return fold_metadata


def main(argv=None):
    args = base.parse_args(argv)
    if args.models.strip() != "mera_dx":
        raise ValueError(
            "This experiment must run only MERA-Dx; use --models mera_dx"
        )

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
                "folds": _sampling_metadata(args.data_dir, folds),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "ablation_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "cric-meradx-balanced-full-v1",
                "objective": "full_factorized_loss",
                "sampling": "inverse_sqrt_class_balanced",
                "models": ["mera_dx"],
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
