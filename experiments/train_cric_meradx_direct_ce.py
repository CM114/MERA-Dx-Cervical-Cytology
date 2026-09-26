"""Run the MERA-Dx architecture on CRIC with direct five-class CE only.

This is a controlled ablation: the architecture, prototype initialization,
folds, transforms, optimizer, and evaluation stay identical to
``train_cric_fiveclass.py``.  Only the factorized auxiliary loss is replaced by
NLL loss on the final five-class diagnosis log-probabilities.
"""

from pathlib import Path
import json

from experiments import train_cric_fiveclass as base

_ORIGINAL_LOSS = base._loss_for_output


def direct_ce_loss(model_name, output, labels):
    import torch.nn.functional as F

    if model_name != "mera_dx":
        return _ORIGINAL_LOSS(model_name, output, labels)
    loss = F.nll_loss(output["diagnosis_log_probs"], labels)
    return loss, {"diagnosis_loss": float(loss.detach())}


def main(argv=None):
    args = base.parse_args(argv)
    if args.models.strip() != "mera_dx":
        raise ValueError(
            "This ablation must run only MERA-Dx; pass --models mera_dx"
        )
    base._loss_for_output = direct_ce_loss
    summary = base.run(args)
    marker = {
        "schema_version": "cric-meradx-direct-ce-ablation-v1",
        "objective": "direct_five_class_nll_on_final_diagnosis",
        "models": ["mera_dx"],
        "data_dir": str(Path(args.data_dir).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "epochs": args.epochs,
        "folds": str(args.folds),
        "pretrained": bool(args.pretrained),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_metadata.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
