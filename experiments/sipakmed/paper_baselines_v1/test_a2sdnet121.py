import pandas as pd
import torch
from torch import nn

from experiments.sipakmed.paper_baselines_v1.a2sdnet121 import (
    A2SDNet121,
    A2SDAtrousDenseBlock,
    SqueezeExcitation,
    make_internal_split,
)
from experiments.sipakmed.paper_baselines_v1.train_a2sdnet121_sipakmed import (
    restore_training_checkpoint,
    should_resume_fold,
)


def test_a2sdnet121_has_the_paper_stem_and_five_class_head():
    model = A2SDNet121(num_classes=5)
    assert model.stem[0].kernel_size == (3, 3)
    assert model.stem[0].stride == (1, 1)
    assert model.stem[3].kernel_size == 2
    assert model.stem[3].stride == 2
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_a2sdnet121_uses_four_adb_se_transition_stages():
    model = A2SDNet121(num_classes=5)
    assert tuple(model.block_config) == (6, 12, 16, 24)
    assert len(model.blocks) == 4
    assert len(model.se_blocks) == 4
    assert len(model.transitions) == 3
    assert all(isinstance(block, A2SDAtrousDenseBlock) for block in model.blocks)
    assert all(isinstance(block, SqueezeExcitation) for block in model.se_blocks)
    for block in model.blocks:
        assert [layer.dilation for layer in block.layers[-3:]] == [(1, 1), (2, 2), (3, 3)]


def test_internal_split_is_group_disjoint_and_keeps_all_classes():
    rows = []
    for label in range(5):
        for group_number in range(10):
            group_id = f"source{group_number}"
            rows.append(
                {
                    "image_path": f"class{label}_{group_id}.bmp",
                    "label": label,
                    "group_id": group_id,
                }
            )
    fit, select = make_internal_split(pd.DataFrame(rows), seed=42)
    assert set(fit["group_id"]).isdisjoint(set(select["group_id"]))
    assert set(fit["label"]) == set(range(5))
    assert set(select["label"]) == set(range(5))


def test_restore_checkpoint_starts_after_saved_epoch_and_aligns_scheduler(tmp_path):
    source_model = nn.Linear(2, 2)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=1e-4)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=30, gamma=0.1)
    for _ in range(29):
        source_scheduler.step()
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {
            "method": "A2SDNet121",
            "epoch": 30,
            "selection_metrics": {"macro_f1": 0.5, "accuracy": 0.6},
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "scheduler_state_dict": source_scheduler.state_dict(),
        },
        checkpoint_path,
    )

    target_model = nn.Linear(2, 2)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=1e-4)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=30, gamma=0.1)
    checkpoint, next_epoch = restore_training_checkpoint(
        checkpoint_path, target_model, target_optimizer, target_scheduler
    )

    assert checkpoint["epoch"] == 30
    assert next_epoch == 31
    assert target_scheduler.last_epoch == 30
    assert target_optimizer.param_groups[0]["lr"] == 1e-5


def test_resume_existing_only_resumes_an_existing_fold_directory(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "fold_2").mkdir(parents=True)

    assert should_resume_fold(run_root, 2, resume_existing=True)
    assert not should_resume_fold(run_root, 3, resume_existing=True)
    assert not should_resume_fold(run_root, 3, resume_existing=False)
