import pandas as pd
import torch

from experiments.sipakmed.paper_baselines_v1.deepcervix_hdff import (
    DeepCervixBranch,
    FeatureFusionHead,
    concatenate_embeddings,
    make_internal_split,
)


def test_four_1024d_embeddings_fuse_to_4096_features():
    embeddings = [torch.randn(3, 1024) for _ in range(4)]
    fused = concatenate_embeddings(embeddings)
    assert fused.shape == (3, 4096)


def test_deepcervix_head_returns_five_class_logits():
    head = FeatureFusionHead(num_classes=5)
    logits = head(torch.randn(2, 4096))
    assert logits.shape == (2, 5)
    assert head.dropout.p == 0.5


def test_author_backbone_branches_return_1024d_features_and_five_logits():
    for architecture in ("vgg16", "vgg19", "xception", "resnet50"):
        branch = DeepCervixBranch(architecture, num_classes=5, pretrained=False)
        branch.eval()
        logits, embedding = branch(torch.randn(1, 3, 96, 96))
        assert logits.shape == (1, 5)
        assert embedding.shape == (1, 1024)


def test_vgg_taps_match_the_authors_intermediate_keras_layers():
    vgg16 = DeepCervixBranch("vgg16", pretrained=False)
    vgg19 = DeepCervixBranch("vgg19", pretrained=False)
    assert len(vgg16.backbone) == 22  # block4_conv3, before its activation
    assert len(vgg19.backbone) == 27  # block5_conv2, before its activation


def test_xception_uses_the_canonical_depthwise_separable_backbone():
    xception = DeepCervixBranch("xception", pretrained=False)
    assert hasattr(xception.backbone, "block12")
    assert xception.backbone.num_features == 2048


def test_internal_split_is_group_disjoint_and_keeps_all_classes():
    rows = []
    for label in range(5):
        for group_number in range(10):
            group_id = f"source{group_number}"
            for image_number in range(2):
                rows.append(
                    {
                        "image_path": f"class{label}_{group_id}_{image_number}.bmp",
                        "label": label,
                        "group_id": group_id,
                    }
                )
    fit, select = make_internal_split(pd.DataFrame(rows), seed=42)
    assert set(fit["group_id"]).isdisjoint(set(select["group_id"]))
    assert set(fit["label"]) == set(range(5))
    assert set(select["label"]) == set(range(5))
