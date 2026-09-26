"""C2-R1 prototype-conditioned full-view residual model."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.tbs.tbs_fv_factorized_model import (
    TBSFVDualPrototypeResidualModel,
)


class TBSFVDualPrototypeConditionedResidualModel(TBSFVDualPrototypeResidualModel):
    """C2 dual prototypes routed into a small, bounded M0 residual context."""

    def __init__(
        self,
        backbone,
        feature_dim,
        semantic_dim=128,
        residual_logit_bound=0.10,
        prototype_temperature=10.0,
        prototype_residual_bound=0.10,
        prototype_context_scale=0.25,
        activation_checkpointing=True,
    ):
        super().__init__(
            backbone,
            feature_dim,
            semantic_dim=semantic_dim,
            residual_logit_bound=residual_logit_bound,
            prototype_temperature=prototype_temperature,
            prototype_residual_bound=prototype_residual_bound,
            activation_checkpointing=activation_checkpointing,
        )
        scale = float(prototype_context_scale)
        if not 0.0 < scale <= 0.25:
            raise ValueError("prototype_context_scale must be within (0, 0.25]")
        self.prototype_context_scale = scale
        residual_input_dim = self.feature_dim + (2 * self.semantic_dim) + 4
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, self.semantic_dim),
            nn.GELU(),
            nn.Linear(self.semantic_dim, 5),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _residual_context(self, branch):
        prototype_logits = torch.cat(
            (branch["morph_prototype_logits"], branch["evidence_prototype_logits"]), dim=1
        )
        return self.prototype_context_scale * torch.tanh(
            prototype_logits / float(self.prototype_temperature)
        )

    def _residual_input(self, full_features, branch):
        context = self._residual_context(branch)
        return torch.cat(
            (
                full_features,
                branch["morph_features"],
                branch["evidence_features"],
                context,
            ),
            dim=1,
        )


def build_tbs_c2r1_conditioned_model(
    model_name="caformer_s18",
    pretrained=True,
    semantic_dim=128,
    residual_logit_bound=0.10,
    prototype_temperature=10.0,
    prototype_residual_bound=0.10,
    prototype_context_scale=0.25,
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the C2-R1 model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFVDualPrototypeConditionedResidualModel(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
        residual_logit_bound=residual_logit_bound,
        prototype_temperature=prototype_temperature,
        prototype_residual_bound=prototype_residual_bound,
        prototype_context_scale=prototype_context_scale,
    )


__all__ = (
    "TBSFVDualPrototypeConditionedResidualModel",
    "build_tbs_c2r1_conditioned_model",
)
