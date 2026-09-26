"""C2-R3 model: independently initialized C2-R1 prototype-conditioned residual."""

from __future__ import annotations

from experiments.tbs.tbs_fv_c2r1_model import (
    TBSFVDualPrototypeConditionedResidualModel,
)


class TBSFVDualPrototypeConditionalBoundaryModel(
    TBSFVDualPrototypeConditionedResidualModel
):
    """Named C2-R3 wrapper; architecture remains identical to C2-R1."""


def build_tbs_c2r3_conditional_boundary_model(
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
        raise RuntimeError("timm is required to build the C2-R3 model") from exc
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSFVDualPrototypeConditionalBoundaryModel(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
        residual_logit_bound=residual_logit_bound,
        prototype_temperature=prototype_temperature,
        prototype_residual_bound=prototype_residual_bound,
        prototype_context_scale=prototype_context_scale,
    )


__all__ = (
    "TBSFVDualPrototypeConditionalBoundaryModel",
    "build_tbs_c2r3_conditional_boundary_model",
)
