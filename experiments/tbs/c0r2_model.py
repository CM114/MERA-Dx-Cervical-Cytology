"""C0-R2 uses the unchanged full-view-primary bounded residual architecture."""

from __future__ import annotations

from experiments.tbs.c0r1_model import TBSC0R1ResidualModel


class TBSC0R2ResidualModel(TBSC0R1ResidualModel):
    """Naming/provenance wrapper; no architectural freedom is added in R2."""


def build_tbs_c0r2_model(model_name="caformer_s18", pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the C0-R2 model") from exc
    backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSC0R2ResidualModel(backbone, feature_dim, residual_logit_bound=0.10)


__all__ = ("TBSC0R2ResidualModel", "build_tbs_c0r2_model")
