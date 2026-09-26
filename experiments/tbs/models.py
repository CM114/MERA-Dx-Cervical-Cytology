import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE1_VARIANTS = ("m0", "m1", "m2")


class ExternalAbnormalClassifier(nn.Module):
    """Shared backbone with a four-class abnormal-only classification head."""

    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.head = nn.Linear(self.feature_dim, 4)

    def forward(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        logits = self.head(features)
        return {
            "features": features,
            "logits": logits,
            "probs": F.softmax(logits.float(), dim=1),
        }


def conditional_log_probs(screen_logits, abnormal_logits):
    if screen_logits.ndim == 2 and screen_logits.shape[1] == 1:
        screen_logits = screen_logits[:, 0]
    if screen_logits.ndim != 1:
        raise ValueError("screen_logits must have shape [batch] or [batch, 1]")
    if abnormal_logits.ndim != 2 or abnormal_logits.shape[1] != 4:
        raise ValueError("abnormal_logits must have shape [batch, 4]")
    if abnormal_logits.shape[0] != screen_logits.shape[0]:
        raise ValueError("screen and abnormal logits must use the same batch size")

    screen_logits = screen_logits.float()
    abnormal_logits = abnormal_logits.float()
    normal_log_prob = F.logsigmoid(-screen_logits).unsqueeze(1)
    abnormal_log_prob = F.logsigmoid(screen_logits).unsqueeze(1)
    abnormal_class_log_prob = F.log_softmax(abnormal_logits, dim=1)
    combined_log_probs = torch.cat(
        (normal_log_prob, abnormal_log_prob + abnormal_class_log_prob),
        dim=1,
    )
    return combined_log_probs - torch.logsumexp(
        combined_log_probs, dim=1, keepdim=True
    )


class Stage1Classifier(nn.Module):
    def __init__(self, backbone, feature_dim, variant):
        super().__init__()
        if variant not in STAGE1_VARIANTS:
            raise ValueError(
                f"Unknown Stage-1 variant: {variant}. Expected one of {STAGE1_VARIANTS}"
            )
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.variant = variant

        if variant in ("m0", "m1"):
            self.diagnosis_head = nn.Linear(self.feature_dim, 5)
        else:
            self.diagnosis_head = None

        if variant in ("m1", "m2"):
            self.screen_head = nn.Linear(self.feature_dim, 1)
        else:
            self.screen_head = None

        if variant == "m2":
            self.abnormal_head = nn.Linear(self.feature_dim, 4)
        else:
            self.abnormal_head = None

    def forward(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)

        if self.variant in ("m0", "m1"):
            diagnosis_logits = self.diagnosis_head(features)
            diagnosis_log_probs = F.log_softmax(diagnosis_logits, dim=1)
            diagnosis_probs = diagnosis_log_probs.exp()
            diagnosis_screen_probs = diagnosis_probs[:, 1:].sum(dim=1)

            if self.variant == "m1":
                screen_logits = self.screen_head(features).squeeze(1)
                screen_probs = torch.sigmoid(screen_logits)
            else:
                screen_logits = None
                screen_probs = diagnosis_screen_probs

            abnormal_logits = None
        else:
            diagnosis_logits = None
            screen_logits = self.screen_head(features).squeeze(1)
            abnormal_logits = self.abnormal_head(features)
            diagnosis_log_probs = conditional_log_probs(screen_logits, abnormal_logits)
            diagnosis_probs = diagnosis_log_probs.exp()
            screen_probs = torch.sigmoid(screen_logits)
            diagnosis_screen_probs = diagnosis_probs[:, 1:].sum(dim=1)

        return {
            "features": features,
            "diagnosis_logits": diagnosis_logits,
            "diagnosis_log_probs": diagnosis_log_probs,
            "diagnosis_probs": diagnosis_probs,
            "diagnosis_screen_probs": diagnosis_screen_probs,
            "screen_logits": screen_logits,
            "screen_probs": screen_probs,
            "abnormal_logits": abnormal_logits,
        }


class TBSSemanticClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, semantic_dim=128):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        if self.semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")

        self.screen_head = nn.Linear(self.feature_dim, 1)
        self.abnormal_head = nn.Linear(self.feature_dim, 4)
        self.morph_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.evidence_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim),
            nn.GELU(),
        )
        self.morph_head = nn.Linear(self.semantic_dim, 2)
        self.evidence_head = nn.Linear(self.semantic_dim, 2)

    def forward(self, images):
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)

        screen_logits = self.screen_head(features).squeeze(1)
        abnormal_logits = self.abnormal_head(features)
        diagnosis_log_probs = conditional_log_probs(screen_logits, abnormal_logits)
        diagnosis_probs = diagnosis_log_probs.exp()

        morph_features = F.normalize(self.morph_projector(features), dim=1)
        evidence_features = F.normalize(self.evidence_projector(features), dim=1)
        morph_logits = self.morph_head(morph_features)
        evidence_logits = self.evidence_head(evidence_features)

        return {
            "features": features,
            "diagnosis_logits": None,
            "diagnosis_log_probs": diagnosis_log_probs,
            "diagnosis_probs": diagnosis_probs,
            "diagnosis_screen_probs": diagnosis_probs[:, 1:].sum(dim=1),
            "screen_logits": screen_logits,
            "screen_probs": torch.sigmoid(screen_logits),
            "abnormal_logits": abnormal_logits,
            "morph_features": morph_features,
            "evidence_features": evidence_features,
            "morph_logits": morph_logits,
            "evidence_logits": evidence_logits,
            "morph_probs": F.softmax(morph_logits.float(), dim=1),
            "evidence_probs": F.softmax(evidence_logits.float(), dim=1),
        }


def build_stage1_model(variant, model_name="caformer_s18", pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the Stage-1 backbone") from exc

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return Stage1Classifier(backbone, feature_dim, variant)


def build_external_abnormal_model(model_name="caformer_s18", pretrained=True):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the external pretraining backbone") from exc

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return ExternalAbnormalClassifier(backbone, feature_dim)


def build_tbs_semantic_model(
    model_name="caformer_s18",
    pretrained=True,
    semantic_dim=128,
):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to build the M3 backbone") from exc

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Backbone {model_name} does not expose num_features")
    return TBSSemanticClassifier(
        backbone,
        feature_dim,
        semantic_dim=semantic_dim,
    )
