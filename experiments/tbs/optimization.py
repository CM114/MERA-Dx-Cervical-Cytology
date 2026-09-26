import math
from numbers import Real


def _validate_positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def validate_backbone_lr_multiplier(value):
    value = _validate_positive_finite(value, "backbone_lr_multiplier")
    if value > 1.0:
        raise ValueError("backbone_lr_multiplier must not exceed 1.0")
    return value


def _unique_trainable(parameters):
    unique = []
    seen = set()
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if getattr(parameter, "requires_grad", False):
            unique.append(parameter)
    return unique


def build_discriminative_parameter_groups(
    model,
    base_lr,
    backbone_lr_multiplier=1.0,
):
    base_lr = _validate_positive_finite(base_lr, "base_lr")
    multiplier = validate_backbone_lr_multiplier(backbone_lr_multiplier)

    backbone = getattr(model, "backbone", None)
    if backbone is None or not callable(getattr(backbone, "parameters", None)):
        raise ValueError("model must expose a backbone with parameters")
    if not callable(getattr(model, "parameters", None)):
        raise ValueError("model must expose parameters")

    all_backbone_parameters = list(backbone.parameters())
    backbone_parameter_ids = {id(parameter) for parameter in all_backbone_parameters}
    backbone_parameters = _unique_trainable(all_backbone_parameters)
    head_parameters = _unique_trainable(
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_parameter_ids
    )

    if not head_parameters:
        raise ValueError("model has no trainable head parameters")
    if not backbone_parameters:
        raise ValueError("model has no trainable backbone parameters")

    return [
        {
            "group_name": "head",
            "params": head_parameters,
            "lr": base_lr,
        },
        {
            "group_name": "backbone",
            "params": backbone_parameters,
            "lr": base_lr * multiplier,
        },
    ]


def get_named_learning_rates(parameter_groups):
    expected_names = {"head", "backbone"}
    learning_rates = {}

    for group in parameter_groups:
        name = group.get("group_name")
        if name not in expected_names or name in learning_rates:
            raise ValueError("optimizer parameter groups must have unique head/backbone names")
        learning_rates[name] = _validate_positive_finite(group.get("lr"), f"{name}_lr")

    if set(learning_rates) != expected_names:
        raise ValueError("optimizer must contain both head and backbone groups")
    return learning_rates
