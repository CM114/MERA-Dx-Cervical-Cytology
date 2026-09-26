"""Deterministic sampling weights for case-aware replacement pilots."""

from __future__ import annotations

from collections import defaultdict


def build_case_class_balanced_weights(records):
    """Return per-image weights with equal class and within-class case mass.

    Every diagnosis class receives the same total mass. Within a class, every
    case receives the same mass, and that case mass is distributed uniformly
    over its images.
    """
    if not records:
        raise ValueError("records must not be empty")

    by_class_case = defaultdict(list)
    for index, record in enumerate(records):
        try:
            label = int(record["diagnosis_label"])
            case_key = str(record["case_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("records must contain diagnosis_label and case identity") from exc
        if label < 0 or label > 4:
            raise ValueError(f"diagnosis_label must be in [0, 4], got {label}")
        if not case_key:
            raise ValueError("records must contain a non-empty case identity")
        by_class_case[label, case_key].append(index)

    labels = sorted({label for label, _ in by_class_case})
    if labels != [0, 1, 2, 3, 4]:
        raise ValueError("records must contain all five diagnosis classes")

    class_count = len(labels)
    weights = [0.0] * len(records)
    for label in labels:
        cases = sorted(case_key for case_label, case_key in by_class_case if case_label == label)
        class_mass = 1.0 / class_count
        case_mass = class_mass / len(cases)
        for case_key in cases:
            indices = by_class_case[label, case_key]
            image_weight = case_mass / len(indices)
            for index in indices:
                weights[index] = image_weight
    return weights
