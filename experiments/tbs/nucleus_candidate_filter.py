"""Pure conservative filtering for audited nucleus-instance candidate groups."""

from dataclasses import dataclass
import math
import re

import pandas as pd


MEMBER_PATTERN = re.compile(r"_(?P<member>[0-3])(?:\.[^.]+)?$")


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for a provisional, auxiliary-only candidate set."""

    foreground_min_agreement: float = 0.99
    instance_min_iou: float = 0.99
    instance_min_match_rate: float = 1.0

    def __post_init__(self):
        for name in (
            "foreground_min_agreement",
            "instance_min_iou",
            "instance_min_match_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")


def _text(value):
    if value is None:
        return ""
    return str(value).strip()


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool(value):
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _pipe_values(value):
    return [item for item in _text(value).split("|") if item]


def _has_four_members(row):
    return (
        _int(row.get("member_count")) == 4
        and _int(row.get("readable_member_count")) == 4
        and _bool(row.get("complete_read"))
    )


def evaluate_candidate_row(row, config=None):
    """Return ``(accepted, reasons)`` for one metrics row."""

    config = config or FilterConfig()
    reasons = []

    if not _has_four_members(row):
        reasons.append("incomplete_or_unreadable_members")

    if _text(row.get("source_structure")) != "augmented" or len(
        _pipe_values(row.get("source_paths"))
    ) != 4:
        reasons.append("missing_four_rgb_members")

    if (
        _text(row.get("pair_status")) != "mask_ineligible"
        or _bool(row.get("binary_control"))
        or (_int(row.get("nonzero_label_count_min")) or 0) < 2
    ):
        reasons.append("not_multivalue_all_members")

    if _bool(row.get("member0_edge_contact")) or (_int(row.get("edge_contact_member_count")) or 0) != 0:
        reasons.append("edge_contact")

    if not _bool(row.get("member0_label_component_match")) or (
        (_int(row.get("label_component_mismatch_member_count")) or 0) != 0
    ):
        reasons.append("label_component_mismatch")

    if _float(row.get("foreground_transform_min_pixel_agreement")) < config.foreground_min_agreement:
        reasons.append("foreground_agreement_below_threshold")

    if _float(row.get("instance_region_transform_min_iou")) < config.instance_min_iou:
        reasons.append("instance_region_iou_below_threshold")

    if _float(row.get("instance_region_match_rate_min")) < config.instance_min_match_rate:
        reasons.append("instance_region_match_rate_below_threshold")

    region_min = _int(row.get("instance_region_count_min"))
    region_max = _int(row.get("instance_region_count_max"))
    if region_min is None or region_max is None or region_min != region_max:
        reasons.append("instance_region_count_changed")

    return not reasons, reasons


def select_candidate_rows(frame, config=None):
    """Split metrics into accepted and rejected rows deterministically."""

    config = config or FilterConfig()
    accepted_rows = []
    rejected_rows = []
    for row in frame.to_dict("records"):
        accepted, reasons = evaluate_candidate_row(row, config)
        if accepted:
            accepted_rows.append(row)
        else:
            row = dict(row)
            row["rejection_reasons"] = "|".join(reasons)
            rejected_rows.append(row)

    columns = list(frame.columns)
    if rejected_rows and "rejection_reasons" not in columns:
        columns.append("rejection_reasons")
    accepted = pd.DataFrame(accepted_rows, columns=list(frame.columns))
    rejected = pd.DataFrame(rejected_rows, columns=columns)
    if not accepted.empty:
        accepted = accepted.sort_values("mask_key", kind="stable").reset_index(drop=True)
    if not rejected.empty:
        rejected = rejected.sort_values("mask_key", kind="stable").reset_index(drop=True)
    return accepted, rejected


def _member_paths(value):
    values = _pipe_values(value)
    indexed = {}
    unindexed = []
    for value in values:
        match = MEMBER_PATTERN.search(value)
        if match:
            indexed[int(match.group("member"))] = value
        else:
            unindexed.append(value)
    for index, value in enumerate(unindexed):
        indexed.setdefault(index, value)
    return [indexed[index] for index in sorted(indexed)]


def _source_scope(path):
    parts = {part.lower() for part in _text(path).replace("\\", "/").split("/")}
    if "train_images" in parts:
        return "external_train"
    if "test_images" in parts:
        return "external_test"
    return "unknown"


def build_representative_rows(candidates, pairing_frame):
    """Join accepted groups to pairing rows and emit member 0 only."""

    if pairing_frame["mask_key"].astype(str).duplicated().any():
        raise ValueError("Pairing frame contains duplicate mask_key rows")
    pairing = pairing_frame.set_index("mask_key", drop=False)
    output = []
    for row in candidates.to_dict("records"):
        key = row["mask_key"]
        if key not in pairing.index:
            raise ValueError(f"Accepted mask_key is missing from pairing frame: {key}")
        pair = pairing.loc[key].to_dict()
        mask_paths = _member_paths(pair.get("member_paths"))
        source_paths = _member_paths(pair.get("source_paths"))
        if len(mask_paths) != 4 or len(source_paths) != 4:
            raise ValueError(f"Accepted group does not have four members: {key}")
        output.append(
            {
                "mask_key": key,
                "representative_member": 0,
                "representative_mask_path": mask_paths[0],
                "representative_source_path": source_paths[0],
                "representative_source_scope": _source_scope(source_paths[0]),
                "member_count_in_group": 4,
                "all_member_mask_paths": "|".join(mask_paths),
                "all_member_source_paths": "|".join(source_paths),
            }
        )
    return pd.DataFrame(
        output,
        columns=(
            "mask_key",
            "representative_member",
            "representative_mask_path",
            "representative_source_path",
            "representative_source_scope",
            "member_count_in_group",
            "all_member_mask_paths",
            "all_member_source_paths",
        ),
    ).sort_values("mask_key", kind="stable").reset_index(drop=True)
