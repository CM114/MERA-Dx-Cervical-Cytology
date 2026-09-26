import unittest

import pandas as pd

from experiments.tbs.nucleus_candidate_filter import (
    FilterConfig,
    build_representative_rows,
    evaluate_candidate_row,
    select_candidate_rows,
)


def metric_row(**overrides):
    row = {
        "mask_key": "group_a",
        "pair_status": "mask_ineligible",
        "source_structure": "augmented",
        "member_count": "4",
        "readable_member_count": "4",
        "complete_read": "True",
        "binary_control": "False",
        "nonzero_label_count_min": "3",
        "edge_contact_member_count": "0",
        "member0_edge_contact": "False",
        "label_component_mismatch_member_count": "0",
        "member0_label_component_match": "True",
        "foreground_transform_min_pixel_agreement": "0.995",
        "instance_region_transform_min_iou": "0.995",
        "instance_region_match_rate_min": "1.0",
        "instance_region_count_min": "3",
        "instance_region_count_max": "3",
        "source_paths": "/data/train_images/group_a_0.jpg|/data/train_images/group_a_1.jpg|/data/train_images/group_a_2.jpg|/data/train_images/group_a_3.jpg",
    }
    row.update(overrides)
    return row


class CandidateFilterTests(unittest.TestCase):
    def test_accepts_complete_nonedge_geometry_consistent_multivalue_group(self):
        accepted, reasons = evaluate_candidate_row(metric_row(), FilterConfig())

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_rejects_threshold_boundary_without_rounding_up(self):
        accepted, reasons = evaluate_candidate_row(
            metric_row(instance_region_transform_min_iou="0.989999999"),
            FilterConfig(),
        )

        self.assertFalse(accepted)
        self.assertIn("instance_region_iou_below_threshold", reasons)

    def test_rejects_edge_mismatch_and_nonmultivalue_rows_with_reasons(self):
        accepted, reasons = evaluate_candidate_row(
            metric_row(
                nonzero_label_count_min="1",
                edge_contact_member_count="1",
                label_component_mismatch_member_count="1",
            ),
            FilterConfig(),
        )

        self.assertFalse(accepted)
        self.assertEqual(
            reasons,
            [
                "not_multivalue_all_members",
                "edge_contact",
                "label_component_mismatch",
            ],
        )

    def test_selection_is_deterministic_and_preserves_rejection_reason(self):
        frame = pd.DataFrame(
            [
                metric_row(mask_key="keep"),
                metric_row(mask_key="drop", foreground_transform_min_pixel_agreement="0.98"),
            ]
        )

        accepted, rejected = select_candidate_rows(frame, FilterConfig())

        self.assertEqual(accepted["mask_key"].tolist(), ["keep"])
        self.assertEqual(rejected["mask_key"].tolist(), ["drop"])
        self.assertEqual(rejected["rejection_reasons"].tolist(), ["foreground_agreement_below_threshold"])

    def test_representatives_emit_only_member_zero(self):
        candidates = pd.DataFrame([metric_row(mask_key="group_a")])
        pairing = pd.DataFrame(
            [
                {
                    "mask_key": "group_a",
                    "member_paths": "/m/a_0.tif|/m/a_1.tif|/m/a_2.tif|/m/a_3.tif",
                    "source_paths": "/i/a_0.jpg|/i/a_1.jpg|/i/a_2.jpg|/i/a_3.jpg",
                    "source_structure": "augmented",
                }
            ]
        )

        representatives = build_representative_rows(candidates, pairing)

        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives.loc[0, "representative_member"], 0)
        self.assertEqual(representatives.loc[0, "representative_mask_path"], "/m/a_0.tif")
        self.assertEqual(representatives.loc[0, "representative_source_path"], "/i/a_0.jpg")
        self.assertEqual(representatives.loc[0, "member_count_in_group"], 4)


if __name__ == "__main__":
    unittest.main()
