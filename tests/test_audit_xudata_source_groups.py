import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.audit_xudata_source_groups import (
    parse_source_candidate,
    run_source_group_audit,
)
from experiments.tbs.labels import CSV_FIELDS, labels_for_folders


class SourceCandidateParserTests(unittest.TestCase):
    def test_parses_and_normalizes_dated_identifiers(self):
        first = parse_source_candidate(
            "/data/2021-07-05 - YZ_201702320 - 12482_21092_65_66.jpg"
        )
        second = parse_source_candidate(
            "/data/2020-12-04 - 201702320 - 12482_21092_65_66.jpg"
        )
        self.assertEqual(first["candidate_group_id"], "source:201702320")
        self.assertEqual(first["candidate_group_id"], second["candidate_group_id"])
        self.assertEqual(first["confidence"], "high")

    def test_parses_prefix_before_diagnosis_or_position_marker(self):
        cases = {
            "SZZX_00129_ASCHposition_12_20index_3.jpg": "source:SZZX_00129",
            "ZL_2_06_HSILposition_10_12index_4.jpg": "source:ZL_2_06",
            "MLY_2012963position_24576_15360index_0.jpg": "source:MLY_2012963",
            "7001034018position_17408_17408position_554_563.jpg": "source:7001034018",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                parsed = parse_source_candidate(filename)
                self.assertEqual(parsed["candidate_group_id"], expected)
                self.assertEqual(parsed["confidence"], "medium")

    def test_keeps_numeric_three_token_rule_low_confidence(self):
        parsed = parse_source_candidate("0_17125_121.jpg")
        self.assertEqual(parsed["candidate_group_id"], "numeric3:0")
        self.assertEqual(parsed["confidence"], "low")

    def test_strips_date_suffix_from_dated_identifier(self):
        parsed = parse_source_candidate(
            "2021-01-18 - 201702218_2020_12_24 - 1_2_3_4.jpg"
        )
        self.assertEqual(parsed["candidate_group_id"], "source:201702218")
        self.assertEqual(parsed["confidence"], "high")

    def test_leaves_coordinate_only_names_unresolved(self):
        parsed = parse_source_candidate("31879_31975_2560_2672.jpg")
        self.assertEqual(parsed["candidate_group_id"], "")
        self.assertEqual(parsed["confidence"], "unresolved")


class SourceGroupAuditTests(unittest.TestCase):
    @staticmethod
    def make_row(path, split, source_split, diagnosis):
        row = {
            "image_path": path,
            "split": split,
            "source_split": source_split,
            **labels_for_folders(diagnosis, "2_Parabasal"),
            "patient_id": "",
            "slide_id": "",
            "content_sha256": f"hash-{Path(path).stem}",
        }
        return row

    def test_reports_cross_split_candidates_without_assigning_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "clean"
            out_dir = root / "audit"
            csv_dir.mkdir()
            rows = [
                self.make_row(
                    "/train/2021-07-05 - YZ_201702320 - 1_2_3_4.jpg",
                    "train",
                    "train",
                    "4_ASC-H",
                ),
                self.make_row(
                    "/val/201702320position_5_6index_7.jpg",
                    "test",
                    "val",
                    "5_HSIL",
                ),
                self.make_row(
                    "/train/SZZX_00129_ASCHposition_1_2index_0.jpg",
                    "train",
                    "train",
                    "4_ASC-H",
                ),
                self.make_row(
                    "/train/SZZX_00129_LSILposition_3_4index_1.jpg",
                    "dev",
                    "train",
                    "2_LSIL",
                ),
                self.make_row("/train/0_11166_139.jpg", "train", "train", "0_NIML"),
                self.make_row("/val/0_17125_121.jpg", "test", "val", "0_NIML"),
                self.make_row(
                    "/train/31879_31975_2560_2672.jpg",
                    "calibration",
                    "train",
                    "1_ASC-US",
                ),
            ]
            fields = (*CSV_FIELDS, "content_sha256")
            for split in ("train", "dev", "calibration", "test"):
                with (csv_dir / f"{split}_xudata_tbs5_clean_v2.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(row for row in rows if row["split"] == split)

            report = run_source_group_audit(csv_dir, out_dir)
            self.assertEqual(report["total_images"], 7)
            self.assertEqual(report["confidence_counts"]["unresolved"], 1)
            self.assertEqual(report["cross_split_candidate_groups"], 3)
            self.assertEqual(report["actionable_cross_split_candidate_groups"], 2)
            self.assertEqual(report["actionable_groups_touching_test"], 1)

            candidates = pd.read_csv(
                out_dir / "source_group_candidates.csv", keep_default_na=False
            )
            self.assertTrue((candidates["patient_id"] == "").all())
            self.assertTrue((candidates["slide_id"] == "").all())
            cross_split = pd.read_csv(out_dir / "cross_split_source_groups.csv")
            self.assertEqual(len(cross_split), 3)
            merged = cross_split.loc[
                cross_split["candidate_group_id"] == "source:201702320"
            ].iloc[0]
            self.assertEqual(merged["confidence"], "high")
            self.assertIn("dated_identifier", merged["parser_rule"])
            self.assertIn("prefix_before_position", merged["parser_rule"])
            saved = json.loads(
                (out_dir / "source_group_audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
