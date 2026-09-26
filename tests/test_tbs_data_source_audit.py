import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from experiments.tbs.data_source_audit import (
    AuditLimits,
    MemberRecord,
    SourceSpec,
    analyze_split_overlap,
    classify_candidate_uses,
    detect_grouping_evidence,
    infer_member_structure,
    inspect_image_sample,
    iter_source_members,
    preview_metadata,
    resolve_sources,
    score_source,
    select_image_samples,
    summarize_structure,
)


class SourceResolutionTests(unittest.TestCase):
    def test_default_whitelist_excludes_projects_and_feature_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_wsiEvaluationData").mkdir()
            (root / "features").mkdir()
            (root / "xuwenjunCervix").mkdir()

            sources = resolve_sources(root, requested_sources=None)

            self.assertEqual(
                [item.name for item in sources],
                ["01_wsiEvaluationData"],
            )

    def test_explicit_source_must_remain_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()

            with self.assertRaisesRegex(ValueError, "under audit root"):
                resolve_sources(root, [outside])

    def test_missing_explicit_source_is_recorded_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            sources = resolve_sources(root, ["missing.zip"])

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].source_type, "missing")


class MemberInventoryTests(unittest.TestCase):
    def test_directory_inventory_is_sorted_and_skips_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            (corpus / "train" / "LSIL").mkdir(parents=True)
            (corpus / "train" / "LSIL" / "b.jpg").write_bytes(b"b")
            (corpus / "train" / "LSIL" / "a.jpg").write_bytes(b"a")
            target = root / "outside.jpg"
            target.write_bytes(b"outside")
            try:
                (corpus / "linked.jpg").symlink_to(target)
            except OSError:
                pass

            source = SourceSpec("corpus", corpus, "directory", True)
            records, summary = iter_source_members(source, AuditLimits())

            self.assertEqual(
                [row.member_path for row in records],
                ["train/LSIL/a.jpg", "train/LSIL/b.jpg"],
            )
            self.assertEqual(summary["inventory_scope"], "complete")
            self.assertFalse(summary["truncated"])

    def test_zip_inventory_stops_at_cap_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "ClassDataset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for index in range(4):
                    handle.writestr(
                        f"train/LSIL/{index}.jpg",
                        f"image-{index}".encode("ascii"),
                    )

            source = SourceSpec(archive.name, archive, "zip", True)
            records, summary = iter_source_members(
                source,
                AuditLimits(max_entries_per_source=2),
            )

            self.assertEqual(len(records), 2)
            self.assertTrue(summary["truncated"])
            self.assertEqual(summary["inventory_scope"], "capped")
            self.assertFalse((root / "train").exists())

    def test_tar_inventory_reads_headers_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "cells.tar"
            with tarfile.open(archive, "w") as handle:
                payload = b"image"
                info = tarfile.TarInfo("val/HSIL/a.jpg")
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))

            source = SourceSpec(archive.name, archive, "tar", True)
            records, summary = iter_source_members(source, AuditLimits())

            self.assertEqual(records[0].member_path, "val/HSIL/a.jpg")
            self.assertEqual(records[0].size_bytes, 5)
            self.assertFalse((root / "val").exists())
            self.assertEqual(summary["status"], "ok")


class StructureInferenceTests(unittest.TestCase):
    def test_structure_inference_finds_split_class_and_five_class_mapping(self):
        record = MemberRecord(
            "seven",
            "root/train/ASC-H/12_0.1_0.2_0.3_0.4.jpg",
            100,
            ".jpg",
            False,
            "0012abcd",
            "zip",
        )

        inferred = infer_member_structure(record)

        self.assertEqual(inferred["split"], "train")
        self.assertEqual(inferred["label_candidate"], "ASC-H")
        self.assertEqual(inferred["five_class_candidate"], "ASC-H")
        self.assertEqual(inferred["attribute_count"], 4)

    def test_normal_maturity_maps_only_as_structure_inferred_candidate(self):
        record = MemberRecord(
            "seven",
            "root/val/basal/7.jpg",
            100,
            ".jpg",
            False,
            None,
            "zip",
        )

        inferred = infer_member_structure(record)

        self.assertEqual(inferred["five_class_candidate"], "Normal")
        self.assertEqual(inferred["maturity_candidate"], "basal")
        self.assertEqual(inferred["evidence_level"], "STRUCTURE_INFERRED")

    def test_summary_keeps_unknown_paths_out_of_class_counts(self):
        records = [
            MemberRecord("source", "train/HSIL/a.jpg", 1, ".jpg", False, None, "directory"),
            MemberRecord("source", "notes/misc/a.jpg", 1, ".jpg", False, None, "directory"),
        ]

        summary = summarize_structure(records)

        self.assertEqual(summary["class_counts"], {"train|HSIL": 1})
        self.assertEqual(summary["unknown_label_members"], 1)


class MetadataEvidenceTests(unittest.TestCase):
    def test_metadata_preview_reads_only_limit_and_reports_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "metadata.zip"
            payload = b"patient_id,wsi_id,label\nP1,W1,HSIL\n" + b"x" * 10000
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("metadata.csv", payload)
            source = SourceSpec(archive.name, archive, "zip", True)
            limits = AuditLimits(max_metadata_preview_bytes=64)
            records, _ = iter_source_members(source, limits)

            rows = preview_metadata(source, records, limits)

            self.assertEqual(rows[0]["bytes_read"], 64)
            self.assertEqual(
                rows[0]["field_candidates"],
                ["patient_id", "wsi_id", "label"],
            )
            self.assertNotIn("P1", rows[0]["preview"])

    def test_filename_prefix_is_candidate_not_verified_grouping(self):
        result = detect_grouping_evidence(
            metadata_fields=[],
            member_paths=["train/HSIL/case12_cell4.jpg"],
        )

        self.assertFalse(result["patient_wsi_grouping_verified"])
        self.assertTrue(result["grouping_candidate"])
        self.assertEqual(result["independent_unit"], "unknown")

    def test_explicit_patient_and_wsi_fields_verify_grouping(self):
        result = detect_grouping_evidence(
            metadata_fields=["patient_id", "slide_id", "diagnosis"],
            member_paths=[],
        )

        self.assertTrue(result["patient_wsi_grouping_verified"])
        self.assertEqual(result["independent_unit"], "patient_or_wsi")


class ImageSamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_stratified(self):
        records = []
        for split in ("train", "val"):
            for label in ("LSIL", "HSIL"):
                for index in range(3):
                    records.append(
                        MemberRecord(
                            "source",
                            f"{split}/{label}/{index}.jpg",
                            10,
                            ".jpg",
                            False,
                            None,
                            "directory",
                        )
                    )

        first = select_image_samples(records, maximum=4)
        second = select_image_samples(list(reversed(records)), maximum=4)

        self.assertEqual(
            [row.member_path for row in first],
            [row.member_path for row in second],
        )
        strata = {
            (
                infer_member_structure(row)["split"],
                infer_member_structure(row)["label_candidate"],
            )
            for row in first
        }
        self.assertEqual(len(strata), 4)

    def test_image_inspection_supports_zip_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "cells.zip"
            buffer = io.BytesIO()
            Image.new("RGB", (37, 29), (180, 80, 150)).save(buffer, format="PNG")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train/HSIL/a.png", buffer.getvalue())
            source = SourceSpec(archive.name, archive, "zip", True)
            records, _ = iter_source_members(source, AuditLimits())

            result = inspect_image_sample(source, records[0], AuditLimits())

            self.assertEqual((result["width"], result["height"]), (37, 29))
            self.assertEqual(result["mode"], "RGB")
            self.assertEqual(result["sample_status"], "ok")
            self.assertFalse((root / "train").exists())


class SplitOverlapTests(unittest.TestCase):
    def test_zip_crc_overlap_detects_cross_split_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "duplicates.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train/LSIL/a.jpg", b"same-bytes")
                handle.writestr("val/LSIL/b.jpg", b"same-bytes")
            source = SourceSpec(archive.name, archive, "zip", True)
            records, _ = iter_source_members(source, AuditLimits())
            structures = [infer_member_structure(row) for row in records]

            result = analyze_split_overlap(
                source,
                records,
                structures,
                AuditLimits(),
            )

            self.assertEqual(result["cross_split_exact_duplicate_keys"], 1)
            self.assertEqual(
                result["exact_overlap_scope"],
                "complete_zip_crc_size",
            )

    def test_missing_group_ids_sets_pseudoreplication_risk(self):
        records = [
            MemberRecord("source", "train/HSIL/a.jpg", 1, ".jpg", False, None, "directory"),
            MemberRecord("source", "val/HSIL/b.jpg", 1, ".jpg", False, None, "directory"),
        ]
        source = SourceSpec("source", Path("source"), "directory", False)

        result = analyze_split_overlap(
            source,
            records,
            [infer_member_structure(row) for row in records],
            AuditLimits(),
        )

        self.assertEqual(result["independent_unit"], "unknown")
        self.assertTrue(result["pseudoreplication_risk"])
        self.assertEqual(result["exact_overlap_scope"], "partial")


class SourceScoringTests(unittest.TestCase):
    def test_unknown_evidence_does_not_score_as_pass(self):
        result = score_source(
            {
                "five_class_match": "unknown",
                "technical_usable": True,
            }
        )

        self.assertEqual(
            result["components"]["five_class_label_match"],
            0,
        )
        self.assertLess(result["verified_weight"], 100)

    def test_wsi_with_grouping_is_external_evaluation_not_cell_training(self):
        evidence = {
            "label_granularity": "wsi",
            "patient_wsi_grouping_verified": True,
            "wsi_context_verified": True,
            "external_source_verified": True,
            "technical_usable": True,
        }

        result = score_source(evidence)

        self.assertIn("external_evaluation", result["recommended_uses"])
        self.assertIn("wsi_context_learning", result["recommended_uses"])
        self.assertNotIn(
            "direct_five_class_training",
            result["recommended_uses"],
        )

    def test_histology_mismatch_applies_documented_deduction(self):
        result = score_source(
            {
                "task_domain": "histology",
                "technical_usable": True,
            }
        )

        self.assertEqual(
            result["deductions"]["cytology_task_mismatch"],
            -20,
        )
        self.assertIn("do_not_integrate", result["recommended_uses"])

    def test_morphology_proxy_five_class_source_gets_separate_uses(self):
        evidence = {
            "five_class_match": True,
            "label_granularity": "cell",
            "morphology_supervision_verified": True,
            "technical_usable": True,
            "provenance_verified": False,
        }

        uses = classify_candidate_uses(evidence)

        self.assertIn("diagnosis_pretraining", uses)
        self.assertIn("morphology_auxiliary", uses)
        self.assertNotIn("external_evaluation", uses)


if __name__ == "__main__":
    unittest.main()
