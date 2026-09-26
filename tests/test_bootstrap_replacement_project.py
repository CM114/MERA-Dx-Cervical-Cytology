import tempfile
import unittest
from pathlib import Path

from experiments.bootstrap_replacement_project import (
    PROJECT_DIRS,
    bootstrap_project,
)


class ReplacementProjectBootstrapTests(unittest.TestCase):
    def test_bootstrap_creates_isolated_tree_and_path_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_source = root / "raw"
            raw_source.mkdir()
            raw_file = raw_source / "keep.jpg"
            raw_file.write_bytes(b"raw")
            project = root / "xuwenjun" / "cervical_replacement_v1"

            summary = bootstrap_project(
                project_root=project,
                raw_source=raw_source,
                code_root=Path("release-code"),
            )

            self.assertEqual(summary["raw_data_modified"], False)
            self.assertEqual(raw_file.read_bytes(), b"raw")
            for relative in PROJECT_DIRS:
                self.assertTrue((project / relative).is_dir(), relative)
            self.assertEqual(
                (project / "data_sources" / "wsl_class_dataset.path").read_text(
                    encoding="utf-8"
                ),
                f"{raw_source.resolve()}\n",
            )
            self.assertTrue((project / "configs" / "dataset_contract.yaml").is_file())
            self.assertTrue((project / "scripts" / "run_wsl_audit.sh").is_file())
            self.assertIn("audit_wsl_class_dataset.py", (project / "scripts" / "run_wsl_audit.sh").read_text(encoding="utf-8"))

    def test_bootstrap_rejects_project_inside_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_source = root / "raw"
            raw_source.mkdir()

            with self.assertRaisesRegex(ValueError, "outside raw source"):
                bootstrap_project(
                    project_root=raw_source / "project",
                    raw_source=raw_source,
                    code_root=Path("/code"),
                )


if __name__ == "__main__":
    unittest.main()
