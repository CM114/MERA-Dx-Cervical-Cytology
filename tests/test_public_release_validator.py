import tempfile
import unittest
from pathlib import Path

from tools.validate_public_release import validate_release


REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "environment.yml",
    "data/README.md",
    "data/data_dictionary.md",
    "docs/data_availability.md",
    "docs/reproducibility.md",
)


class PublicReleaseValidatorTests(unittest.TestCase):
    def make_valid_release(self, root: Path) -> None:
        for relative in REQUIRED:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("public release\n", encoding="utf-8")

    def test_accepts_required_path_safe_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_valid_release(root)
            report = validate_release(root)
            self.assertEqual(report["errors"], [])

    def test_rejects_weight_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_valid_release(root)
            (root / "weights.pt").write_bytes(b"not public")
            (root / "bad.txt").write_text("source is /mnt/private/data\n", encoding="utf-8")
            report = validate_release(root)
            self.assertTrue(any("weights.pt" in item for item in report["forbidden_files"]))
            self.assertTrue(report["absolute_path_hits"])

    def test_rejects_identifier_columns_in_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_valid_release(root)
            (root / "bad.csv").write_text("patient_id,label\nP1,HSIL\n", encoding="utf-8")
            report = validate_release(root)
            self.assertTrue(report["identifier_hits"])


if __name__ == "__main__":
    unittest.main()
