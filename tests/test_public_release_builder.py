import json
import tempfile
import unittest
from pathlib import Path

from tools.build_public_release import build_release, write_build_manifest


class PublicReleaseBuilderTests(unittest.TestCase):
    def test_copies_manifest_files_and_writes_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "release"
            (source / "experiments").mkdir(parents=True)
            (source / "tests").mkdir(parents=True)
            (source / "weights").mkdir(parents=True)
            (source / "archive" / "experiments").mkdir(parents=True)
            (source / "experiments" / "train.py").write_text("print('ok')\n", encoding="utf-8")
            (source / "tests" / "test_train.py").write_text("pass\n", encoding="utf-8")
            (source / "archive" / "experiments" / "old.py").write_text("print('old')\n", encoding="utf-8")
            (source / "weights" / "model.pt").write_bytes(b"weights")
            manifest = root / "release_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "include": ["experiments/*.py", "tests/*.py"],
                        "exclude": ["**/*.pt"],
                    }
                ),
                encoding="utf-8",
            )

            copied = build_release(source, destination, manifest)

            self.assertEqual(
                {path.relative_to(destination.resolve()).as_posix() for path in copied},
                {"experiments/train.py", "tests/test_train.py"},
            )
            self.assertTrue((destination / "docs" / "build_manifest.json").is_file())
            self.assertFalse((destination / "weights" / "model.pt").exists())

            (destination / "experiments" / "train.py").write_text("print('changed')\n", encoding="utf-8")
            refreshed = write_build_manifest(destination)
            self.assertEqual(refreshed["file_count"], 2)
            self.assertFalse((destination / "archive").exists())


if __name__ == "__main__":
    unittest.main()
