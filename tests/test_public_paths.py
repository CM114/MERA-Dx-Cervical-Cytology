import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.public_paths import get_data_root, resolve_data_path


class PublicPathsTests(unittest.TestCase):
    def test_cli_value_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_root = root / "cli"
            env_root = root / "env"
            with mock.patch.dict(os.environ, {"MERADX_DATA_ROOT": str(env_root)}):
                self.assertEqual(get_data_root(str(cli_root)), cli_root.resolve())

    def test_environment_value_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "env"
            with mock.patch.dict(os.environ, {"MERADX_DATA_ROOT": str(env_root)}):
                self.assertEqual(get_data_root(), env_root.resolve())

    def test_relative_data_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_data_path("../outside/file.csv", Path(tmp))


if __name__ == "__main__":
    unittest.main()
