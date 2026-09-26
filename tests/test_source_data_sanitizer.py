import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.sanitize_source_data import sanitize_csv


class SourceDataSanitizerTests(unittest.TestCase):
    def test_removes_identifier_and_path_columns_without_changing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            output = root / "sanitized.csv"
            source.write_text(
                "model,image_path,patient_id,slide_id,source,filename,macro_f1\n"
                "MERA-Dx,data/local/image.png,P1,S1,local,image.png,0.7752\n",
                encoding="utf-8",
            )

            report = sanitize_csv(source, output)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"model": "MERA-Dx", "macro_f1": "0.7752"}])
            self.assertEqual(
                report["removed_columns"],
                ["image_path", "patient_id", "slide_id", "source", "filename"],
            )
            self.assertTrue(output.with_suffix(".csv.metadata.json").is_file())
            json.loads(output.with_suffix(".csv.metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
