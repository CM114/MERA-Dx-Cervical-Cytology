import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.figure_export.export_figure import export_figure
from tools.figure_export.setup_style import setup_style
from tools.figure_export.visual_qa import audit_layout, render_preview


class FigureExportTests(unittest.TestCase):
    def test_export_figure_writes_requested_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            export_figure(fig, str(Path(tmp) / "figure"), formats=["png", "svg"], dpi=80)
            plt.close(fig)
            self.assertTrue((Path(tmp) / "figure.png").is_file())
            self.assertTrue((Path(tmp) / "figure.svg").is_file())

    def test_setup_style_and_visual_qa_are_callable(self):
        setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
        fig, _ = plt.subplots()
        self.assertEqual(audit_layout(fig)["axes_count"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            render_preview(fig, str(Path(tmp) / "preview.png"), dpi=80)
            self.assertTrue((Path(tmp) / "preview.png").is_file())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
