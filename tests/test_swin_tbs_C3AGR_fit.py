import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.fit_swin_tbs_C3AGR_risk import fit_agr_oof


class C3AGRFitTests(unittest.TestCase):
    def test_fit_writes_gate_manifest_and_oof_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "risk"
            for fold in range(3):
                fold_dir = root / f"risk_fold_{fold}"
                fold_dir.mkdir(parents=True)
                rows = []
                labels = []
                for i in range(12):
                    pred = (i + fold) % 4 + 1
                    p = np.full(5, 0.02)
                    p[0] = 0.02
                    p[pred] = 0.80
                    p[(pred % 4) + 1] = 0.10
                    p = p / p.sum()
                    rows.append(p)
                    labels.append(pred if i % 3 else (pred % 4) + 1)
                c1 = np.asarray(rows)
                q_morph = np.tile(np.asarray([[.7, .3], [.3, .7]]), (6, 1))
                q_evidence = np.tile(np.asarray([[.6, .4], [.4, .6]]), (6, 1))
                np.savez_compressed(
                    fold_dir / "heldout.npz",
                    c1_probabilities=c1,
                    labels=np.asarray(labels),
                    sample_ids=np.asarray([f"{fold}-{i}" for i in range(len(labels))]),
                    q_morph=q_morph,
                    q_evidence=q_evidence,
                )
            out = Path(td) / "model"
            manifest = fit_agr_oof(root, out)
            self.assertIn(manifest["route"], {"C3AGR_RISK_HEAD_READY", "C3AGR_R1_FALLBACK"})
            self.assertTrue((out / "c3agr_manifest.json").is_file())
            self.assertTrue((out / "c3agr_oof_scores.csv").is_file())


if __name__ == "__main__":
    unittest.main()
