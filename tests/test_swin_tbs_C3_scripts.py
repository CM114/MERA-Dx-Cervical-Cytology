import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.C3.protocol import C3_CONFIG
from experiments.fit_swin_tbs_C3_risk import fit_oof_risk
from experiments.evaluate_swin_tbs_C3_outer import evaluate_outer


class C3ScriptContractTests(unittest.TestCase):
    def test_route_names_are_locked(self):
        self.assertEqual(C3_CONFIG["route_ready"], "C3_RISK_HEADS_READY")
        self.assertEqual(C3_CONFIG["route_outer"], "C3_OUTER_EVALUATION_READY")

    def test_oof_fit_and_outer_evaluation_preserve_c1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            risk_root = root / "risk_splits"
            for fold in range(3):
                d = risk_root / f"risk_fold_{fold}"
                d.mkdir(parents=True)
                labels = np.asarray([1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 3])
                probs = np.full((len(labels), 5), 0.01)
                for i, label in enumerate(labels):
                    pred = int(label if i % 3 else (1 if label != 1 else 2))
                    probs[i, pred] = 0.96
                np.savez_compressed(
                    d / "heldout.npz", labels=labels, sample_ids=np.asarray([f"f{fold}_{i}" for i in range(len(labels))]),
                    c1_probabilities=probs, q_morph=np.tile([[.7, .3]], (len(labels), 1)), q_evidence=np.tile([[.6, .4]], (len(labels), 1)),
                    q_proto=np.tile([[.5, .2, .2, .1]], (len(labels), 1)),
                    morph_assignments=np.tile([[[.7, .3], [.4, .6]]], (len(labels), 1, 1)), evidence_assignments=np.tile([[[.6, .4], [.5, .5]]], (len(labels), 1, 1)),
                    morph_nearest_distance=np.tile([[.2, .8]], (len(labels), 1)), evidence_nearest_distance=np.tile([[.3, .7]], (len(labels), 1)),
                )
            model_dir = root / "model"
            manifest = fit_oof_risk(risk_root, model_dir)
            self.assertEqual(manifest["route"], "C3_RISK_HEADS_READY")
            outer = risk_root / "risk_fold_0" / "heldout.npz"
            out_dir = root / "outer"
            result = evaluate_outer(model_dir, outer, out_dir)
            self.assertEqual(result["route"], "C3_OUTER_EVALUATION_READY")
            self.assertTrue((out_dir / "c3_outer_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
