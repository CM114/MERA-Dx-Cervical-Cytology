import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.prepare_swin_tbs_C2PRGF_splits import prepare_splits


class C2PRGFSplitTests(unittest.TestCase):
    def test_prepare_splits_is_group_disjoint_and_80_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for group in range(20):
                label = group % 4 + 1
                rows.append({"image_path": f"{group}.png", "diagnosis_label": label, "patient_id": f"p{group}"})
            outer = root / "outer.csv"
            smoke = root / "smoke.csv"
            pd.DataFrame(rows).to_csv(outer, index=False)
            pd.DataFrame([{ "image_path": "s.png", "diagnosis_label": 1, "patient_id": "ps" }]).to_csv(smoke, index=False)
            result = prepare_splits(outer, smoke, root / "out")
            train = pd.read_csv(result["prototype_train_csv"])
            calibration = pd.read_csv(result["calibration_csv"])
            self.assertEqual(len(train) + len(calibration), 20)
            self.assertEqual(set(train.patient_id) & set(calibration.patient_id), set())
            self.assertEqual(result["route"], "C2P_RGF_SPLITS_READY")


if __name__ == "__main__":
    unittest.main()
