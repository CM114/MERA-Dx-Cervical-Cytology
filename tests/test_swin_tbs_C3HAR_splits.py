import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.prepare_swin_tbs_C3HAR_splits import prepare_splits


class C3HARSplitsTests(unittest.TestCase):
    def test_train_calibration_and_smoke_are_group_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            train = pd.DataFrame({'image_path':[f'a/{i}.jpg' for i in range(20)], 'diagnosis_label':[i%5 for i in range(20)], 'patient_id':[f'p{i}' for i in range(20)]})
            smoke = pd.DataFrame({'image_path':['b/0.jpg'], 'diagnosis_label':[0], 'patient_id':['s0']})
            train_path, smoke_path = root/'train.csv', root/'smoke.csv'
            train.to_csv(train_path, index=False); smoke.to_csv(smoke_path, index=False)
            result = prepare_splits(train_path, smoke_path, root/'out')
            self.assertIn('calibration_csv', result)
            self.assertTrue(Path(result['calibration_csv']).is_file())
            self.assertTrue(Path(result['untouched_smoke_csv']).is_file())


if __name__ == '__main__':
    unittest.main()
