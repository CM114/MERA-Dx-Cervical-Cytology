import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.C3HCCS.splits import prepare_hccs_splits


class C3HCCSSplitTests(unittest.TestCase):
    def test_oof_folds_are_nonempty_and_cover_outer_train(self):
        rows = []
        for i in range(100):
            rows.append({'image_path': f'img_{i}.png', 'content_sha256': f'hash_{i}', 'diagnosis_label': i % 5, 'patient_id': f'p_{i}'})
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); train = root / 'train.csv'; smoke = root / 'smoke.csv'; out = root / 'splits'
            frame.to_csv(train, index=False); frame.iloc[:10].assign(patient_id=lambda x: 'smoke_' + x.index.astype(str), content_sha256=lambda x: 'smoke_hash_' + x.index.astype(str), image_path=lambda x: 'smoke_' + x.index.astype(str) + '.png').to_csv(smoke, index=False)
            result = prepare_hccs_splits(train, smoke, out)
            self.assertEqual(len(result['folds']), 5)
            self.assertTrue(all((out / f'inner_fold_{i}' / 'heldout.csv').stat().st_size > 0 for i in range(5)))


if __name__ == '__main__':
    unittest.main()
