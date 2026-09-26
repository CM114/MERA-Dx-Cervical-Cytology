import unittest

from experiments.tbs.sampling import build_case_class_balanced_weights


class CaseClassBalancedSamplingTests(unittest.TestCase):
    def test_equalizes_class_mass_and_case_mass_within_each_class(self):
        records = []
        image_counts = {
            0: {"n0": 3, "n1": 1},
            1: {"a0": 2, "a1": 4},
            2: {"l0": 1, "l1": 2},
            3: {"h0": 2, "h1": 1},
            4: {"s0": 4, "s1": 2},
        }
        for label, cases in image_counts.items():
            for case_id, count in cases.items():
                records.extend(
                    {"diagnosis_label": label, "case_key": case_id}
                    for _ in range(count)
                )

        weights = build_case_class_balanced_weights(records)
        class_mass = {}
        case_mass = {}
        for record, weight in zip(records, weights):
            label = int(record["diagnosis_label"])
            case_id = record["case_key"]
            class_mass[label] = class_mass.get(label, 0.0) + weight
            case_mass[(label, case_id)] = case_mass.get((label, case_id), 0.0) + weight

        self.assertEqual(len(weights), len(records))
        self.assertTrue(all(weight > 0.0 for weight in weights))
        for label in range(1, 5):
            self.assertAlmostEqual(class_mass[label], class_mass[0])
        for label, cases in image_counts.items():
            masses = [case_mass[(label, case_id)] for case_id in cases]
            for mass in masses[1:]:
                self.assertAlmostEqual(mass, masses[0])

    def test_rejects_missing_case_identity(self):
        with self.assertRaisesRegex(ValueError, "case identity"):
            build_case_class_balanced_weights(
                [{"diagnosis_label": 0, "case_key": ""}]
            )


if __name__ == "__main__":
    unittest.main()
