import numpy as np


def test_inverse_sqrt_sampling_weights_downweight_majority_class():
    from experiments.train_cric_meradx_balanced import build_sampling_weights

    labels = np.array([0] * 6 + [1] * 1 + [2] * 2 + [3] * 2 + [4] * 3)
    weights, class_weights, counts = build_sampling_weights(labels, n_classes=5)

    assert counts.tolist() == [6, 1, 2, 2, 3]
    assert class_weights[0] < class_weights[1]
    assert class_weights[1] > class_weights[4]
    assert np.allclose(weights, class_weights[labels])


def test_balanced_benchmark_exposes_the_same_sampling_rule():
    from experiments.train_cric_balanced_benchmark import build_sampling_weights

    labels = np.array([0] * 4 + [1] * 1 + [2] * 2 + [3] * 2 + [4] * 1)
    weights, class_weights, counts = build_sampling_weights(labels, n_classes=5)

    assert counts.tolist() == [4, 1, 2, 2, 1]
    assert np.allclose(weights, class_weights[labels])
    assert class_weights[1] == class_weights[4]
