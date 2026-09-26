import pandas as pd


def test_aggregate_metrics_mean_and_std(tmp_path):
    from experiments.aggregate_cric_fiveclass import aggregate_fold_metrics

    frame = pd.DataFrame(
        {
            "model": ["A", "A"],
            "fold": [0, 1],
            "accuracy": [0.8, 0.9],
            "macro_f1": [0.7, 0.9],
            "macro_sensitivity": [0.6, 0.8],
            "macro_specificity": [0.9, 0.95],
            "macro_auc": [0.75, 0.85],
        }
    )
    summary = aggregate_fold_metrics(frame)
    row = summary.iloc[0]
    assert row["accuracy_mean"] == 0.85
    assert row["macro_f1_mean"] == 0.8
    assert row["fold_count"] == 2
