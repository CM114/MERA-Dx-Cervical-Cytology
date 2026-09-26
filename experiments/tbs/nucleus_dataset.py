"""Dataset joining xudata cell images with pre-extracted nucleus features.

The nucleus features are read from CSV files produced by
``extract_nucleus_features.py`` and indexed by ``sample_id`` (image stem).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class DualViewDataset:
    """Yields (image_tensor, nucleus_features_18d, diagnosis_label, sample_id).

    Parameters
    ----------
    manifest_csv : Path
        clean_v2 CSV with columns ``image_path``, ``diagnosis_label``.
    features_csv : Path
        CSV from ``extract_nucleus_features.py``, indexed by ``sample_id``.
    transform : callable
        Image transform returning a torch.Tensor (letterbox-224 eval).
    scaler : sklearn.preprocessing.StandardScaler or None
        If None, features are returned raw. Fit on train, transform on dev.
    feature_columns : list of str
        Ordered list of feature column names to extract.
    """

    def __init__(
        self,
        manifest_csv,
        features_csv,
        transform,
        scaler=None,
        feature_columns=None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.features = pd.read_csv(features_csv)
        self.transform = transform
        self.scaler = scaler

        if feature_columns is None:
            feature_columns = [
                "nucleus_area_px",
                "nucleus_area_frac",
                "nucleus_perimeter",
                "equivalent_diameter",
                "solidity",
                "eccentricity",
                "circularity",
                "component_count",
                "largest_component_area_px",
                "largest_component_area_frac",
                "edge_contact",
                "boundary_gradient_mean",
                "nucleus_r_mean",
                "nucleus_r_std",
                "nucleus_g_mean",
                "nucleus_g_std",
                "nucleus_b_mean",
                "nucleus_b_std",
            ]
        self.feature_columns = list(feature_columns)

        # Validate columns exist in features CSV
        missing = set(self.feature_columns) - set(self.features.columns)
        if missing:
            raise ValueError(
                f"Features CSV missing columns: {sorted(missing)}"
            )

        # Build sample_id index from image_path stems
        self.manifest["sample_id"] = self.manifest["image_path"].apply(
            lambda p: Path(p).stem
        )

        # Validate features exist for all samples
        feat_ids = set(self.features["sample_id"])
        manifest_ids = set(self.manifest["sample_id"])
        missing_feats = manifest_ids - feat_ids
        if missing_feats:
            raise ValueError(
                f"{len(missing_feats)} samples in manifest have no nucleus features, "
                f"e.g. {sorted(missing_feats)[:5]}"
            )

        # Index features by sample_id for fast lookup
        self._feature_index = {}
        for _, row in self.features.iterrows():
            vec = row[self.feature_columns].values.astype(np.float32)
            # Fill NaN with 0 for empty masks or missing features
            nan_mask = np.isnan(vec)
            if nan_mask.any():
                vec[nan_mask] = 0.0
            self._feature_index[row["sample_id"]] = vec

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        sample_id = row["sample_id"]

        # Load and transform image
        with Image.open(row["image_path"]) as img:
            img_tensor = self.transform(img.convert("RGB"))

        # Look up nucleus features
        nucleus_vec = self._feature_index[sample_id].copy()

        # Apply scaler if fitted
        if self.scaler is not None:
            nucleus_vec = self.scaler.transform(nucleus_vec.reshape(1, -1)).ravel()

        label = int(row["diagnosis_label"])

        return img_tensor, nucleus_vec.astype(np.float32), label, sample_id


def fit_scaler(features_csv, feature_columns=None):
    """Fit StandardScaler on feature columns from a features CSV.

    Parameters
    ----------
    features_csv : Path
        Train set feature CSV.
    feature_columns : list or None

    Returns
    -------
    sklearn.preprocessing.StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    if feature_columns is None:
        feature_columns = [
            "nucleus_area_px",
            "nucleus_area_frac",
            "nucleus_perimeter",
            "equivalent_diameter",
            "solidity",
            "eccentricity",
            "circularity",
            "component_count",
            "largest_component_area_px",
            "largest_component_area_frac",
            "edge_contact",
            "boundary_gradient_mean",
            "nucleus_r_mean",
            "nucleus_r_std",
            "nucleus_g_mean",
            "nucleus_g_std",
            "nucleus_b_mean",
            "nucleus_b_std",
        ]

    df = pd.read_csv(features_csv)
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Features CSV missing columns: {sorted(missing)}")

    data = df[feature_columns].values.astype(np.float64)
    nan_mask = np.isnan(data)
    if nan_mask.any():
        data[nan_mask] = 0.0

    scaler = StandardScaler()
    scaler.fit(data)
    return scaler
