from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # The read-only manifest tools remain usable off-server.
    class Dataset:
        pass


REQUIRED_COLUMNS = ("image_path", "label", "label_name")


class ExternalAbnormalDataset(Dataset):
    """Four-class abnormal pretraining dataset.

    This loader intentionally has a smaller contract than XUDataTBS5Dataset and
    must not be used for target TBS5 evaluation.
    """

    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.csv_path}")
        frame = pd.read_csv(self.csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"Manifest is empty: {self.csv_path}")
        frame["label"] = frame["label"].astype(int)
        if not frame["label"].between(0, 3).all():
            raise ValueError("External abnormal labels must be in [0, 3]")
        self.records = frame.to_dict("records")
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(record["label"]),
            "label_name": str(record["label_name"]),
            "image_path": str(image_path),
        }
