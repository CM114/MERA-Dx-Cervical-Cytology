from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from experiments.tbs.image_transforms import LetterboxResize


REQUIRED_COLUMNS = (
    "image_path",
    "diagnosis_label",
    "diagnosis_name",
    "screen_label",
    "morph_label",
    "evidence_label",
    "semantic_mask",
    "maturity_label",
    "maturity_name",
)

INPUT_MODES = ("crop", "letterbox")


class XUDataTBS5Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Manifest {self.csv_path} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"Manifest is empty: {self.csv_path}")

        frame["diagnosis_label"] = frame["diagnosis_label"].astype(int)
        frame["screen_label"] = frame["screen_label"].astype(int)
        frame["morph_label"] = frame["morph_label"].astype(int)
        frame["evidence_label"] = frame["evidence_label"].astype(int)
        frame["semantic_mask"] = frame["semantic_mask"].astype(int)
        frame["maturity_label"] = frame["maturity_label"].astype(int)
        expected_semantic_mask = (frame["diagnosis_label"] > 0).astype(int)
        if not frame["semantic_mask"].equals(expected_semantic_mask):
            raise ValueError(
                f"Manifest {self.csv_path} has semantic_mask inconsistent with diagnosis"
            )
        normal_mask = frame["semantic_mask"] == 0
        if not (
            (frame.loc[normal_mask, "morph_label"] == -1).all()
            and (frame.loc[normal_mask, "evidence_label"] == -1).all()
        ):
            raise ValueError(
                f"Manifest {self.csv_path} must mask Normal semantic labels with -1"
            )
        abnormal_mask = ~normal_mask
        if not (
            frame.loc[abnormal_mask, "morph_label"].isin([0, 1]).all()
            and frame.loc[abnormal_mask, "evidence_label"].isin([0, 1]).all()
        ):
            raise ValueError(
                f"Manifest {self.csv_path} has invalid abnormal semantic labels"
            )
        self.records = frame.to_dict("records")
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image_path = Path(row["image_path"])
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "diagnosis_label": int(row["diagnosis_label"]),
            "screen_label": int(row["screen_label"]),
            "morph_label": int(row["morph_label"]),
            "evidence_label": int(row["evidence_label"]),
            "semantic_mask": int(row["semantic_mask"]),
            "maturity_label": int(row["maturity_label"]),
            "diagnosis_name": str(row["diagnosis_name"]),
            "maturity_name": str(row["maturity_name"]),
            "image_path": str(image_path),
        }


def build_transforms(img_size, input_mode="crop"):
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"Unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}"
        )

    if input_mode == "letterbox":
        train_geometry = LetterboxResize(img_size, fill="border_median")
        eval_geometry = LetterboxResize(img_size, fill="border_median")
    else:
        train_geometry = transforms.RandomResizedCrop(
            img_size,
            scale=(0.75, 1.0),
        )
        eval_geometry = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
            ]
        )

    train_transform = transforms.Compose(
        [
            train_geometry,
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            eval_geometry,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform
