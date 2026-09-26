import json
from pathlib import Path

import pandas as pd
from PIL import Image


def test_prepare_maps_labels_excludes_scc_and_keeps_slide_groups(tmp_path):
    from experiments.prepare_cric_fiveclass import (
        CLASS_TO_INDEX,
        assign_grouped_folds,
        map_cric_label,
    )

    assert map_cric_label("Negative for intraepithelial lesion") == "Normal"
    assert map_cric_label("SCC") is None
    assert set(CLASS_TO_INDEX) == {"Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"}

    frame = pd.DataFrame(
        {
            "image_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    folded = assign_grouped_folds(frame, n_splits=5, seed=42)
    assert folded["fold"].between(0, 4).all()
    for image_id, group in folded.groupby("image_id"):
        assert group["fold"].nunique() == 1, image_id


def test_crop_center_pads_outside_image(tmp_path):
    from experiments.prepare_cric_fiveclass import crop_centered_patch

    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), (12, 34, 56)).save(source)
    patch = crop_centered_patch(Image.open(source), x=0, y=0, patch_size=8)
    assert patch.size == (8, 8)
    assert patch.getpixel((0, 0)) == (255, 255, 255)


def test_preparation_metadata_is_json_serializable(tmp_path):
    metadata = {"kept_rows": 3, "excluded_scc_rows": 1}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["kept_rows"] == 3
