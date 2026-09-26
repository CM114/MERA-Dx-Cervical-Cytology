SCHEMA_VERSION = "xudata-tbs5-v1"

DIAGNOSIS_FOLDERS = {
    "0_NIML": {
        "diagnosis_label": 0,
        "diagnosis_name": "Normal",
        "screen_label": 0,
        "morph_label": -1,
        "evidence_label": -1,
        "semantic_mask": 0,
    },
    "1_ASC-US": {
        "diagnosis_label": 1,
        "diagnosis_name": "ASC-US",
        "screen_label": 1,
        "morph_label": 0,
        "evidence_label": 0,
        "semantic_mask": 1,
    },
    "2_LSIL": {
        "diagnosis_label": 2,
        "diagnosis_name": "LSIL",
        "screen_label": 1,
        "morph_label": 0,
        "evidence_label": 1,
        "semantic_mask": 1,
    },
    "4_ASC-H": {
        "diagnosis_label": 3,
        "diagnosis_name": "ASC-H",
        "screen_label": 1,
        "morph_label": 1,
        "evidence_label": 0,
        "semantic_mask": 1,
    },
    "5_HSIL": {
        "diagnosis_label": 4,
        "diagnosis_name": "HSIL",
        "screen_label": 1,
        "morph_label": 1,
        "evidence_label": 1,
        "semantic_mask": 1,
    },
}

MATURITY_FOLDERS = {
    "0_Superficial": {"maturity_label": 0, "maturity_name": "Superficial"},
    "1_Intermediate": {"maturity_label": 1, "maturity_name": "Intermediate"},
    "2_Parabasal": {"maturity_label": 2, "maturity_name": "Parabasal"},
}

DIAGNOSIS_NAMES = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")

CSV_FIELDS = (
    "image_path",
    "split",
    "source_split",
    "diagnosis_label",
    "diagnosis_name",
    "screen_label",
    "morph_label",
    "evidence_label",
    "semantic_mask",
    "maturity_label",
    "maturity_name",
    "source_diagnosis_folder",
    "source_maturity_folder",
    "patient_id",
    "slide_id",
)


def labels_for_folders(diagnosis_folder, maturity_folder):
    if diagnosis_folder not in DIAGNOSIS_FOLDERS:
        raise ValueError(f"Unknown diagnosis folder: {diagnosis_folder}")
    if maturity_folder not in MATURITY_FOLDERS:
        raise ValueError(f"Unknown maturity folder: {maturity_folder}")

    return {
        **DIAGNOSIS_FOLDERS[diagnosis_folder],
        **MATURITY_FOLDERS[maturity_folder],
        "source_diagnosis_folder": diagnosis_folder,
        "source_maturity_folder": maturity_folder,
    }

