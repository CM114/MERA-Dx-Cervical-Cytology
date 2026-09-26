# Data dictionary

The public example schema is designed for XUData TBS5-style five-class development experiments.

| Field | Meaning | Allowed values or unit | Public-release rule |
| --- | --- | --- | --- |
| `image_path` | Path relative to the user-supplied data root | Relative path string | Synthetic examples only |
| `split` | Development partition | `train`, `dev`, `calibration`, `test` | No real row is distributed |
| `fold` | Cross-validation fold | Integer 0–4 | Fixed five-fold protocol |
| `diagnosis_label` | Numeric Bethesda diagnosis | 0–4 | Class mapping below |
| `diagnosis_name` | Bethesda diagnosis name | Class mapping below | No subject metadata |
| `source_split` | Original source partition if known | Provider-defined string | Provider metadata remains local |
| `source_group` | Group used for audit | Provider-defined string | No real group identifier |

Class mapping:

| Label | Name |
| ---: | --- |
| 0 | Normal |
| 1 | ASC-US |
| 2 | LSIL |
| 3 | ASC-H |
| 4 | HSIL |

Missing values are represented as empty fields in CSV manifests. An empty patient or slide field must not be interpreted as evidence of patient-level or slide-level independence; the current development record does not support those claims.
