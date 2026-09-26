"""Path helpers that keep dataset locations outside the public repository."""

from __future__ import annotations

import os
from pathlib import Path


def get_data_root(cli_value: str | None = None) -> Path:
    """Return the configured dataset root in CLI, environment, or local order."""

    raw_value = cli_value or os.environ.get("MERADX_DATA_ROOT")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "local").resolve()


def resolve_data_path(relative_path: str, data_root: Path | None = None) -> Path:
    """Resolve a path below the data root and reject directory traversal."""

    root = (data_root or get_data_root()).resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"data path escapes configured root: {relative_path}")
    return candidate
