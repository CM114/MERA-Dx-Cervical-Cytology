"""Minimal, deterministic visual quality checks for release plots."""

from __future__ import annotations

import json
from pathlib import Path


def audit_layout(fig) -> dict[str, object]:
    """Return basic layout facts without making scientific claims."""

    return {
        "axes_count": len(fig.axes),
        "width_inches": float(fig.get_figwidth()),
        "height_inches": float(fig.get_figheight()),
    }


def print_report(report: dict[str, object]) -> None:
    print(json.dumps(report, sort_keys=True))


def render_preview(fig, output_path: str, dpi: int = 150) -> Path:
    """Render a PNG preview and return its path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return output
