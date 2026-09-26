"""Export matplotlib figures without a machine-specific dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def export_figure(
    fig,
    stem: str,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    size_inches: tuple[float, float] | None = None,
    dpi: int = 600,
    grayscale_preview: bool = False,
    tight: bool = False,
    pad_inches: float = 0.06,
    **_kwargs,
) -> list[Path]:
    """Save a figure in the requested formats and return output paths."""

    del grayscale_preview
    if size_inches is not None:
        fig.set_size_inches(*size_inches)
    stem_path = Path(stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in formats:
        extension = extension.lstrip(".").lower()
        output = stem_path.with_suffix(f".{extension}")
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": pad_inches})
        fig.savefig(output, **save_kwargs)
        outputs.append(output)
    return outputs
