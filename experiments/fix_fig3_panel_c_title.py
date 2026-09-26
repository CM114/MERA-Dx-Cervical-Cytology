#!/usr/bin/env python3
"""Correct a stale panel-C title in the manuscript diagnostic figure.

The underlying matrices are unchanged; only the accidental ``Bank Sector``
label is replaced with a domain-neutral cervical-cytology label.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE = Path(__file__).resolve().parents[1] / "data" / "local" / "fig3_diagnostic_curves_matrices.png"
BACKUP = SOURCE.with_name("fig3_diagnostic_curves_matrices_before_title_fix.png")


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    if not BACKUP.exists():
        image.save(BACKUP)

    draw = ImageDraw.Draw(image)
    # The stale second title line is confined to panel C; leave panel D intact.
    draw.rectangle((120, 810, 1315, 965), fill="white")
    font = ImageFont.load_default()
    label = "(c) Five-class confusion matrices"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    x = 710 - (right - left) / 2
    draw.text((x, 818), label, font=font, fill="#263238")
    image.save(SOURCE)


if __name__ == "__main__":
    main()
