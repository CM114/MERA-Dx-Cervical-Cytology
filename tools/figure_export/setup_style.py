"""Deterministic matplotlib style used by release plotting scripts."""

from __future__ import annotations

import matplotlib as mpl


def setup_style(
    journal: str = "ieee",
    lang: str = "en",
    use_sciplots: bool = False,
    constrained_layout: bool = False,
) -> None:
    """Apply a compact, dependency-free publication style."""

    del journal, lang, use_sciplots
    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": constrained_layout,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
