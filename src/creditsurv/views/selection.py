"""The record ``creditsurv select`` wrote, as views: what the selection saw and decided.

Read from the CSV files beside the selection report rather than recomputed: the correlation
and the variance inflation alone read ~60 million rows of expanded panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.reporting.selection import FITS_FILE, TABLE_FILES
from creditsurv.views.tables import View

if TYPE_CHECKING:
    from pathlib import Path

_DESCRIBED: Final[dict[str, str]] = {
    "correlation": "Exposure-weighted correlation between the continuous candidates.",
    "collinear": "Pairs correlated beyond 0.8, reported and not resolved.",
    "inflation": "Candidates removed for variance inflation above 10, in the fixed order.",
    "screening": "Each candidate beside the loan block: sign, effect, likelihood ratio.",
    "elimination": "Backward elimination, one covariate a step, and the rule that fired.",
    "stability": "Standardised effects on the training half and on each half of the book.",
    "fits": "Every model the selection estimated, its time, and whether it was cached.",
}


def selection_views(directory: Path) -> list[View]:
    """Every selection table found in ``directory``; the correlation matrix in long form."""
    views = []
    for name, filename in {**TABLE_FILES, "fits": FITS_FILE}.items():
        path = directory / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0 if name == "correlation" else None)
        if name == "correlation":
            frame = (
                frame.rename_axis("first")
                .reset_index()
                .melt(id_vars="first", var_name="second", value_name="correlation")
            )
        views.append(
            View(
                f"selection_{name}",
                f"Selection: {name}",
                _DESCRIBED[name],
                frame,
                source="selection",
            )
        )
    return views
