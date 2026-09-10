"""Persisting the aggregated cells.

Parquet rather than CSV: it round-trips the monthly ``Period`` columns and the
categorical dtypes without a schema of its own, and the table is large enough that
the difference in size and load time is worth having.

Only the cells are stored. An intermediate loan-month panel used to be written here
too, from a single quarter; it stopped being read once every command took the
aggregated path, and a saved artefact nothing loads is a way to end up fitting last
week's data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from creditsurv.config import processed_dir

if TYPE_CHECKING:
    from pathlib import Path

CELLS_FILE = "cells.parquet"


def cells_path() -> Path:
    return processed_dir() / CELLS_FILE


def save_cells(cells: pd.DataFrame) -> Path:
    """Persist the aggregated cells."""
    path = cells_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(path)
    return path


def load_cells() -> pd.DataFrame:
    """Read the aggregated cells, or say how to build them."""
    path = cells_path()
    if not path.exists():
        message = (
            f"No aggregated cells at {path}. Build them first:\n"
            "  uv run creditsurv ingest\n"
            "  uv run creditsurv aggregate"
        )
        raise FileNotFoundError(message)
    return pd.read_parquet(path)
