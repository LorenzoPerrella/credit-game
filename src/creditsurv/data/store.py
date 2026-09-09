"""Persisting built panels.

Parquet rather than CSV: it round-trips the monthly ``Period`` columns and the
categorical dtypes without a schema of its own, and a loan-month panel is large
enough that the difference in size and load time is worth having.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from creditsurv.config import processed_dir

if TYPE_CHECKING:
    from pathlib import Path

PANEL_FILE = "loan_month_panel.parquet"


def panel_path() -> Path:
    return processed_dir() / PANEL_FILE


def save_panel(panel: pd.DataFrame) -> Path:
    path = panel_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return path


def load_panel() -> pd.DataFrame:
    path = panel_path()
    if not path.exists():
        message = (
            f"No panel at {path}. Build one first:\n"
            "  uv run creditsurv build-data --source synthetic"
        )
        raise FileNotFoundError(message)
    return pd.read_parquet(path)
