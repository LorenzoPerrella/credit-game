"""The committed format of a view: one parquet per table, and a manifest describing them.

The manifest is what the site reads first. It says what each table is, what it was computed
from and when, and it records the fit every model-based view was scored with, so that no page
can set a view of one model beside a view of another without it showing.

Two things are refused at write time rather than trusted: a column that identifies a loan --
the site is public and the dataset's terms allow aggregates only -- and a period column, which
parquet would store in a form the site's build step would have to special-case. Months are
written as ``YYYY-MM`` text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MANIFEST: Final = "manifest.json"

#: Columns no committed view may carry.
IDENTIFYING: Final[frozenset[str]] = frozenset({"loan_id", "loan_identifier"})


@dataclass(frozen=True)
class View:
    """One table behind the site, with what a reader needs to trust it."""

    name: str
    title: str
    description: str
    frame: pd.DataFrame
    #: What it was computed from: ``cells``, ``parquet``, ``fit`` or a report table.
    source: str


def _portable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    for column in out.columns:
        values = out[column]
        if (
            isinstance(values.dtype, (pd.PeriodDtype, pd.CategoricalDtype))
            or values.dtype == object
        ):
            out[column] = values.astype(str)
    return out


def write_views(views: Sequence[View], directory: Path, *, fit: str | None = None) -> Path:
    """Write every view and a manifest describing them, replacing any earlier manifest entry."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST
    manifest: dict[str, dict[str, object]] = json.loads(path.read_text()) if path.exists() else {}
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    for view in views:
        identifying = IDENTIFYING & set(map(str, view.frame.columns))
        if identifying:
            message = f"View {view.name!r} carries {sorted(identifying)}; views are aggregates."
            raise ValueError(message)
        frame = _portable(view.frame)
        frame.to_parquet(directory / f"{view.name}.parquet", index=False)
        manifest[view.name] = {
            "title": view.title,
            "description": view.description,
            "source": view.source,
            "rows": len(frame),
            "columns": list(map(str, frame.columns)),
            "fit": fit if view.source == "fit" else None,
            "generated": generated,
        }
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    return path


def load_manifest(directory: Path) -> dict[str, dict[str, object]]:
    path = directory / MANIFEST
    if not path.exists():
        return {}
    loaded: dict[str, dict[str, object]] = json.loads(path.read_text())
    return loaded


def load_view(name: str, directory: Path) -> pd.DataFrame:
    """A committed view, or an error naming the command that writes it."""
    path = directory / f"{name}.parquet"
    if not path.exists():
        message = f"No view {name!r} at {path}. Run `uv run creditsurv views` first."
        raise FileNotFoundError(message)
    return pd.read_parquet(path)
