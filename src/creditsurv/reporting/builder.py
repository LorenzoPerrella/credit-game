"""A small markdown builder for the generated reports.

Reports are markdown rather than HTML or a notebook because they are read on
GitHub, diff cleanly between runs, and cost nothing to keep in the repository.

Tables are rendered here rather than through ``DataFrame.to_markdown`` so the
project does not take a dependency on ``tabulate`` for one function.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from creditsurv.names import readable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pandas as pd


def _format(value: object, decimals: int) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if value == 0 or 1e-4 <= abs(value) < 1e6:
            return f"{value:.{decimals}f}"
        return f"{value:.3g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, *, decimals: int = 4, index: bool = False) -> str:
    """Render a frame as a GitHub-flavoured markdown table."""
    working = frame.reset_index() if index else frame
    headers = [str(column) for column in working.columns]
    rows = [
        [_format(value, decimals) for value in record]
        for record in working.itertuples(index=False, name=None)
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


class Report:
    """Accumulates markdown sections and writes them to a file."""

    def __init__(self, title: str, *, subtitle: str = "") -> None:
        self._parts: list[str] = [f"# {title}"]
        if subtitle:
            self._parts.append(f"*{subtitle}*")

    def heading(self, text: str, *, level: int = 2) -> Report:
        self._parts.append(f"{'#' * level} {text}")
        return self

    def text(self, body: str) -> Report:
        self._parts.append(body.strip())
        return self

    def bullets(self, items: Iterable[str]) -> Report:
        self._parts.append("\n".join(f"- {item}" for item in items))
        return self

    def table(
        self,
        frame: pd.DataFrame,
        *,
        caption: str = "",
        decimals: int = 4,
        index: bool = False,
    ) -> Report:
        if caption:
            self._parts.append(f"**{caption}**")
        # Every table in a report is shown in labels: a reader sees "Credit score" and
        # "Cash-out refinance", not the names the code uses.
        shown = readable(frame.reset_index() if index else frame)
        self._parts.append(markdown_table(shown, decimals=decimals))
        return self

    def figure(self, path: Path, alt: str, *, caption: str = "") -> Report:
        # Reports live in docs/reports/ and figures in docs/reports/figures/, so a path
        # relative to the report file survives being moved or served elsewhere.
        self._parts.append(f"![{alt}]({path.parent.name}/{path.name})")
        if caption:
            self._parts.append(f"*{caption}*")
        return self

    def key_values(self, values: dict[str, object], *, decimals: int = 4) -> Report:
        self._parts.append(
            "\n".join(f"- **{key}**: {_format(value, decimals)}" for key, value in values.items())
        )
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(self._parts).rstrip() + "\n")
        return path


def provenance(lines: Sequence[str]) -> str:
    """A standard footer recording how a report was produced."""
    body = "\n".join(f"- {line}" for line in lines)
    return f"---\n\n## How this was produced\n\n{body}"


def specification(
    formula: str, shape: str | bool | None, *, today: date | None = None
) -> list[str]:
    """The provenance lines that say which model a report is of, and when it was written.

    Three lines, and all three were missing something. **The formula** was written by three
    reports of eleven, so `windows.md` and `family.md` could drift from the configuration with
    nothing to compare. **The shape** was written by none of them, which is how four fits of a
    scalar shape came to publish every out-of-sample number in the project: a report that does
    not state its ancillary formula cannot be told from one that has none. And **the date** was
    stamped by nothing at all, which is why three current reports sat in the navigation labelled
    with the month of the run before -- a nav label and a report body have to be comparable, and
    `git log` is not a thing a reader of the site has.

    The shape is printed as "scalar" rather than omitted when there is none, because an absent
    line is the ambiguity this function exists to remove.
    """
    written = "the scale's own formula" if shape is True else (shape or "scalar")
    return [
        f"Formula: `{formula}`",
        f"Shape: `{written}`",
        f"Generated: {(today or date.today()).isoformat()}",
    ]
