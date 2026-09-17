"""MkDocs hook: every page's figures, tables and numbers, from the committed views.

Placeholders are HTML comments, so a page read on GitHub shows nothing where a figure goes
rather than broken syntax:

* ``<!-- table: name -->`` and ``<!-- value: name -->`` become Markdown before the page is
  rendered, so tables are styled as every other table and numbers sit in running text;
* ``<!-- figure: name -->`` becomes the figure after rendering, so the Markdown processor
  never sees the figure's JSON.

A placeholder naming nothing, or a view the manifest does not list, fails the build. So does a
set of views taken from more than one fit: a page must not set a calibration from one model
beside coefficients from another.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mkdocs.exceptions import PluginError

from creditsurv.site.content import TABLES, VALUES, Sources, to_markdown
from creditsurv.site.figures import FIGURES, render

if TYPE_CHECKING:
    from collections.abc import Callable

    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

PLACEHOLDER: Final = re.compile(r"<!--\s*(figure|table|value):\s*([\w.]+)\s*-->")

#: A fenced code block, possibly indented inside an admonition: a placeholder shown as an
#: example is left alone.
_FENCE: Final = re.compile(
    r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL
)


def substitute(markdown: str, replace: Callable[[re.Match[str]], str]) -> str:
    """Replace the placeholders outside fenced code blocks."""
    pieces, position = [], 0
    for fence in _FENCE.finditer(markdown):
        pieces.append(PLACEHOLDER.sub(replace, markdown[position : fence.start()]))
        pieces.append(fence.group(0))
        position = fence.end()
    pieces.append(PLACEHOLDER.sub(replace, markdown[position:]))
    return "".join(pieces)


def placeholders(markdown: str) -> list[tuple[str, str]]:
    """The kind and name of every placeholder outside fenced code blocks."""
    found: list[tuple[str, str]] = []

    def record(match: re.Match[str]) -> str:
        found.append((match.group(1), match.group(2)))
        return match.group(0)

    substitute(markdown, record)
    return found


#: Where the build writes plotly.js, loaded in the head of the pages that draw a figure.
PLOTLY_JS: Final = "javascripts/plotly.min.js"

_state: dict[str, Sources] = {}


def _sources() -> Sources:
    return _state["sources"]


def _require(kind: str, name: str, needed: tuple[str, ...], page: Page) -> None:
    missing = [view for view in needed if view not in _sources().manifest]
    if missing:
        message = (
            f"{page.file.src_uri}: {kind} {name!r} needs the views {missing}, which "
            f"{_sources().tables} does not hold. Run `uv run creditsurv views`."
        )
        raise PluginError(message)


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    docs = Path(config.docs_dir)
    sources = Sources(docs / "tables", docs / "reports")
    fits = {entry["fit"] for entry in sources.manifest.values() if entry.get("fit")}
    if len(fits) > 1:
        named = sorted(map(str, fits))
        message = f"The views in {sources.tables} come from {len(fits)} fits: {named}."
        raise PluginError(message)
    _state["sources"] = sources
    return config


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    figures: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kind, name = match.groups()
        if kind == "figure":
            if name not in FIGURES:
                raise PluginError(f"{page.file.src_uri}: no figure {name!r}")
            _require(kind, name, FIGURES[name].views, page)
            figures.append(name)
            return match.group(0)
        if kind == "table":
            if name not in TABLES:
                raise PluginError(f"{page.file.src_uri}: no table {name!r}")
            _require(kind, name, TABLES[name].views, page)
            # Inside an admonition every line of the table has to carry the placeholder's
            # indentation, or the block ends at the table's second line.
            # Measured in the text the match was made in: a piece between two code blocks,
            # not the whole page.
            start = match.string.rfind("\n", 0, match.start()) + 1
            indent = match.string[start : match.start()]
            lines = to_markdown(TABLES[name].build(_sources())).splitlines()
            return f"\n{indent}".join(lines) if indent.isspace() else "\n".join(lines)
        if name not in VALUES:
            raise PluginError(f"{page.file.src_uri}: no value {name!r}")
        _require(kind, name, VALUES[name].views, page)
        return VALUES[name].read(_sources())

    replaced = substitute(markdown, replace)
    if figures:
        page.meta["plotly"] = True
    return replaced


def on_page_content(html: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    def replace(match: re.Match[str]) -> str:
        kind, name = match.groups()
        if kind != "figure":
            return match.group(0)
        return render(FIGURES[name].build(_sources()), name)

    return PLACEHOLDER.sub(replace, html)


def on_post_build(config: MkDocsConfig) -> None:
    from plotly.offline import get_plotlyjs

    target = Path(config.site_dir) / PLOTLY_JS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(get_plotlyjs(), encoding="utf-8")
