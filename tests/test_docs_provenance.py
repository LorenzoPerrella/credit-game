"""Every precise number on a section page is placed from a view or recorded with its date.

The row "0.458 / 0.823 / 1.420" stood in `docs/validation.md` and `docs/calibration.md` for the
scalar-shape cuts for weeks after the campaign had measured 0.4920 / 1.3211 / 0.9849 on the
shaped model, and nothing failed: a number typed by hand says nothing when it goes stale. On
2 November 2026 the section pages held 72 such numbers, three of them copies of a value the site
already places.
"""

from __future__ import annotations

import re

import pytest

from creditsurv.config import project_root
from creditsurv.site.content import VALUES, Sources
from creditsurv.site.provenance import (
    SECTION_PAGES,
    free_numbers,
    live_copies,
    render_recorded,
    section_pages,
)

DOCS = project_root() / "docs"


@pytest.mark.parametrize("name", SECTION_PAGES)
def test_a_section_page_types_no_number(name: str) -> None:
    page = DOCS / name
    found = free_numbers(page.read_text(encoding="utf-8"))
    listed = "; ".join(item.describe() for item in found)
    assert not found, (
        f"{name} types {len(found)} precise number(s) by hand: {listed}. Place a run's output "
        "with a `value` placeholder, or put a dated figure between `<!-- recorded: YYYY-MM-DD -->` "
        "and `<!-- /recorded -->`."
    )


def test_every_section_page_exists() -> None:
    assert [page.name for page in section_pages(DOCS)] == list(SECTION_PAGES)


def test_no_free_number_is_a_copy_of_a_placed_value() -> None:
    """The stricter half, kept separate so its message names the placeholder to use."""
    sources = Sources(DOCS / "tables", DOCS / "reports")
    rendered = {}
    for name, value in VALUES.items():
        try:
            rendered[name] = value.read(sources)
        except Exception:
            continue
    copies = {
        page.name: live_copies(free_numbers(page.read_text(encoding="utf-8")), rendered)
        for page in section_pages(DOCS)
    }
    named = {
        page: [f"{item.describe()} is `{value}`" for item, value in found.items()]
        for page, found in copies.items()
        if found
    }
    assert not named


# What the check reads and what it leaves alone.


def test_a_precise_number_in_prose_is_found() -> None:
    page = "The cuts read 0.458, 0.823 and 1.420, and a coefficient of \u22120.0920.\n"
    assert [item.text for item in free_numbers(page)] == ["0.458", "0.823", "1.420", "\u22120.0920"]


def test_a_percentage_with_two_decimals_is_precise() -> None:
    assert [item.text for item in free_numbers("of which 0.11% and 17%")] == ["0.11%"]


@pytest.mark.parametrize(
    "prose",
    [
        "a band of 0.80 to 1.25",  # declared constants, two decimals
        "in 2026-10-23 at version 1.2.3",  # a date and a version
        "72,671,500 rows and 1,671,207 defaults",  # counts
        "the code `0.458` inline",
        "[a link](https://example.org/v0.123.456)",
        "<!-- value: backtest.ae -->",
    ],
)
def test_what_is_not_a_claim_is_left_alone(prose: str) -> None:
    assert free_numbers(prose) == []


def test_a_fenced_block_is_left_alone() -> None:
    page = "Prose.\n\n```text\n0.458 0.823\n```\n\n    ~~~\n    1.420\n    ~~~\nEnd.\n"
    assert free_numbers(page) == []


def test_a_recorded_block_is_left_alone_and_the_line_count_holds() -> None:
    page = (
        "<!-- recorded: 2026-10-19 -->\nThe September model read 0.920.\n<!-- /recorded -->\n"
        "And today 0.906.\n"
    )
    (found,) = free_numbers(page)
    assert (found.line, found.text) == (4, "0.906")


def test_a_recorded_block_without_its_date_is_not_recorded() -> None:
    page = "<!-- recorded -->\n0.920\n<!-- /recorded -->\n"
    assert [item.text for item in free_numbers(page)] == ["0.920"]


def test_a_recorded_block_shows_its_date_to_the_reader() -> None:
    page = "Before.\n<!-- recorded: 2026-10-19 -->\nIt read 0.920.\n<!-- /recorded -->\nAfter.\n"
    shown = render_recorded(page)
    assert "*Recorded on 2026-10-19:" in shown
    assert "It read 0.920." in shown
    assert "recorded:" not in shown
    assert "/recorded" not in shown


def test_a_copy_of_a_placed_value_is_named() -> None:
    (item,) = free_numbers("out of time 0.9065")
    assert live_copies([item], {"backtest.ae": "0.906"}) == {item: "backtest.ae"}
    # Fewer digits than the value prints cannot be a copy of it.
    (short,) = free_numbers("about 0.907 or so")
    assert live_copies([short], {"backtest.ae": "0.9065"}) == {}


# The navigation: `omitted_files: warn` fails a strict build on a page left out, but only in the
# build. The same fact here fails the suite, which is what the commit gate runs.


def _nav_pages(config: str) -> set[str]:
    nav = config.split("\nnav:", 1)[1]
    return set(re.findall(r":\s*([\w/.-]+\.md)\s*$", nav, re.MULTILINE))


def test_every_page_is_in_the_navigation() -> None:
    config = (project_root() / "mkdocs.yml").read_text(encoding="utf-8")
    pages = {
        str(path.relative_to(DOCS))
        for path in DOCS.rglob("*.md")
        if path.relative_to(DOCS).parts[0] != "tables"
    }
    missing = sorted(pages - _nav_pages(config))
    assert not missing, f"Pages outside the nav in mkdocs.yml: {missing}"


def test_the_navigation_names_no_missing_page() -> None:
    config = (project_root() / "mkdocs.yml").read_text(encoding="utf-8")
    absent = sorted(page for page in _nav_pages(config) if not (DOCS / page).is_file())
    assert not absent
