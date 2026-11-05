"""Where a number on the site comes from: a run, a dated record, or nowhere.

The site already has the mechanism for numbers that come from a run: a page writes a
placeholder, and `creditsurv.site.hooks` fills it from `docs/tables` at build time, so the number
on the page is the number the run produced. What nothing checked was whether a page **bypassed**
it, by typing a run's output by hand. A typed copy is right on the day it is written and wrong the
day the model changes, and it says nothing when it goes stale. The row "0.458 / 0.823 / 1.420" in
`docs/validation.md` described the scalar-shape cuts for weeks after the campaign had measured
0.4920 / 1.3211 / 0.9849.

So on a **section page** every precise number in the prose is one of two things:

- **placed**: a `value` placeholder, filled from a view at build time; or
- **recorded**: inside a block that names the date of its measurement, between the comments
  `<!-- recorded: YYYY-MM-DD -->` and `<!-- /recorded -->`. The site renders the date visibly,
  because a dated number should declare itself dated to the reader as well as to the test.

Anything else is a **free number**, and `tests/test_docs_provenance.py` fails on it. A *precise*
number is a decimal with three or more fractional digits, or a percentage with two decimals:
declared constants such as 0.80 or 1.25 have two and are not caught, and neither are versions,
dates or counts. The long records -- the rules, the register, the engineering notes, the plans --
are history by definition and are not section pages.

Standard library only at import, so the PostToolUse hook can load this file directly without
importing the package. Classifying a free number as a live copy of a current view needs pandas
and imports it inside :func:`live_copies`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

PLACEHOLDER: Final = re.compile(r"<!--\s*(figure|table|value):\s*([\w.]+)\s*-->")

#: A fenced code block, possibly indented inside an admonition: a placeholder shown as an
#: example is left alone.
FENCE: Final = re.compile(
    r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL
)

#: Inline code. The site hook used to fill placeholders even here, and only fenced blocks were
#: skipped, so a page could not mention a placeholder in prose without the build trying to fill it.
INLINE_CODE: Final = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")

RECORDED: Final = re.compile(
    r"<!--\s*recorded:\s*(\d{4}-\d{2}-\d{2})\s*-->(.*?)<!--\s*/recorded\s*-->", re.DOTALL
)

NUMBER: Final = re.compile(
    r"(?<![\w.,])[-+\u2212]?\d{1,3}(?:,\d{3})*\.\d{3,}(?!\d|\.\d)"
    r"|(?<![\w.,])\d+\.\d{2,}\s*%"
)

#: The pages that summarise the project for a reader, where a number is a claim about the model
#: as it is now. Everything else under `docs/` is a record, a plan or a generated report.
SECTION_PAGES: Final[tuple[str, ...]] = (
    "index.md",
    "data.md",
    "portfolio.md",
    "methodology.md",
    "model.md",
    "model_card.md",
    "calibration.md",
    "validation.md",
    "decisions.md",
    "reproduce.md",
)


@dataclass(frozen=True)
class Found:
    """A free number: its line on the page, the text as typed, and the value it reads as."""

    line: int
    text: str
    value: float
    decimals: int

    def describe(self) -> str:
        return f"line {self.line}: {self.text}"


def _masked(markdown: str) -> str:
    """The page with everything that is not prose replaced by spaces, keeping line numbers."""

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    text = FENCE.sub(blank, markdown)
    text = RECORDED.sub(blank, text)
    text = PLACEHOLDER.sub(blank, text)
    text = INLINE_CODE.sub(blank, text)
    # Link targets carry their own digits: a URL is not a claim.
    return re.sub(r"\]\([^)]*\)", blank, text)


def free_numbers(markdown: str) -> list[Found]:
    """Every precise number in the prose that is neither placed nor recorded."""
    masked = _masked(markdown)
    found = []
    for match in NUMBER.finditer(masked):
        text = match.group(0).strip()
        digits = re.sub(r"[^\d.\-]", "", text.replace("\u2212", "-"))
        try:
            value = float(digits)
        except ValueError:
            continue
        decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
        line = masked.count("\n", 0, match.start()) + 1
        found.append(Found(line, text, value, decimals))
    return found


def render_recorded(markdown: str) -> str:
    """Show a recorded block's date to the reader, and drop the markers."""

    def note(match: re.Match[str]) -> str:
        date, body = match.group(1), match.group(2)
        return (
            f"*Recorded on {date}: these figures are of the model and window of that date.*\n{body}"
        )

    return RECORDED.sub(note, markdown)


def section_pages(docs: Path) -> Iterator[Path]:
    for name in SECTION_PAGES:
        page = docs / name
        if page.is_file():
            yield page


def live_copies(found: Iterable[Found], rendered: dict[str, str]) -> dict[Found, str]:
    """Which free numbers are a named site value typed by hand, so a placeholder exists for them.

    `rendered` maps each `value` placeholder's name to what the site would show for it. A free
    number matches a value when it equals that value's number at the value's own printed
    precision -- 0.9065 is `backtest.ae` shown as 0.906.

    Matched against the named values only, and not against every cell of every view: the first
    version did that, and with thousands of cells at three decimals nearly every number "matched"
    something by coincidence, which is a check that says nothing.
    """
    targets = []
    for name, shown in rendered.items():
        for number in re.findall(r"-?\d[\d,]*\.\d+|-?\d[\d,]*", shown):
            digits = number.replace(",", "")
            places = len(digits.split(".", 1)[1]) if "." in digits else 0
            if places:
                targets.append((name, float(digits), places))
    out = {}
    for item in found:
        for name, number, places in targets:
            if item.decimals >= places and round(item.value, places) == round(number, places):
                out[item] = name
                break
    return out
