"""The site: every figure, table and number built from real views, and the committed views."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest
from mkdocs.exceptions import PluginError

from creditsurv.backtest.runner import predicted_hazard
from creditsurv.backtest.splits import cell_split
from creditsurv.config import project_root, reports_dir
from creditsurv.data.aggregate import build_cells
from creditsurv.data.ingest import Quarter, ingest_quarter
from creditsurv.data.panel import WEIGHT, to_interval_censored
from creditsurv.models.aft import fit_aft
from creditsurv.models.lifetime_pd import origination_book
from creditsurv.site import figures, hooks
from creditsurv.site.content import TABLES, VALUES, Sources, to_markdown
from creditsurv.site.figures import FIGURES, group_key, group_label, render
from creditsurv.views.model import (
    calibration_views,
    coefficient_view,
    covariates_over_time,
    projection_views,
)
from creditsurv.views.portfolio import portfolio_views
from creditsurv.views.selection import selection_views
from creditsurv.views.tables import IDENTIFYING, MANIFEST, load_manifest, write_views
from fixtures import (
    DEFAULT_PARAMS,
    build_panel,
    origination_row,
    performance_row,
    write_archives,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

    from creditsurv.site.content import Table, Value
    from creditsurv.site.figures import Figure

COVARIATES = ["credit_score", "ltv_change", "unemployment_change"]
PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)

#: The size the committed views must stay under, so the repository does not grow a data store.
TABLES_BUDGET_BYTES = 20 * 1024 * 1024

PLACEHOLDER_PAGES = sorted((project_root() / "docs").rglob("*.md"))


@pytest.fixture(scope="module")
def published(
    book_dir: Path,
    macro_module: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Sources]:
    """Every view the site reads, computed from fixtures by the functions `views` calls."""
    root = tmp_path_factory.mktemp("site")
    tables, reports = root / "tables", root / "reports"
    reports.mkdir()
    (reports / "portfolio_summary.json").write_text(
        (reports_dir() / "portfolio_summary.json").read_text()
    )

    panel, _ = build_panel(book_dir, macro_module, n_loans=1500, seed=71, params=PARAMS)
    encoded = to_interval_censored(panel).assign(**{WEIGHT: 1})
    split = cell_split(encoded, pd.PeriodIndex(encoded["period"]).max() - 12)
    fitted = fit_aft(split.train, COVARIATES, " + ".join(COVARIATES), weights_col=WEIGHT)
    train_hazard = predicted_hazard(fitted, split.train, COVARIATES).to_numpy()
    test_hazard = predicted_hazard(fitted, split.test, COVARIATES).to_numpy()
    model = [
        *calibration_views(
            split,
            train_hazard=train_hazard,
            test_hazard=test_hazard,
            families={"weibull": train_hazard, "loglogistic": train_hazard * 1.1},
        ),
        coefficient_view(fitted, split.train, COVARIATES),
        covariates_over_time(split, ["ltv_change", "unemployment_change"]),
        *projection_views(
            fitted,
            origination_book(split.train, macro_module, 40),
            macro_module,
            COVARIATES,
            horizon_months=24,
        ),
    ]
    write_views(model, tables, fit="fixture")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root / "data"))
        origination = [
            origination_row("F000000001", upb="200000", first_payment="201503", purpose="P"),
            origination_row("F000000002", upb="300000", first_payment="201503", purpose="C"),
        ]
        performance = [
            performance_row("F000000001", "201503", "0"),
            performance_row("F000000001", "201504", "1"),
            performance_row("F000000002", "201503", "0"),
            performance_row("F000000002", "201504", "1", delinquency="3"),
        ]
        write_archives(root / "data" / "FREDDIE MAC", 2015, {1: (origination, performance)})
        ingest_quarter(2015, 1)
        sources = (
            str(Quarter(2015, 1).parquet_path("perf")),
            str(Quarter(2015, 1).parquet_path("orig")),
        )
        cells = build_cells(*sources)
        write_views(portfolio_views(cells, macro_module, *sources), tables)

    write_views(selection_views(reports_dir()), tables)
    yield Sources(tables, reports)


@pytest.fixture
def unfloored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture book is a few thousand loan-months: every point would fall under the floor."""
    monkeypatch.setattr(figures, "EXPOSURE_FLOOR", 0)


def _only(sources: Sources, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """The views a builder declares, and nothing else: reading another one fails the test."""
    return {name: sources[name] for name in names}


@pytest.mark.usefixtures("unfloored")
@pytest.mark.parametrize("name", sorted(FIGURES))
def test_every_figure_builds_from_the_views_it_declares(published: Sources, name: str) -> None:
    figure = FIGURES[name].build(_only(published, FIGURES[name].views))

    assert len(figure.data) > 0
    html = render(figure, name)
    assert f'id="figure-{name}"' in html
    assert "<script src=" not in html


@pytest.mark.usefixtures("unfloored")
def test_a_segmented_figure_offers_every_segment_and_shows_one_at_a_time(
    published: Sources,
) -> None:
    table = published["km_vs_model"]
    figure = FIGURES["km_vs_model"].build(published)

    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == table["segment"].nunique()
    for button in buttons:
        shown = sum(button.args[0]["visible"])
        segment = next(
            name
            for name in table["segment"].unique()
            if figures.segment_title(name) == button.label
        )
        # Kaplan-Meier and the model, once for every group of the segment.
        assert shown == 2 * table.loc[table["segment"] == segment, "group"].nunique()
    assert [trace.visible for trace in figure.data].count(True) == 2


def test_the_floor_removes_points_resting_on_too_few_loan_months(published: Sources) -> None:
    with pytest.raises(ValueError, match="exposure floor"):
        FIGURES["km_vs_model"].build(published)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_every_table_builds_as_a_pipe_table(published: Sources, name: str) -> None:
    frame = TABLES[name].build(_only(published, TABLES[name].views))
    lines = to_markdown(frame).splitlines()

    assert len(lines) == len(frame) + 2
    assert all(line.count("|") == len(frame.columns) + 1 for line in lines)


@pytest.mark.parametrize("name", sorted(VALUES))
def test_every_value_reads_as_text(published: Sources, name: str) -> None:
    text = VALUES[name].read(published)

    assert text
    assert "nan" not in text


def test_the_backtest_numbers_are_the_whole_book_row(published: Sources) -> None:
    row = published["acceptance_by_segment"].set_index("segment").loc["all"]

    assert VALUES["backtest.ae"].read(published) == f"{row['actual_over_expected']:.3f}"
    assert VALUES["views.fit"].read(published) == "`fixture`"


def test_bands_are_ordered_by_their_lower_edge() -> None:
    labels = ["80 to 90", "not mapped", "0 to 60", "purchase", "up to 2003", "60 to 80"]

    ordered = sorted(labels, key=group_key)

    assert ordered == ["up to 2003", "0 to 60", "60 to 80", "80 to 90", "purchase", "not mapped"]


@pytest.mark.usefixtures("unfloored")
def test_age_bands_are_drawn_in_order_of_age(published: Sources) -> None:
    figure = FIGURES["ae_by_age_band"].build(published)

    order = list(figure.layout.xaxis.categoryarray)
    assert order == sorted(order, key=lambda band: int(band.split()[0]))
    for trace in figure.data:
        if trace.x is not None and len(trace.x) > 1:
            starts = [int(band.split()[0]) for band in trace.x]
            assert starts == sorted(starts)


def test_a_group_reads_as_its_label_under_current_and_former_codes() -> None:
    assert group_label("purpose", "cash_out_refinance") == "Cash-out refinance"
    assert group_label("purpose", "refinance_cashout") == "Cash-out refinance"
    assert group_label("mortgage_insurance", "uninsured") == "Not insured"
    assert group_label("channel", "third_party") == "Broker or correspondent"
    assert group_label("fico", "580 to 660") == "580 to 660"


# ----- the hook ----------------------------------------------------------------------------


CONFIG = cast("MkDocsConfig", None)
FILES = cast("Files", None)


def _page() -> Page:
    return cast(
        "Page", SimpleNamespace(file=SimpleNamespace(src_uri="page.md"), url="calibration/")
    )


@pytest.fixture
def hooked(published: Sources, monkeypatch: pytest.MonkeyPatch) -> Sources:
    monkeypatch.setitem(hooks._state, "sources", published)
    return published


@pytest.mark.usefixtures("hooked", "unfloored")
def test_the_hook_puts_numbers_and_tables_in_the_markdown_and_figures_in_the_html() -> None:
    page = _page()
    markdown = (
        "A/E <!-- value: backtest.ae -->.\n\n<!-- table: acceptance -->\n\n"
        "<!-- figure: ae_by_year -->\n"
    )

    rendered = hooks.on_page_markdown(markdown, page, CONFIG, FILES)
    html = hooks.on_page_content(f"<p>{rendered}</p>", page, CONFIG, FILES)

    assert "<!-- value" not in rendered
    assert "| Segment |" in rendered
    assert "<!-- figure: ae_by_year -->" in rendered
    assert 'id="figure-ae_by_year"' in html
    # plotly.js once, relative to the page, before the script that draws the figure.
    assert html.count("<script src=") == 1
    assert html.index('<script src="../javascripts/plotly.min.js">') < html.index(
        "figure-ae_by_year"
    )


@pytest.mark.usefixtures("hooked")
def test_a_placeholder_shown_as_code_is_left_alone() -> None:
    markdown = "```markdown\n<!-- value: backtest.ae -->\n```\n\n<!-- value: backtest.ae -->\n"

    rendered = hooks.on_page_markdown(markdown, _page(), CONFIG, FILES)

    assert rendered.count("<!-- value: backtest.ae -->") == 1
    assert rendered.startswith("```markdown\n<!-- value")


@pytest.mark.usefixtures("hooked")
def test_a_table_inside_an_admonition_keeps_its_indentation() -> None:
    # A code block first, so the placeholder's offset in its piece is not its offset in the page.
    markdown = (
        "```mermaid\nflowchart LR\n    A --> B\n```\n\n"
        '??? abstract "The table"\n    <!-- table: acceptance -->\n'
    )

    rendered = hooks.on_page_markdown(markdown, _page(), CONFIG, FILES)

    body = rendered.split('"The table"\n', 1)[1].splitlines()
    assert len(body) > 2
    assert all(line.startswith("    |") for line in body)


@pytest.mark.usefixtures("hooked")
@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        ("<!-- figure: no_such_figure -->", "no figure"),
        ("<!-- value: no.such.value -->", "no value"),
    ],
)
def test_the_hook_fails_the_build_on_a_placeholder_naming_nothing(
    markdown: str, message: str
) -> None:
    with pytest.raises(PluginError, match=message):
        hooks.on_page_markdown(markdown, _page(), CONFIG, FILES)


# ----- the committed pages and views -------------------------------------------------------

COMMITTED = project_root() / "docs" / "tables"


def _placeholders() -> list[tuple[str, str, str]]:
    found = []
    for page in PLACEHOLDER_PAGES:
        for kind, name in hooks.placeholders(page.read_text()):
            found.append((str(page.relative_to(project_root())), kind, name))
    return found


def test_every_placeholder_on_a_page_names_something_the_committed_views_can_build() -> None:
    manifest = load_manifest(COMMITTED)
    registries: dict[str, Mapping[str, Figure | Table | Value]] = {
        "figure": FIGURES,
        "table": TABLES,
        "value": VALUES,
    }

    for page, kind, name in _placeholders():
        registry = registries[kind]
        assert name in registry, f"{page}: no {kind} {name!r}"
        missing = set(registry[name].views) - set(manifest)
        assert not missing, f"{page}: {kind} {name!r} needs {sorted(missing)}"


def test_the_committed_views_are_listed_small_and_anonymous() -> None:
    manifest = load_manifest(COMMITTED)
    files = {path.stem for path in COMMITTED.glob("*.parquet")}

    assert set(manifest) == files
    total = sum(path.stat().st_size for path in COMMITTED.iterdir())
    assert total < TABLES_BUDGET_BYTES, f"docs/tables holds {total / 1e6:.1f} MB"
    for name, entry in manifest.items():
        assert not IDENTIFYING & set(cast("list[str]", entry["columns"])), name
    fits = {entry["fit"] for entry in manifest.values() if entry["fit"]}
    assert len(fits) <= 1, f"views from several fits: {fits}"
    assert json.loads((COMMITTED / MANIFEST).read_text()) == manifest


#: Pages that describe the model as it stands. The validation response and the decision log
#: record what was measured at the time, and quote those figures as history.
CURRENT_PAGES = ("index", "data", "portfolio", "methodology", "model", "calibration", "reproduce")


def test_a_number_the_views_provide_is_placed_on_a_page_never_typed() -> None:
    sources = Sources(COMMITTED, reports_dir())
    texts = {name: value.read(sources) for name, value in VALUES.items()}
    numbers = {
        name: text
        for name, text in texts.items()
        if len(text) >= 4 and any(character.isdigit() for character in text)
    }

    for page in CURRENT_PAGES:
        markdown = (project_root() / "docs" / f"{page}.md").read_text()
        # On a word boundary, because a substring match reads a number inside a longer one: the
        # Gini threshold 0.45 is in the actual-over-expected 0.4581 of a backtest window, which
        # is a different number from a different report and not a typed value at all.
        typed = [
            name
            for name, text in numbers.items()
            if re.search(rf"(?<![\d.]){re.escape(text)}(?![\d])", markdown)
        ]
        assert not typed, f"docs/{page}.md types {typed} instead of placing them"


# ----- no name the code uses reaches a reader -----------------------------------------------


def _code_names() -> set[str]:
    """Every name and level code that reads differently from its label."""
    from creditsurv import names

    found = set()
    for entry in names.VARIABLES.values():
        for name in (entry.name, entry.former):
            if name and "_" in name and name != entry.label:
                found.add(name)
        for level in entry.levels:
            for code in (level.code, level.former):
                if code and "_" in code:
                    found.add(code)
    return found


def _texts(figure: object) -> list[str]:
    layout = figure.layout  # type: ignore[attr-defined]
    texts = [str(trace.name or "") for trace in figure.data]  # type: ignore[attr-defined]
    for trace in figure.data:  # type: ignore[attr-defined]
        for axis in (trace.x, trace.y):
            if axis is not None and len(axis) and isinstance(axis[0], str):
                texts.extend(str(value) for value in axis)
    for menu in layout.updatemenus or ():
        texts.extend(str(button.label) for button in menu.buttons)
    return texts


@pytest.mark.usefixtures("unfloored")
def test_no_figure_or_table_shows_a_name_the_code_uses(published: Sources) -> None:
    pattern = re.compile(
        "|".join(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])" for name in _code_names())
    )
    shown = [text for figure in FIGURES.values() for text in _texts(figure.build(published))]
    # The Variables page is the one place the names are the point.
    for table in (table for name, table in TABLES.items() if not name.startswith("variables_")):
        frame = table.build(published)
        shown.extend(str(column) for column in frame.columns)
        shown.extend(str(value) for value in frame.to_numpy().ravel())

    leaks = sorted({match.group(0) for text in shown for match in pattern.finditer(text)})
    assert leaks == []


#: Every generated report, against the scale formula and the **shape** it should carry.
#:
#: A tuple of three was the blind spot the third validation found twice over. It covered
#: `methodology`, `calibration` and `backtesting`, so `windows.md` and `family.md` -- the two
#: that showed their formula in a table column instead of a provenance line -- could drift with
#: nothing comparing them; and it compared the scale formula alone, so four fits of a **scalar
#: shape** published every out-of-sample number in the project and no test could see it.
#:
#: It is a mapping and not a longer tuple because the expectation is not uniform. The prepayment
#: selection is deliberately unshaped (rule 14 is about the default hazard) and the log-logistic
#: record is of another family, so each report says which formula and which shape it owes.
#: ``...`` means "whatever the configuration says", and ``None`` means "a scalar shape, and say
#: so" -- an absent line is the ambiguity the provenance helper exists to remove.
EXPECTED_SPECIFICATION: dict[str, tuple[object, object]] = {
    "methodology": (..., ...),
    "calibration": (..., ...),
    "backtesting": (..., ...),
    "windows": (..., ...),
    "selection": (..., ...),
    "moratorium": (..., ...),
    # Another family and another cause: both are records of their own run, and holding them to
    # the configured formula would fail for being what they are.
    "selection_loglogistic": (None, None),
    "selection_weibull_prepayment": (None, None),
    "family": (None, None),
}

#: Kept for the reports whose formula is held to the configuration.
GENERATED_REPORTS = tuple(
    name for name, (formula, _) in EXPECTED_SPECIFICATION.items() if formula is ...
)

#: The marker a superseded report carries in its first lines, so that the test below can tell
#: "stale and said so" from "stale and silent". The second is the failure this test exists for.
SUPERSEDED_MARKER = "Superseded: this report describes a model that no longer exists"


def _predates_the_provenance(text: str) -> bool:
    """Whether a report was written before the provenance stated what it is.

    A self-retiring exemption, and the reason it is safe: a report written by today's package
    carries a ``Generated:`` line, so one without it cannot be held to lines that did not exist
    when it was written. The moment it is regenerated it gains the date **and** the formula and
    the shape, and from then on the checks bind. Nothing new can slip through, because nothing
    new is written without the date.

    Preferred over widening the exemption by name. A list of excused reports is a list somebody
    has to shorten; this one shortens itself, which is the difference between an exemption and
    a hole.
    """
    return not re.search(r"^[-*]?\s*Generated:\s*\d{4}-\d{2}-\d{2}", text, re.MULTILINE)


def test_a_generated_report_states_the_formula_it_was_run_on_or_says_it_is_superseded() -> None:
    """The drift the second validation found, caught by a test instead of a reader.

    Three reports generated on 17 September 2026 sat in the site's navigation describing a model
    with different covariate names, no HARP level, a different estimation window and prepayment as
    censoring -- while the main pages linked to them and `CLAUDE.md` quoted their backtest as the
    current state. Nothing failed, because nothing compared a generated report to the
    configuration it claims to describe.

    So: a generated report either carries the configured formula, or says in its first lines that
    it is superseded. Both are honest; silence is not. Regenerating is the cheap fix when the
    specification is settled, and the notice is the honest one while it is not.
    """
    from creditsurv.config import default_formula

    formula = default_formula()
    # Word boundaries, for the reason the test above gives: `dti` is inside `debt_to_income`, and
    # a substring match would read an old name as present in a new one.
    configured = set(re.findall(r"\w+", formula))

    for name in GENERATED_REPORTS:
        report = reports_dir() / f"{name}.md"
        if not report.exists():
            continue
        text = report.read_text()
        if SUPERSEDED_MARKER in text[:4000] or _predates_the_provenance(text):
            continue
        stated = re.search(r"^[-*]?\s*Formula:\s*`(?P<formula>[^`]+)`", text, re.MULTILINE)
        assert stated is not None, (
            f"docs/reports/{name}.md names no formula and carries no superseded notice, "
            "so nothing says which model it describes"
        )
        named = set(re.findall(r"\w+", stated.group("formula")))
        assert named == configured, (
            f"docs/reports/{name}.md was generated from a different specification: "
            f"{sorted(named - configured)} are not in the configuration, "
            f"{sorted(configured - named)} are missing. Regenerate it, or mark it superseded."
        )


def test_every_generated_report_states_its_shape_and_the_configured_ones_agree() -> None:
    """The half of the drift the formula could not catch, and did not for two days.

    `windows.md` published three regime cuts, an anchoring multiplier and a master scale of a
    **scalar-shape** model, while carrying the configured scale formula -- because
    `backtest/campaign.py` never passed the ancillary formula on, and nothing in the package
    stated or compared a shape. A report naming only its scale cannot be told from one of a model
    that has no shape at all, which is why the provenance prints "scalar" rather than omitting
    the line.
    """
    from creditsurv.config import default_shape

    shape = default_shape()
    configured = set(re.findall(r"\w+", shape))

    for name, (_, expected) in EXPECTED_SPECIFICATION.items():
        report = reports_dir() / f"{name}.md"
        if not report.exists():
            continue
        text = report.read_text()
        if SUPERSEDED_MARKER in text[:4000] or _predates_the_provenance(text):
            continue
        stated = re.findall(r"^[-*]?\s*(?:\w+ -- )?Shape:\s*`([^`]+)`", text, re.MULTILINE)
        assert stated, (
            f"docs/reports/{name}.md states no shape, so it cannot be told from a report of a "
            "model fitted without one -- which is the defect this test exists for"
        )
        if expected is not ...:
            continue
        named = set(re.findall(r"\w+", stated[0]))
        assert named == configured, (
            f"docs/reports/{name}.md was generated with a different shape: "
            f"{sorted(named - configured)} are not in `config.default_shape()`, "
            f"{sorted(configured - named)} are missing. Regenerate it, or mark it superseded."
        )


def test_no_generated_report_escapes_the_freshness_checks() -> None:
    """Every report a reporting module writes is named above, or the next one drifts unseen.

    `windows.md` and `family.md` drifted for a month because the check listed three reports by
    hand and nobody added the fourth. The list is still by hand -- it has to be, because the
    expectation differs per report -- so this is what notices when it falls behind: the writers
    name their own output files, and every one of them has to be accounted for.
    """
    import re as regex

    written = set()
    reporting = project_root() / "src" / "creditsurv" / "reporting"
    for module in sorted(reporting.glob("*.py")):
        source = module.read_text()
        written |= {
            match.group("name")
            for match in regex.finditer(r'reports_dir\s*/\s*f?"(?P<name>[\w_]+)\.md"', source)
        }
    # `selection.py` writes its name from a variable, so its three outputs cannot be read off
    # the source. They are named here for the same reason the mapping is by hand.
    written |= {"selection", "selection_loglogistic", "selection_weibull_prepayment"}
    unaccounted = written - set(EXPECTED_SPECIFICATION)
    assert unaccounted == set(), (
        f"these reports are written by the package and checked by nothing: {sorted(unaccounted)}. "
        "Add each to EXPECTED_SPECIFICATION with the formula and shape it owes."
    )


def test_the_views_manifest_and_the_configured_specification_name_one_model() -> None:
    """The site's model views come from one fit, and that fit is of the configured formula.

    `site/hooks.py` already refuses a manifest holding views from more than one fit. What it
    cannot see is whether that one fit is of the specification the configuration now declares --
    which is the other half of the drift the second validation found, and the half that would
    leave every figure on the site one specification out of date without a single placeholder
    failing.

    Skipped where the fit cache is not on the machine, which is every CI run: the manifest is
    committed and the pickles are not, so this can only be checked where the data lives. That
    makes it a local guard rather than a gate, and it is still the only thing that would catch a
    manifest naming a fit of a formula nobody configures any more.
    """
    from creditsurv.config import default_formula
    from creditsurv.data.store import find_fits

    manifest = load_manifest(COMMITTED)
    fits = {entry["fit"] for entry in manifest.values() if entry["fit"]}
    if not fits:
        pytest.skip("no model views are committed")
    assert len(fits) == 1, f"views from several fits: {fits}"
    named = fits.pop()

    described = dict(find_fits())
    if named not in described:
        pytest.skip(f"the manifest's fit {named} is not in this machine's cache")
    formula = described[named].get("formula")
    assert formula is not None, f"the cached fit {named} carries no formula to compare"
    # Word boundaries, for the reason the tests above give: a substring match reads an old
    # covariate name as present inside a new one.
    assert set(re.findall(r"\w+", str(formula))) == set(re.findall(r"\w+", default_formula())), (
        f"the views come from fit {named}, whose formula is not the configured one. "
        "Re-run `creditsurv views` after `creditsurv report`, or the site publishes figures "
        "of a specification the configuration has replaced."
    )
    # And the shape, which this test ignored while `views/build.py` already *searched* by it: a
    # fit of the right scale and no shape would have passed here and published every figure on
    # the site of a model nobody configured.
    from creditsurv.config import default_shape

    stated_shape = described[named].get("ancillary")
    assert stated_shape is not None, (
        f"the cached fit {named} carries no ancillary formula, so the views are of a scalar-shape "
        "model where rule 14 makes the shape part of the specification"
    )
    assert set(re.findall(r"\w+", str(stated_shape))) == set(re.findall(r"\w+", default_shape())), (
        f"the views come from fit {named}, whose shape is not the configured one."
    )
