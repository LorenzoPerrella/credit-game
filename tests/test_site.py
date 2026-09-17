"""The site: every figure, table and number built from real views, and the committed views."""

from __future__ import annotations

import json
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
from creditsurv.site.figures import FIGURES, group_key, render, term_label
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


def test_a_categorical_term_is_named_against_its_reference() -> None:
    term = "C(purpose, Treatment('purchase'))[T.cash_out_refinance]"

    assert term_label(term) == "purpose: cash_out_refinance (against purchase)"
    assert term_label("credit_score") == "credit_score"


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
        typed = [name for name, text in numbers.items() if text in markdown]
        assert not typed, f"docs/{page}.md types {typed} instead of placing them"
