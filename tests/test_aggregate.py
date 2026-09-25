"""Tests for the DuckDB aggregation.

Small fixtures, real SQL. The point is not volume but that the event definition, the
truncation and the collapse are right — none of which needs a billion rows to check,
and all of which are impossible to check on one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.data.aggregate import CellSpec, build_cells, cardinality_report
from creditsurv.data.ingest import Quarter, ingest_quarter
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


def _ingested(tmp_path: Path, origination: list[str], performance: list[str]) -> None:
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)


def _sources(tmp_path: Path) -> tuple[str, str]:
    return (
        str(Quarter(2015, 1).parquet_path("perf")),
        str(Quarter(2015, 1).parquet_path("orig")),
    )


def test_identical_loans_collapse_into_one_cell(tmp_path: Path) -> None:
    """The claim the whole pipeline rests on."""
    origination = [origination_row(f"F{i:09d}") for i in range(20)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(20) for age in range(3)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    # Twenty identical loans over three months: one cell per month, weight twenty.
    # Episodes are monthly because the time-varying covariates are, so age is not
    # collapsed and the compression comes from identical loans instead.
    assert len(cells) == 3
    assert set(cells["loan_months"]) == {20}
    assert int(cells["loan_months"].sum()) == 60


def test_weights_account_for_every_loan_month(tmp_path: Path) -> None:
    origination = [origination_row(f"F{i:09d}", fico=str(620 + i * 20)) for i in range(6)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(6) for age in range(4)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["loan_months"].sum()) == 24


def test_default_is_flagged_once_and_the_loan_is_cut(tmp_path: Path) -> None:
    """Servicing files keep reporting through foreclosure; the loan must stop."""
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="1"),
        performance_row("F000000001", "201505", "2", delinquency="3"),
        performance_row("F000000001", "201506", "3", delinquency="4"),
        performance_row("F000000001", "201507", "4", delinquency="RA", zero_balance="09"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["loan_months"].sum()) == 3, "the loan should stop at its first defaulted month"
    assert int(cells.loc[cells["outcome"] == "default", "loan_months"].sum()) == 1
    assert int(cells.loc[cells["outcome"] == "default", "age"].iloc[0]) == 2


def test_an_reo_code_counts_even_when_delinquency_is_alphanumeric(tmp_path: Path) -> None:
    """`RA` casts to null, which compares false and reads as performing.

    The zero-balance code has to be checked alongside the delinquency status, not
    instead of it, or an REO acquisition silently becomes a surviving loan.
    """
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="RA", zero_balance="09"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells.loc[cells["outcome"] == "default", "loan_months"].sum()) == 1


def test_the_three_outcomes_are_told_apart(tmp_path: Path) -> None:
    """Default, a voluntary payoff, and the loan leaving the dataset are three things.

    Prepayment is a competing risk now, so it has to be the borrower's own decision to repay:
    a reperforming sale (16) and a removal (96) are neither default nor repayment. They end
    observation, as they did, but as censoring.
    """
    origination = [origination_row(f"F00000000{i}") for i in range(1, 5)]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="3"),
        performance_row("F000000002", "201503", "0"),
        performance_row("F000000002", "201504", "1", zero_balance="01", upb="0"),
        performance_row("F000000003", "201503", "0"),
        performance_row("F000000003", "201504", "1", zero_balance="16", upb="0"),
        performance_row("F000000004", "201503", "0"),
        performance_row("F000000004", "201504", "1", zero_balance="96", upb="0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))
    by_outcome = cells.groupby(cells["outcome"].astype(str))["loan_months"].sum()

    assert int(by_outcome.get("default", 0)) == 1
    assert int(by_outcome.get("prepayment", 0)) == 1
    # Eight loan-months in, two of them ending in an event: the other six are still at risk
    # as far as the likelihood is concerned, including the two sold and removed.
    assert int(by_outcome.get("none", 0)) == 6
    assert int(cells["loan_months"].sum()) == 8


def test_sentinel_values_drop_the_loan(tmp_path: Path) -> None:
    """9999 is not a credit score, and a loan without one cannot be modelled."""
    origination = [origination_row("F000000001", fico="9999"), origination_row("F000000002")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["loan_months"].sum()) == 1


def test_negative_ages_are_dropped(tmp_path: Path) -> None:
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201502", "-1"),
        performance_row("F000000001", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["loan_months"].sum()) == 1
    assert int(cells["age"].min()) == 0


def test_binning_puts_neighbouring_loans_in_one_cell(tmp_path: Path) -> None:
    """Coarse classing is what makes the collapse possible: without it every
    loan-month is its own cell and the aggregation saves nothing."""
    # Two scores a point apart fall in the same band.
    origination = [
        origination_row("F000000001", fico="742"),
        origination_row("F000000002", fico="743"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert len(cells) == 1
    assert int(cells["loan_months"].iloc[0]) == 2


def test_term_is_reduced_to_fifteen_or_thirty_years(tmp_path: Path) -> None:
    origination = [
        origination_row("F000000001", term="180"),
        origination_row("F000000002", term="360"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert set(cells["term_years"]) == {15, 30}


def test_cardinality_report_measures_the_collapse(tmp_path: Path) -> None:
    """Run before fixing the grain, not after: the decision needs a number."""
    origination = [origination_row(f"F{i:09d}") for i in range(10)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(10) for age in range(5)
    ]
    _ingested(tmp_path, origination, performance)

    report = cardinality_report(*_sources(tmp_path))

    assert int(report["loan_months"].iloc[0]) == 50
    assert int(report["weight_total"].iloc[0]) == 50
    # Ten identical loans over five months: one cell per month.
    assert int(report["cells"].iloc[0]) == 5
    assert float(report["compression"].iloc[0]) == pytest.approx(10.0)


def test_no_ingested_data_says_what_to_run() -> None:
    with pytest.raises(FileNotFoundError, match="creditsurv ingest"):
        build_cells()


def test_spec_rejects_an_unknown_continuous_covariate() -> None:
    spec = CellSpec(continuous={"not_a_covariate": (0.0, 1.0)}, categorical=())

    with pytest.raises(ValueError, match="Unknown continuous"):
        spec.validate()


def test_spec_rejects_an_unknown_categorical_covariate() -> None:
    spec = CellSpec(continuous={}, categorical=("not_a_covariate",))

    with pytest.raises(ValueError, match="Unknown categorical"):
        spec.validate()


def test_age_is_carried_in_months(tmp_path: Path) -> None:
    """Age is the episode's start in months, not an index, so the episode bounds read
    straight off it. Episodes are monthly because the covariates are."""
    origination = [origination_row("F000000001")]
    performance = [performance_row("F000000001", "201503", str(age)) for age in (0, 7, 30, 100)]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert sorted(cells["age"]) == [0, 7, 30, 100]


def test_a_narrower_spec_collapses_harder(tmp_path: Path) -> None:
    """The specification is the cardinality: it is the product of the band counts.

    This is why it has to be chosen after variable selection rather than before --
    aggregating on everything available produced a table too large to fit.
    """
    origination = [origination_row(f"F{i:09d}", fico=str(600 + i * 30)) for i in range(8)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(8) for age in range(3)
    ]
    _ingested(tmp_path, origination, performance)

    wide = build_cells(*_sources(tmp_path))
    narrow = build_cells(
        *_sources(tmp_path),
        spec=CellSpec(continuous={"credit_score": (-3.0, 3.0)}, categorical=()),
    )

    assert len(narrow) < len(wide)
    assert int(narrow["loan_months"].sum()) == int(wide["loan_months"].sum()) == 24


def test_the_event_flag_is_never_null(tmp_path: Path) -> None:
    """Most rows carry no zero-balance code at all.

    `FALSE OR NULL` is NULL in SQL, so a membership test against an absent code
    turns the event flag nullable, and that propagates all the way to the fitter --
    which rejects it with a message about numpy dtypes, a long way from the cause.
    Found on real data, where the field is absent; fixtures write it empty, which
    pyarrow reads as null in exactly the same way.
    """
    origination = [origination_row("F000000001"), origination_row("F000000002")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1"),
        performance_row("F000000002", "201503", "0", delinquency="3", zero_balance="09"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert set(cells["outcome"].astype(str)) <= {"default", "prepayment", "none"}
    assert not cells["outcome"].isna().any()
    assert int(cells.loc[cells["outcome"] == "default", "loan_months"].sum()) == 1


def test_no_categorical_mapping_has_an_else_branch() -> None:
    """An ELSE folds an unseen code into whichever level was written last.

    That is how `9` -- a "not available" marker -- ended up counted as a rate-and-term
    refinance. Every branch is listed explicitly so an unmapped code becomes NULL and
    the loan is dropped, which is the honest outcome for a value nobody has looked at.
    """
    from creditsurv.data.aggregate import _CATEGORICAL

    for name, expression in _CATEGORICAL.items():
        # Only the code mappings. A CASE on a numeric condition -- "is the term under
        # 190 months", "is there mortgage insurance" -- has no unseen values to
        # absorb, so an ELSE there is a genuine two-way split.
        if "WHEN '" not in expression:
            continue
        assert " ELSE " not in expression.upper(), f"{name} has an ELSE branch"


def test_an_unmapped_code_drops_the_loan(tmp_path: Path) -> None:
    """`9` is the dataset's not-available marker for loan purpose."""
    origination = [
        origination_row("F000000001", purpose="P"),
        origination_row("F000000002", purpose="9"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["loan_months"].sum()) == 1
    assert set(cells["purpose"]) == {"purchase"}


def test_channel_is_collapsed_to_a_comparable_binary(tmp_path: Path) -> None:
    """The four-level split is not comparable across the history.

    Until 2008 about half of originations are coded T, third-party not otherwise
    specified, and broker and correspondent are near zero; from 2009 T vanishes and
    those two absorb it. That is a change in how the field was coded, not in how
    loans were sold, and a model given four levels reads it as a risk effect.
    Retail's own share is stable throughout, so the binary split is the part that
    means the same thing in every vintage.
    """
    origination = [
        origination_row("F000000001", channel="R"),
        origination_row("F000000002", channel="C"),
        origination_row("F000000003", channel="T"),
        origination_row("F000000004", channel="B"),
    ]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(1, 5)]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(
        *_sources(tmp_path),
        spec=CellSpec(continuous={}, categorical=("channel",)),
    )

    assert set(cells["channel"]) == {"retail", "broker_or_correspondent"}
    assert int(cells.loc[cells["channel"] == "broker_or_correspondent", "loan_months"].sum()) == 3


def test_the_default_formula_only_names_covariates_the_cells_carry() -> None:
    """The specification and the formula have to agree, or nothing can be fitted.

    A cell is a covariate combination: whatever the key does not carry is not
    recoverable from it, so a formula naming such a covariate cannot be evaluated at
    all. The two drifted apart once -- the formula kept five covariates the
    aggregation had stopped producing -- and the symptom was ``creditsurv fit``
    failing on real data while every test passed, because the tests used a
    loan-level panel that carried everything.

    Derived covariates are exempt: they are functions of the key rather than part of
    it, which is the whole reason they are cheap.
    """
    from creditsurv.config import (
        CATEGORICAL_REFERENCE,
        ORDINAL,
        STATIC_CONTINUOUS,
        TIME_VARYING_CONTINUOUS,
    )
    from creditsurv.data.aggregate import DEFAULT_SPEC
    from creditsurv.features import MACRO_DERIVED

    in_key = set(DEFAULT_SPEC.continuous) | set(DEFAULT_SPEC.categorical)
    #: Rebuilt by cells_to_episodes from the vintage, the age and the macro path.
    derived = set(MACRO_DERIVED)

    modelled = set(STATIC_CONTINUOUS) | set(TIME_VARYING_CONTINUOUS) | set(ORDINAL)
    modelled |= set(CATEGORICAL_REFERENCE)

    missing = modelled - in_key - derived
    assert not missing, f"the formula names {sorted(missing)}, which no cell carries"


def test_a_modified_loan_is_cut_at_the_modification(tmp_path: Path) -> None:
    """The dataset restarts ``loan_age`` at a modification, and it must not be believed.

    A real loan from the 2006 vintage runs to age 192 at twenty months delinquent,
    is modified, and reappears the next month at age 3 with a clean delinquency
    status. Believing that gives the same loan two episodes at the same age, and
    re-files a previously-distressed month as a performing one at a young age --
    where it dilutes the part of the hazard curve the model is most sensitive to.

    Observation ends at the modification, as it does at a prepayment: the modified
    contract is a different loan.
    """
    import duckdb

    from creditsurv.data.aggregate import _state_of_the_book_sql

    performance = [
        performance_row("F15Q1000001", "201503", "0"),
        performance_row("F15Q1000001", "201504", "1"),
        performance_row("F15Q1000001", "201505", "2"),
        # Modified: age restarts, and every later row carries the prior-modification
        # flag. Ages 0 and 1 now appear twice for this loan.
        performance_row("F15Q1000001", "201506", "0", modification="Y"),
        performance_row("F15Q1000001", "201507", "1", modification="P"),
    ]
    _ingested(tmp_path, [origination_row("F15Q1000001")], performance)

    perf, orig = _sources(tmp_path)
    book = duckdb.connect().execute(_state_of_the_book_sql(), [perf, orig]).df()
    book = book.sort_values("period_key")

    assert book["age"].tolist() == [0, 1, 2], "history should stop before the modification"
    assert not book["event"].any(), "a modification is censoring, never a default"


def test_truncation_follows_calendar_time_not_age(tmp_path: Path) -> None:
    """Cutting on ``MIN(age)`` picks the wrong row once age is not monotone.

    The loan below defaults at age 4, is modified, and the post-modification rows
    carry lower ages than the default did. Truncating on the smallest terminating
    *age* would cut at age 1 and lose the default entirely; truncating on the
    earliest terminating *period* keeps it.
    """
    import duckdb

    from creditsurv.data.aggregate import _state_of_the_book_sql

    performance = [
        performance_row("F15Q1000001", "201503", "0"),
        performance_row("F15Q1000001", "201504", "1"),
        performance_row("F15Q1000001", "201505", "2"),
        performance_row("F15Q1000001", "201506", "3"),
        performance_row("F15Q1000001", "201507", "4", delinquency="3"),
        performance_row("F15Q1000001", "201508", "1", modification="Y"),
        performance_row("F15Q1000001", "201509", "2", modification="P"),
    ]
    _ingested(tmp_path, [origination_row("F15Q1000001")], performance)

    perf, orig = _sources(tmp_path)
    book = duckdb.connect().execute(_state_of_the_book_sql(), [perf, orig]).df()
    book = book.sort_values("period_key")

    assert book["age"].tolist() == [0, 1, 2, 3, 4]
    assert book["event"].tolist() == [False, False, False, False, True]


def test_the_eliminated_covariates_are_out_of_the_model() -> None:
    """The record of what was dropped and the specification cannot drift apart.

    A covariate removed by the selection that quietly reappears in the formula is a
    silent reversal of a documented decision, and nothing else in the suite would
    notice it.

    The formula is read as names, not as text: ``equity_volatility`` is inside ``volatility_change``
    and ``inflation_rate`` inside ``inflation_change``, so a level eliminated beside its own gap
    form would read as still fitted. And a candidate is anything a cell carries -- the selection
    screens loan characteristics and categoricals too, not only macro series.
    """
    from creditsurv.config import ELIMINATED, default_formula
    from creditsurv.data.aggregate import DEFAULT_SPEC
    from creditsurv.features import MACRO_DERIVED

    carried = set(DEFAULT_SPEC.continuous) | set(DEFAULT_SPEC.categorical) | set(MACRO_DERIVED)
    names = set(re.findall(r"\w+", default_formula()))
    for name, reason in ELIMINATED.items():
        assert name in carried, f"{name} is recorded as eliminated but no cell carries it"
        assert name not in names, f"{name} was eliminated ({reason}) but is still fitted"


def test_the_production_grid_is_a_subset_of_the_documented_one() -> None:
    """Two classing schemes used to coexist and only one built cells.

    `data_preparation.md` justified a DTI break at 43 as a long-standing underwriting
    threshold while the model used 45, and named LTV breaks at 85 and 95 the model did
    not have. A reader checking the economic justification of the bands found a
    justification that did not describe the model.

    Making the production grid a strict subset leaves only the coarsening to explain,
    and every boundary that exists is one the documentation argues for.
    """
    from creditsurv.data.aggregate import PRODUCTION_EDGES
    from creditsurv.features import BIN_EDGES

    for name, edges in PRODUCTION_EDGES.items():
        documented = set(BIN_EDGES[name])
        stray = [edge for edge in edges if edge not in documented]
        assert not stray, f"{name} cuts at {stray}, which BIN_EDGES does not justify"

    assert 43.0 in PRODUCTION_EDGES["debt_to_income"], "the documented underwriting threshold is 43"
    assert 80.0 in PRODUCTION_EDGES["original_ltv"], "the mortgage-insurance threshold"


def _moratorium_quarter() -> tuple[list[str], list[str]]:
    """A loan on a statutory payment holiday, and one that simply stops paying."""
    return (
        [origination_row("F15Q1000001"), origination_row("F15Q1000002")],
        [
            # Forbearance: reported 90+ by statute, then cured.
            performance_row("F15Q1000001", "201503", "0"),
            performance_row("F15Q1000001", "201504", "1"),
            performance_row("F15Q1000001", "201505", "2", delinquency="3", assistance="F"),
            performance_row("F15Q1000001", "201506", "3", delinquency="4", assistance="F"),
            performance_row("F15Q1000001", "201507", "4"),
            performance_row("F15Q1000001", "201508", "5"),
            # A real default, after the accommodation ended.
            performance_row("F15Q1000001", "201509", "6", delinquency="3"),
            # An ordinary borrower who stops paying, with no marker at all.
            performance_row("F15Q1000002", "201503", "0"),
            performance_row("F15Q1000002", "201504", "1"),
            performance_row("F15Q1000002", "201505", "2", delinquency="3"),
        ],
    )


def test_a_statutory_payment_holiday_is_not_a_default(tmp_path: Path) -> None:
    """The CARES Act required loans in forbearance to be reported as delinquent, so a
    payment holiday reads identically to a borrower who has stopped paying.

    It is 17% of this book's events, and on 2019Q3 **87% of all 90+ rows carry an
    accommodation marker**. Counting them is not a rounding error in the dependent
    variable; it is most of the 2020 peak.
    """
    import duckdb

    from creditsurv.data.aggregate import MoratoriumPolicy, _state_of_the_book_sql

    _ingested(tmp_path, *_moratorium_quarter())
    perf, orig = _sources(tmp_path)

    holiday, stopped = "F15Q1000001", "F15Q1000002"

    def book(policy: MoratoriumPolicy) -> pd.DataFrame:
        frame = duckdb.connect().execute(_state_of_the_book_sql(policy), [perf, orig]).df()
        return frame.sort_values(["loan_identifier", "period_key"])

    def event_ages(frame: pd.DataFrame, loan: str) -> list[int]:
        # Per loan, always. The first version of this test pooled both loans and read
        # the unmarked borrower's default at age 2 as a failure of the exclusion.
        rows = frame[frame["loan_identifier"] == loan]
        return [int(age) for age in rows.loc[rows["outcome"] == "default", "age"]]

    def last_age(frame: pd.DataFrame, loan: str) -> int:
        return int(frame.loc[frame["loan_identifier"] == loan, "age"].max())

    ignored = book(MoratoriumPolicy.IGNORE)
    assert event_ages(ignored, holiday) == [2], "ignoring markers, the holiday is a default"
    assert event_ages(ignored, stopped) == [2]

    excluded = book(MoratoriumPolicy.EXCLUDE)
    # The accommodated months are kept and carry no event, and the loan stays at risk,
    # so the real default after the holiday is still caught.
    assert event_ages(excluded, holiday) == [6], "the default after the holiday, not the holiday"
    assert last_age(excluded, holiday) == 6
    assert event_ages(excluded, stopped) == [2], "an unmarked delinquency is still a default"

    censored = book(MoratoriumPolicy.CENSOR)
    # Observation ends the month before the flagged one, so the holiday is not a
    # default -- and neither is the real one that followed it, which is the price.
    assert event_ages(censored, holiday) == []
    assert last_age(censored, holiday) == 1
    assert event_ages(censored, stopped) == [2]


def test_the_two_moratorium_treatments_are_not_equivalent(tmp_path: Path) -> None:
    """They differ in what they keep, which is why the choice is measured rather than
    argued: EXCLUDE keeps the exposure and catches the later default, CENSOR does not."""
    import duckdb

    from creditsurv.data.aggregate import MoratoriumPolicy, _state_of_the_book_sql

    _ingested(tmp_path, *_moratorium_quarter())
    perf, orig = _sources(tmp_path)
    connection = duckdb.connect()

    kept = connection.execute(_state_of_the_book_sql(MoratoriumPolicy.EXCLUDE), [perf, orig]).df()
    lost = connection.execute(_state_of_the_book_sql(MoratoriumPolicy.CENSOR), [perf, orig]).df()

    assert len(kept) > len(lost), "censoring gives up the exposure after the accommodation"


def test_text_keys_stay_categorical_through_the_concatenation() -> None:
    """A quarter declaring different levels must not turn a column back into strings.

    pandas keeps a categorical through ``concat`` only when the levels agree, and the
    exact key's five text columns over ~66 million cells would otherwise be gigabytes of
    Python strings. So the levels are unified, and sorted, before the quarters are stacked.
    """
    from creditsurv.data.aggregate import _compact, _concatenate

    first = _compact(
        pd.DataFrame(
            {
                "occupancy": ["owner_occupied", "investment_property"],
                "origination_month": [24_000, 24_001],
                "loan_months": [3, 4],
            }
        )
    )
    second = _compact(
        pd.DataFrame(
            {"occupancy": ["second_home"], "origination_month": [24_002], "loan_months": [5]}
        )
    )

    combined = _concatenate([first, second])

    assert isinstance(combined["occupancy"].dtype, pd.CategoricalDtype)
    assert list(combined["occupancy"].cat.categories) == [
        "investment_property",
        "owner_occupied",
        "second_home",
    ]
    assert combined["occupancy"].tolist() == [
        "owner_occupied",
        "investment_property",
        "second_home",
    ]
    assert combined["origination_month"].dtype == "int32"


def test_cells_are_categorical_as_built_and_as_saved(tmp_path: Path) -> None:
    """So no command downstream pays for a table of strings."""
    from creditsurv.data.store import load_cells, save_cells

    origination = [origination_row(f"F{i:09d}", purpose="P" if i % 2 else "C") for i in range(6)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(6) for age in range(3)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    for column in ("vintage", "purpose", "occupancy", "mortgage_insurance", "buyer_type"):
        assert isinstance(cells[column].dtype, pd.CategoricalDtype), column
    save_cells(cells, "exclude")
    pd.testing.assert_frame_equal(load_cells("exclude"), cells)


def test_the_loans_left_out_are_counted_by_what_they_lack_and_how_they_default(
    tmp_path: Path,
) -> None:
    """D4: dropping incomplete loans is harmless only if they are few or default like the
    rest, and neither can be assumed. Every field that drops a loan is counted, and the
    dropped loans' default rate is set beside the kept ones'."""
    from creditsurv.data.aggregate import incomplete_cases

    origination = [
        origination_row("F000000001"),
        origination_row("F000000002"),
        origination_row("F000000003", debt_to_income="999"),
        origination_row("F000000004", purpose="9"),
    ]
    performance = [
        *(performance_row(f"F00000000{i}", "201503", "0") for i in (1, 2, 3, 4)),
        performance_row("F000000003", "201504", "1", delinquency="3"),
    ]
    _ingested(tmp_path, origination, performance)

    row = incomplete_cases(*_sources(tmp_path)).iloc[0]

    assert row["loans"] == 4
    assert row["dropped"] == 2
    assert row["no_debt_to_income"] == 1
    assert row["no_purpose"] == 1
    assert row["default_rate_kept"] == pytest.approx(0.0)
    assert row["default_rate_dropped"] == pytest.approx(0.5)


def test_a_reperforming_sale_is_censored_only_when_nothing_came_before_it(
    tmp_path: Path,
) -> None:
    """D5: code 16 is credit by definition, so censoring the sale could lose a default. It
    does not when the book has already acted: a loan that reached 90 days defaulted there,
    and a modified one was censored at the modification. Only a loan performing when sold
    is censored at the sale. A prepayment is not one of these exits."""
    from creditsurv.data.aggregate import credit_adjacent_exits

    origination = [origination_row(f"F00000000{i}") for i in range(1, 6)]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="3"),
        performance_row("F000000001", "201505", "2", zero_balance="16", upb="0"),
        performance_row("F000000002", "201503", "0"),
        performance_row("F000000002", "201504", "1", zero_balance="16", upb="0"),
        performance_row("F000000003", "201503", "0"),
        performance_row("F000000003", "201504", "1", modification="Y"),
        performance_row("F000000003", "201505", "0", zero_balance="16", upb="0"),
        performance_row("F000000004", "201503", "0"),
        performance_row("F000000004", "201504", "1", zero_balance="96", upb="0"),
        performance_row("F000000005", "201503", "0"),
        performance_row("F000000005", "201504", "1", zero_balance="01", upb="0"),
    ]
    _ingested(tmp_path, origination, performance)

    table = credit_adjacent_exits(*_sources(tmp_path)).set_index("code")
    counts = ["loans", "defaulted_first", "censored_earlier", "censored_at_exit"]

    assert table.loc["16", counts].to_numpy().tolist() == [3, 1, 1, 1]
    assert table.loc["96", counts].to_numpy().tolist() == [1, 0, 0, 1]
    assert "01" not in table.index


def test_the_cells_give_back_the_monthly_default_series_exactly(tmp_path: Path) -> None:
    """M1. Origination month plus age is the month a default happened in, so the series
    read back from the cells must equal the one counted on the loan-months. With the
    origination quarter in the key, the two peaked two months out of step."""
    from creditsurv.data.aggregate import defaults_by_month
    from creditsurv.data.panel import defaults_by_observation_month

    origination = [origination_row(f"F{i:09d}") for i in range(6)]
    performance = []
    for i in range(6):
        for age, period in enumerate(("201503", "201504", "201505", "201506")):
            defaulted = (i, age) in {(1, 2), (4, 3)}
            performance.append(
                performance_row(
                    f"F{i:09d}", period, str(age), delinquency="3" if defaulted else "0"
                )
            )
            if defaulted:
                break
    _ingested(tmp_path, origination, performance)

    truth = defaults_by_month(*_sources(tmp_path))
    rebuilt = defaults_by_observation_month(build_cells(*_sources(tmp_path)))

    assert int(truth.sum()) == 2
    assert truth[truth > 0].to_dict() == rebuilt[rebuilt > 0].to_dict()


def test_the_monthly_default_rate_is_defaults_over_the_loan_months_at_risk(
    tmp_path: Path,
) -> None:
    """D1's evidence was a monthly rate: defaults over the loan-months at risk that month.
    Four loans, one defaulting in its second month; the month after has three at risk."""
    from creditsurv.data.panel import default_rate_by_observation_month

    origination = [origination_row(f"F{i:09d}") for i in range(4)]
    performance = [
        performance_row(
            f"F{i:09d}", period, str(age), delinquency="3" if (i, age) == (0, 1) else "0"
        )
        for i in range(4)
        for age, period in enumerate(("201503", "201504", "201505"))
        if i != 0 or age <= 1
    ]
    _ingested(tmp_path, origination, performance)

    rates = default_rate_by_observation_month(build_cells(*_sources(tmp_path)))

    assert rates["period"].astype(str).tolist() == ["2015-03", "2015-04", "2015-05"]
    assert rates["loan_months"].tolist() == [4, 4, 3]
    assert rates["defaults"].tolist() == [0, 1, 0]
    assert rates["rate_bp"].tolist() == pytest.approx([0.0, 2500.0, 0.0])


def test_the_super_conforming_flag_is_mapped_from_what_the_field_holds(tmp_path: Path) -> None:
    """D6. The layout calls a blank "not super conforming" and the first mapping followed it,
    turning NULL into N. The field holds N or Y on every one of 49.2 million loans, so that
    mapping would have dropped 98% of the book had the flag ever entered a key."""
    from creditsurv.data.aggregate import PRODUCTION_EDGES

    origination = [origination_row(f"F{i:09d}", loan_size="Y" if i < 2 else "N") for i in range(6)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(6)]
    _ingested(tmp_path, origination, performance)

    spec = CellSpec(continuous=PRODUCTION_EDGES, categorical=("loan_size",))
    cells = build_cells(*_sources(tmp_path), spec=spec)

    loans = cells.groupby("loan_size", observed=True)["loan_months"].sum().to_dict()
    assert loans == {"conforming": 4, "super_conforming": 2}


def test_a_harp_refinance_is_kept_with_its_ratio_missing_and_its_level_set(
    tmp_path: Path,
) -> None:
    """The open question closed. HARP loans report no debt-to-income, and the
    complete-case rule dropped every one of them: 18% of the 2009Q2 to 2019Q1 vintages, at
    three times the default rate of the loans kept.

    They come in with a level of their own and the ratio still missing. Missing, not
    filled: what the model does with it is decided on the model's side, where a level
    absorbs it -- see ``features.NOT_REPORTED``.
    """
    origination = [
        origination_row("F000000001", harp="Y", debt_to_income=""),
        origination_row("F000000002", harp="N", debt_to_income="32"),
        # No ratio and no HARP: still dropped, since nothing explains the gap.
        origination_row("F000000003", harp="N", debt_to_income=""),
    ]
    performance = [
        performance_row(loan, f"2015{month:02d}", str(month - 3))
        for loan in ("F000000001", "F000000002", "F000000003")
        for month in (3, 4)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    by_level = cells.groupby("harp", observed=True)["loan_months"].sum()
    assert by_level.to_dict() == {"harp": 2, "standard": 2}
    refinanced = cells[cells["harp"] == "harp"]
    assert refinanced["debt_to_income"].isna().all(), "nothing is imputed in the cells"
    assert cells[cells["harp"] == "standard"]["debt_to_income"].notna().all()


def test_the_payment_state_is_the_month_before_not_the_month_itself(tmp_path: Path) -> None:
    """A loan 90 days late has already defaulted, so the state during the month is the
    event. The month before is what a servicer knows when the month opens.
    """
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0", delinquency="0"),
        performance_row("F000000001", "201504", "1", delinquency="1"),
        performance_row("F000000001", "201505", "2", delinquency="2"),
        performance_row("F000000001", "201506", "3", delinquency="3"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    state = dict(zip(cells["age"], cells["delinquency_state"].astype(str), strict=True))
    # Age 0 has no earlier month: the loan opens current, which is not a missing value.
    assert state == {0: "current", 1: "current", 2: "one_month", 3: "two_months"}
    defaulted = cells[cells["outcome"] == "default"]
    assert list(defaulted["age"]) == [3]
    assert list(defaulted["delinquency_state"].astype(str)) == ["two_months"]


def test_an_unreadable_payment_state_drops_the_month_rather_than_reading_as_current(
    tmp_path: Path,
) -> None:
    """``RA`` is a real value of the field -- an REO acquisition -- and casting it to a
    number first would turn it into the same NULL as "this is the loan's first month",
    which reads as up to date. Every other mapping here drops a code nobody has looked at,
    and so does this one.
    """
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0", delinquency="0"),
        performance_row("F000000001", "201504", "1", delinquency="RA"),
        performance_row("F000000001", "201505", "2", delinquency="0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert sorted(cells["age"]) == [0, 1], "the month after RA has no readable state"


def test_the_note_rate_enters_the_key_as_a_band(tmp_path: Path) -> None:
    """It is there for the spread and the refinancing incentive, both of which are the
    rate against a market rate of a month the key already carries. A band of the rate is
    a band of both.

    Not in the production key: measured at 2.13x on its own, it was the second rung the
    give-up order of docs/rules.md reached. The extension still has to work -- the ceiling
    is a function of the book, and a coarser grid or a bigger machine puts it back.
    """
    from creditsurv.data.aggregate import BASE_SPEC, Extension, extended

    origination = [
        origination_row("F000000001", rate="3.10"),
        origination_row("F000000002", rate="3.40"),
        origination_row("F000000003", rate="6.90"),
    ]
    performance = [performance_row(f"F00000000{i}", "201503", "0") for i in (1, 2, 3)]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path), spec=extended(BASE_SPEC, Extension.ORIGINATION_SPREAD))

    # 3.10 and 3.40 share the band (3.0, 3.5] and collapse into one cell, which is the
    # whole reason the rate is banded rather than carried; 6.90 sits in (6.5, 7.0].
    assert sorted(cells["note_rate"]) == [3.25, 6.75]
    assert int(cells.loc[cells["note_rate"] == 3.25, "loan_months"].sum()) == 2


def test_each_extension_of_the_key_switches_on_alone(tmp_path: Path) -> None:
    """The cell count is the product of the band counts, so an extension has to be priced
    before it is adopted -- which is only possible one at a time. The order they are given
    up in is fixed in docs/rules.md, before the measurement.
    """
    from creditsurv.data.aggregate import BASE_SPEC, DEFAULT_SPEC, Extension, extended

    assert Extension.HARP not in _GIVE_UP_ORDER(), "HARP is a correction, not a refinement"
    assert set(DEFAULT_SPEC.categorical) >= set(BASE_SPEC.categorical)

    harp_only = extended(BASE_SPEC, Extension.HARP)
    assert "harp" in harp_only.categorical
    assert "delinquency_state" not in harp_only.categorical
    assert harp_only.continuous == BASE_SPEC.continuous

    finer = extended(BASE_SPEC, Extension.FINE_BANDS)
    assert finer.categorical == BASE_SPEC.categorical
    assert len(finer.continuous["credit_score"]) > len(BASE_SPEC.continuous["credit_score"])

    spread = extended(BASE_SPEC, Extension.ORIGINATION_SPREAD)
    assert "note_rate" in spread.continuous

    assert extended(BASE_SPEC, *Extension) == extended(BASE_SPEC, *reversed(list(Extension)))


def _GIVE_UP_ORDER() -> tuple[object, ...]:
    from creditsurv.data.aggregate import GIVE_UP_ORDER

    return GIVE_UP_ORDER


def test_a_window_of_the_cells_is_read_without_the_rest_of_the_table(tmp_path: Path) -> None:
    """A backtest window is two years of observation, about 3% of the table, and the whole
    table is a few gigabytes as a frame. The filter is on the observation month, which is
    origination plus age and so not a column parquet can be asked about by name.
    """
    from creditsurv.data.store import load_cells_window, save_cells

    origination = [origination_row(f"F{i:09d}") for i in range(4)]
    performance = [
        performance_row(f"F{i:09d}", f"2015{month:02d}", str(month - 3))
        for i in range(4)
        for month in (3, 4, 5, 6)
    ]
    _ingested(tmp_path, origination, performance)
    cells = build_cells(*_sources(tmp_path))
    save_cells(cells, "exclude")

    april = 2015 * 12 + 4 - 1
    window = load_cells_window("exclude", first=april, last=april + 1)
    whole = load_cells_window("exclude")

    observed = window["origination_month"] + window["age"]
    assert set(observed) == {april, april + 1}
    assert len(whole) == len(cells)
    assert int(window["loan_months"].sum()) < int(whole["loan_months"].sum())
    # Open at either end, as a development window is.
    assert len(load_cells_window("exclude", last=april)) < len(whole)
    assert len(load_cells_window("exclude", first=april)) < len(whole)
