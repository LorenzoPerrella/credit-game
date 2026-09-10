"""Tests for the DuckDB aggregation.

Small fixtures, real SQL. The point is not volume but that the event definition, the
truncation and the collapse are right — none of which needs a billion rows to check,
and all of which are impossible to check on one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    assert set(cells["n"]) == {20}
    assert int(cells["n"].sum()) == 60


def test_weights_account_for_every_loan_month(tmp_path: Path) -> None:
    origination = [origination_row(f"F{i:09d}", fico=str(620 + i * 20)) for i in range(6)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(6) for age in range(4)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 24


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

    assert int(cells["n"].sum()) == 3, "the loan should stop at its first defaulted month"
    assert int(cells.loc[cells["event"], "n"].sum()) == 1
    assert int(cells.loc[cells["event"], "age"].iloc[0]) == 2


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

    assert int(cells.loc[cells["event"], "n"].sum()) == 1


def test_prepayment_is_censoring_not_an_event(tmp_path: Path) -> None:
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", zero_balance="01"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert not cells["event"].any()
    assert int(cells["n"].sum()) == 2


def test_sentinel_values_drop_the_loan(tmp_path: Path) -> None:
    """9999 is not a credit score, and a loan without one cannot be modelled."""
    origination = [origination_row("F000000001", fico="9999"), origination_row("F000000002")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 1


def test_negative_ages_are_dropped(tmp_path: Path) -> None:
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201502", "-1"),
        performance_row("F000000001", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 1
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
    assert int(cells["n"].iloc[0]) == 2


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
        spec=CellSpec(continuous={"fico_s": (-3.0, 3.0)}, categorical=()),
    )

    assert len(narrow) < len(wide)
    assert int(narrow["n"].sum()) == int(wide["n"].sum()) == 24


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

    assert cells["event"].dtype == bool
    assert not cells["event"].isna().any()
    assert int(cells.loc[cells["event"], "n"].sum()) == 1


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

    assert int(cells["n"].sum()) == 1
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

    assert set(cells["channel"]) == {"retail", "third_party"}
    assert int(cells.loc[cells["channel"] == "third_party", "n"].sum()) == 3
