"""Tests for the Freddie Mac loader.

The dataset itself is behind a registration, so these run against fixture files
written in the published format. That is the honest way to test a loader for gated
data: it exercises the parsing, the sentinels and the event logic without pretending
to have the data or trying to fetch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.data.freddiemac import (
    ORIGINATION_COLUMNS,
    FreddieMacDataMissingError,
    load_sample,
    read_origination,
    to_canonical_panel,
)
from creditsurv.data.panel import validate_episodes
from fixtures import origination_row, performance_row

if TYPE_CHECKING:
    from pathlib import Path


def write(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n")
    return path


def rows_for(panel: pd.DataFrame, loan_id: str) -> pd.DataFrame:
    """All rows of one loan.

    Boolean selection rather than ``set_index(...).loc[key]``: pandas-stubs cannot
    tell whether label selection yields a Series or a DataFrame, so the chained
    access it invites needs an ignore on every line.
    """
    return panel[panel["loan_id"] == loan_id].reset_index(drop=True)


def value_for(panel: pd.DataFrame, loan_id: str, column: str) -> object:
    return rows_for(panel, loan_id)[column].iloc[0]


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    """One performing loan, one that defaults, one that prepays."""
    origination = write(
        tmp_path / "sample_orig_2015.txt",
        [
            origination_row("F15Q1000001", state="NY", purpose="C"),
            origination_row("F15Q1000002", fico="620", ltv="95", state="TX"),
            origination_row("F15Q1000003", state="IL", channel="B"),
        ],
    )
    performance = write(
        tmp_path / "sample_svcg_2015.txt",
        [
            performance_row("F15Q1000001", "201503", "0"),
            performance_row("F15Q1000001", "201504", "1"),
            performance_row("F15Q1000001", "201505", "2"),
            # Defaults at age 2, then keeps reporting through foreclosure.
            performance_row("F15Q1000002", "201503", "0"),
            performance_row("F15Q1000002", "201504", "1", delinquency="1"),
            performance_row("F15Q1000002", "201505", "2", delinquency="3"),
            performance_row("F15Q1000002", "201506", "3", delinquency="4"),
            performance_row("F15Q1000002", "201507", "4", delinquency="RA", zero_balance="09"),
            # Prepays at age 1.
            performance_row("F15Q1000003", "201503", "0"),
            performance_row("F15Q1000003", "201504", "1", zero_balance="01"),
        ],
    )
    return origination, performance


def test_missing_files_point_at_the_registration(tmp_path: Path) -> None:
    """The data cannot be fetched, so the error has to say where to get it.

    There is no fallback source to suggest any more: this dataset is the project's
    data, so the message has to be enough to act on.
    """
    with pytest.raises(FreddieMacDataMissingError) as caught:
        read_origination(tmp_path / "absent.txt")

    message = str(caught.value)
    assert "claritydownload" in message
    assert "sample_orig" in message


def test_columns_are_read_positionally(files: tuple[Path, Path]) -> None:
    """The files carry no header, so a misplaced name shifts everything after it."""
    origination = read_origination(files[0])

    assert list(origination.columns) == list(ORIGINATION_COLUMNS)
    assert origination.loc[0, "loan_identifier"] == "F15Q1000001"
    assert origination.loc[0, "property_state"] == "NY"


def test_missing_value_sentinels_become_nan(tmp_path: Path) -> None:
    """9999 is not a credit score, and left alone it produces a portfolio whose
    average score is several thousand."""
    path = write(
        tmp_path / "orig.txt", [origination_row("F15Q1000009", fico="9999", debt_to_income="999")]
    )

    origination = read_origination(path)

    # bool(...) defeats the TypeGuard on pd.isna, which otherwise narrows the value
    # to NaType and makes every following assertion look unreachable.
    assert bool(pd.isna(origination["classic_fico"].iloc[0]))
    assert bool(pd.isna(origination["original_dti"].iloc[0]))


def test_panel_satisfies_the_canonical_invariants(files: tuple[Path, Path]) -> None:
    validate_episodes(load_sample(*files))


def test_origination_month_is_recovered_from_age(files: tuple[Path, Path]) -> None:
    """The dataset has no origination date, only a first payment date one or two
    months later, so age is the reliable anchor."""
    panel = load_sample(*files)

    assert (panel["origination_period"] == panel["period"] - panel["age"]).all()
    assert str(panel["origination_period"].iloc[0]) == "2015-03"


def test_default_is_flagged_once_despite_continued_reporting(
    files: tuple[Path, Path],
) -> None:
    """Servicing files keep reporting through foreclosure and loss settlement.

    Left alone the loan carries three defaulted rows, breaking the one-event
    invariant and counting a single default three times in the likelihood.
    """
    defaulted = rows_for(load_sample(*files), "F15Q1000002")

    assert int(defaulted["event"].sum()) == 1
    assert defaulted["age"].max() == 2  # cut at the first defaulted month
    assert bool(defaulted.loc[defaulted["age"] == 2, "event"].iloc[0])


def test_prepayment_is_censoring_not_an_event(files: tuple[Path, Path]) -> None:
    prepaid = rows_for(load_sample(*files), "F15Q1000003")

    assert not prepaid["event"].any()
    assert int(prepaid["prepaid"].sum()) == 1


def test_a_performing_loan_records_no_event(files: tuple[Path, Path]) -> None:
    performing = rows_for(load_sample(*files), "F15Q1000001")

    assert not performing["event"].any()
    assert not performing["prepaid"].any()
    assert len(performing) == 3


def test_categorical_codes_are_decoded(files: tuple[Path, Path]) -> None:
    panel = load_sample(*files)

    assert value_for(panel, "F15Q1000001", "purpose") == "cash_out_refinance"
    assert value_for(panel, "F15Q1000003", "channel") == "broker_or_correspondent"
    assert value_for(panel, "F15Q1000001", "occupancy") == "owner_occupied"


def test_states_map_to_census_regions(files: tuple[Path, Path]) -> None:
    """Four regions rather than fifty states: the design matrix multiplies against
    a panel of hundreds of thousands of rows."""
    panel = load_sample(*files)

    assert value_for(panel, "F15Q1000001", "region") == "Northeast"
    assert value_for(panel, "F15Q1000002", "region") == "South"
    assert value_for(panel, "F15Q1000003", "region") == "Midwest"


def test_loans_missing_covariates_are_dropped_not_imputed(tmp_path: Path) -> None:
    """Imputing an underwriting characteristic invents the thing being measured."""
    origination = write(
        tmp_path / "o.txt",
        [origination_row("F15Q1000001"), origination_row("F15Q1000002", fico="9999")],
    )
    performance = write(
        tmp_path / "p.txt",
        [
            performance_row("F15Q1000001", "201503", "0"),
            performance_row("F15Q1000002", "201503", "0"),
        ],
    )

    panel = load_sample(origination, performance)

    assert set(panel["loan_id"]) == {"F15Q1000001"}


def test_negative_loan_ages_are_dropped(tmp_path: Path) -> None:
    """The dataset can report a month before the first payment date."""
    origination = write(tmp_path / "o.txt", [origination_row("F15Q1000001")])
    performance = write(
        tmp_path / "p.txt",
        [
            performance_row("F15Q1000001", "201502", "-1"),
            performance_row("F15Q1000001", "201503", "0"),
            performance_row("F15Q1000001", "201504", "1"),
        ],
    )

    panel = load_sample(origination, performance)

    assert panel["age"].min() == 0
    assert len(panel) == 2


def test_frames_can_be_mapped_without_touching_disk(files: tuple[Path, Path]) -> None:
    """to_canonical_panel takes frames, so a caller with the full dataset can chunk
    it rather than loading fifty million rows at once."""
    from creditsurv.data.freddiemac import read_performance

    panel = to_canonical_panel(read_origination(files[0]), read_performance(files[1]))

    assert not panel.empty
