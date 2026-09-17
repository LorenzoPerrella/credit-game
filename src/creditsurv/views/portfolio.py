"""The portfolio, opened by segment.

Two passes over the book, quarter by quarter, each grouping by every segment at once with
``GROUPING SETS`` rather than scanning the performance files once per segment:

* :func:`book_by_segment` -- loan-months, defaults and prepayments by calendar month, from
  which the monthly default rate and the conditional prepayment rate follow;
* :func:`lending_by_segment` -- loans and amounts written, by vintage year.

Both read the book the cells are built from, ``_state_of_the_book_sql``, with its cleaning,
truncation and event definition, and label their segments with the same
:mod:`creditsurv.views.segments` the calibration views use -- so a loan-month is "620 to 660"
on the portfolio page exactly when it is on the calibration page. The book is taken whole
here (``complete_only=False``): the portfolio is what was lent, including the loans the model
leaves out, and the page says so.

:func:`vintage_curves` reads the cells instead: cumulative default by vintage year and age.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.data.aggregate import (
    _CATEGORICAL,
    _SOURCE,
    PRODUCTION_EDGES,
    MoratoriumPolicy,
    PathSpec,
    _case_expression,
    _connect,
    _resolve,
    _state_of_the_book_sql,
)
from creditsurv.data.panel import origination_months
from creditsurv.views.calibration import WHOLE_BOOK, survival_by_age
from creditsurv.views.segments import SEGMENTS

if TYPE_CHECKING:
    import duckdb

#: Categorical fields of the book the portfolio is opened by, beyond the model's own.
PORTFOLIO_CATEGORICAL: Final[tuple[str, ...]] = (
    "purpose",
    "occupancy",
    "channel",
    "region",
    "property_type",
    "first_time_buyer",
    "has_mi",
    "term_years",
)

#: Banded fields, labelled after the query by the segment of the same grid.
PORTFOLIO_BANDED: Final[dict[str, str]] = {"fico_s": "fico", "orig_ltv": "ltv", "dti": "dti"}

#: How each column of the book is labelled: a segment where one exists, the level as it is
#: mapped otherwise.
_SEGMENT_OF: Final[dict[str, str]] = {
    "purpose": "purpose",
    "occupancy": "occupancy",
    "first_time_buyer": "first_time_buyer",
    "has_mi": "has_mi",
    "term_years": "term",
    **PORTFOLIO_BANDED,
}

_ORIGINATION_YEAR: Final = "((period_key // 100) * 12 + (period_key % 100) - 1 - age) // 12"


def _labelled_columns() -> list[str]:
    columns = [f"{_CATEGORICAL[name]} AS {name}" for name in PORTFOLIO_CATEGORICAL]
    columns += [
        _case_expression(_SOURCE[name], PRODUCTION_EDGES[name], name) for name in PORTFOLIO_BANDED
    ]
    return columns


def _grouped(fields: list[str]) -> list[str]:
    return [*PORTFOLIO_CATEGORICAL, *PORTFOLIO_BANDED, *fields]


def _long(frame: pd.DataFrame, key: str, fields: list[str]) -> pd.DataFrame:
    """The grouping sets as one long table: key, segment, group, and the sums."""
    measures = [
        column
        for column in frame.columns
        if column not in {key, *fields} and not column.startswith("g_")
    ]
    pieces = []
    whole = frame[[f"g_{name}" for name in fields]].eq(1).all(axis=1)
    overall = frame.loc[whole, [key, *measures]].assign(segment="all", group=WHOLE_BOOK)
    pieces.append(overall)
    for name in fields:
        rows = frame.loc[frame[f"g_{name}"].eq(0), [key, name, *measures]]
        segment = _SEGMENT_OF.get(name, name)
        if segment in SEGMENTS:
            labels = SEGMENTS[segment].label(
                rows[[name]].rename(columns={name: SEGMENTS[segment].columns[0]})
            )
            group = labels.astype(str).where(labels.notna(), "not mapped")
        else:
            group = rows[name].astype(str).where(rows[name].notna(), "not mapped")
        pieces.append(rows[[key, *measures]].assign(segment=segment, group=group.to_numpy()))
    stacked = pd.concat(pieces, ignore_index=True)
    summed: pd.DataFrame = stacked.groupby([key, "segment", "group"], as_index=False)[
        measures
    ].sum()
    return summed


def book_by_segment(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Loan-months, defaults and prepayments by calendar month, for every segment.

    The rates follow from the counts: the monthly default rate in basis points, the single
    monthly mortality of prepayment, and its annualised form, the conditional prepayment
    rate ``1 - (1 - smm)^12`` a mortgage desk reads.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)
    fields = _grouped(["vintage_year"])
    gaps = ", ".join(f"GROUPING({name}) AS g_{name}" for name in fields)
    sets = ", ".join(f"(period_key, {name})" for name in fields)
    query = f"""
    WITH book AS ({_state_of_the_book_sql(policy, complete_only=False)}),
    labelled AS (
        SELECT period_key, event, prepaid, {", ".join(_labelled_columns())},
               {_ORIGINATION_YEAR} AS vintage_year
        FROM book
    )
    SELECT period_key, {gaps}, {", ".join(fields)},
           COUNT(*) AS loan_months,
           SUM(CAST(event AS INTEGER)) AS defaults,
           SUM(CAST(prepaid AS INTEGER)) AS prepayments
    FROM labelled
    GROUP BY GROUPING SETS ({sets}, (period_key))
    """
    con = connection or _connect()
    frames = [
        con.execute(query, [perf_path, orig_path]).df()
        for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True)
    ]
    table = _long(pd.concat(frames, ignore_index=True), "period_key", fields)
    keys = table["period_key"].astype(int)
    table.insert(
        0, "month", (keys // 100).astype(str) + "-" + (keys % 100).astype(str).str.zfill(2)
    )
    table = table.drop(columns="period_key")
    table["default_rate_bp"] = table["defaults"] / table["loan_months"] * 1e4
    table["smm"] = table["prepayments"] / table["loan_months"]
    table["cpr"] = 1.0 - (1.0 - table["smm"]) ** 12
    ordered: pd.DataFrame = table.sort_values(["segment", "group", "month"]).reset_index(drop=True)
    return ordered


def lending_by_segment(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Loans and amounts written, by vintage year, for every segment, with their shares.

    Each loan counted once, with the attributes and the origination year of its first month
    in the book.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)
    fields = _grouped([])
    firsts = ", ".join(f"ARG_MIN({name}, period_key) AS {name}" for name in fields)
    gaps = ", ".join(f"GROUPING({name}) AS g_{name}" for name in fields)
    sets = ", ".join(f"(vintage_year, {name})" for name in fields)
    query = f"""
    WITH book AS ({_state_of_the_book_sql(MoratoriumPolicy.EXCLUDE, complete_only=False)}),
    labelled AS (
        SELECT loan_identifier, period_key, orig_upb, {", ".join(_labelled_columns())},
               {_ORIGINATION_YEAR} AS vintage_year
        FROM book
    ),
    loans AS (
        SELECT loan_identifier, ARG_MIN(vintage_year, period_key) AS vintage_year,
               ARG_MIN(orig_upb, period_key) AS orig_upb, {firsts}
        FROM labelled
        GROUP BY loan_identifier
    )
    SELECT vintage_year, {gaps}, {", ".join(fields)},
           COUNT(*) AS loans, SUM(orig_upb) AS amount
    FROM loans
    GROUP BY GROUPING SETS ({sets}, (vintage_year))
    """
    con = connection or _connect()
    frames = [
        con.execute(query, [perf_path, orig_path]).df()
        for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True)
    ]
    table = _long(pd.concat(frames, ignore_index=True), "vintage_year", fields)
    table = table.rename(columns={"vintage_year": "year"})
    table["year"] = table["year"].astype(int)
    totals = table[table["segment"] == "all"].set_index("year")[["loans", "amount"]]
    table["loan_share"] = table["loans"] / table["year"].map(totals["loans"])
    table["amount_share"] = table["amount"] / table["year"].map(totals["amount"])
    ordered: pd.DataFrame = table.sort_values(["segment", "group", "year"]).reset_index(drop=True)
    return ordered


def vintage_curves(cells: pd.DataFrame) -> pd.DataFrame:
    """Cumulative default by vintage year and loan age, from the cells.

    The Kaplan-Meier of each vintage year on its own risk sets: the curves every mortgage
    report opens with, and the plainest picture of what the macro covariates have to explain
    -- vintages written into 2006 against those written into 2012.
    """
    years = pd.Series(origination_months(cells).to_numpy() // 12, index=cells.index)
    table = survival_by_age(cells, groups=years)
    table = table.rename(columns={"group": "vintage_year"})
    table["vintage_year"] = table["vintage_year"].astype(int)
    table["cumulative_default"] = 1.0 - table["km_survival"]
    return table.drop(columns=["km_lower", "km_upper"]).assign(
        cumulative_default_pct=lambda frame: frame["cumulative_default"] * 100.0
    )
