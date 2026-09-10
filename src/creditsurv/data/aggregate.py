"""Collapsing 1.75 billion loan-months into a table a model can be fitted to.

Episodes that agree on every covariate and on their position in time are
exchangeable, so they can be replaced by one row carrying a count. That is the whole
trick, and at this scale it is not an optimisation but the only thing that makes the
problem tractable: a fit over two billion rows is out of reach, a fit over a million
weighted cells is a minute.

The work is done in DuckDB rather than pandas because the join, the truncation and
the group-by all have to happen out of core — the inputs are far larger than memory
and never need to be resident.

**The macro series are deliberately absent from the grouping key.** Since
``period = orig_period + age``, unemployment, house prices and financial conditions
are a deterministic function of two columns that are already in the key, so they can
be recomputed on the aggregate at no cost in cardinality. Putting them in the key
instead would multiply it by the number of distinct months and destroy the collapse.

An earlier measurement in this project found the same technique compressing
1.00x and concluded it was not worth having. That was true at 215,000 rows, where
the possible cells vastly outnumbered the rows. At 1.75 billion the ratio inverts.
The measurement was right for the regime it was taken in.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final, TypeAlias

import duckdb
import pandas as pd

from creditsurv.data.ingest import completed_files

_LOGGER: Final = logging.getLogger(__name__)

#: What the readers accept: nothing (use the manifest), one path, or many.
PathSpec: TypeAlias = str | Path | Sequence[str] | None

#: Delinquency at which a loan counts as defaulted: three missed payments.
DEFAULT_DELINQUENCY: Final = 3

#: Zero-balance codes that end a loan through credit loss rather than repayment.
DEFAULT_ZERO_BALANCE: Final = ("02", "03", "09", "15")
PREPAYMENT_ZERO_BALANCE: Final = "01"

#: Upper edges of the loan-age bands, in months. Episodes become the intervals
#: between them rather than single months.
#:
#: Widening with age on purpose. The hazard moves fastest in the first two years and
#: flattens afterwards, so fine resolution early costs little and buys the shape,
#: while a single band covering years eight to twelve loses almost nothing. Measured
#: on 1999Q1: monthly ages give 12.8 million cells, quarterly gives *exactly the
#: same* -- because the time-varying covariate moves anyway -- and these bands give
#: 920 thousand. Age was the wrong lever until it was made coarse enough to matter.
AGE_BANDS: Final[tuple[int, ...]] = (6, 12, 24, 36, 60, 96, 144)

_STATE_TO_REGION: Final[dict[str, str]] = {}
for _region, _states in {
    "Northeast": ("CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"),
    "Midwest": ("IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"),
    "South": (
        "AL",
        "AR",
        "DE",
        "DC",
        "FL",
        "GA",
        "KY",
        "LA",
        "MD",
        "MS",
        "NC",
        "OK",
        "SC",
        "TN",
        "TX",
        "VA",
        "WV",
    ),
    "West": ("AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"),
}.items():
    for _state in _states:
        _STATE_TO_REGION[_state] = _region


def _case_expression(column: str, edges: Sequence[float], alias: str) -> str:
    """Render a coarse-classing rule as SQL.

    The band's midpoint is used as its value, so a binned covariate keeps the scale
    of the one it replaces and its coefficient stays comparable with an unbinned fit.
    """
    clauses = []
    for lower, upper in pairwise(edges):
        midpoint = (lower + upper) / 2.0
        clauses.append(f"WHEN {column} <= {upper} THEN {midpoint}")
    outer = (edges[0] + edges[1]) / 2.0
    last = (edges[-2] + edges[-1]) / 2.0
    return (
        f"CASE WHEN {column} IS NULL THEN NULL "
        f"WHEN {column} <= {edges[0]} THEN {outer} "
        + " ".join(clauses)
        + f" ELSE {last} END AS {alias}"
    )


def _region_case() -> str:
    whens = " ".join(
        f"WHEN property_state = '{state}' THEN '{region}'"
        for state, region in _STATE_TO_REGION.items()
    )
    return f"CASE {whens} ELSE 'Other' END AS region"


def _state_of_the_book_sql() -> str:
    """The loan-month panel, cleaned and truncated, before any aggregation."""
    default_codes = ", ".join(f"'{code}'" for code in DEFAULT_ZERO_BALANCE)
    return f"""
    WITH perf AS (
        SELECT
            loan_identifier,
            CAST(loan_age AS INTEGER)                                   AS age,
            CAST(period AS INTEGER)                                     AS period_key,
            TRY_CAST(estimated_loan_to_value AS DOUBLE)                 AS eltv,
            -- COALESCE wraps the whole expression, not just the cast. Most rows
            -- have no zero-balance code, so `IN (...)` is NULL there, and
            -- `FALSE OR NULL` is NULL rather than FALSE -- which propagates a
            -- nullable event flag all the way to the fitter. The fixtures did not
            -- catch it because they write an empty string where the real files
            -- leave the field absent.
            COALESCE(
                COALESCE(TRY_CAST(current_loan_delinquency_status AS INTEGER)
                         >= {DEFAULT_DELINQUENCY}, FALSE)
                OR COALESCE(zero_balance_code IN ({default_codes}), FALSE),
                FALSE
            )                                                           AS defaulted,
            COALESCE(zero_balance_code = '{PREPAYMENT_ZERO_BALANCE}', FALSE) AS prepaid
        FROM read_parquet(?)
        WHERE TRY_CAST(loan_age AS INTEGER) >= 0
    ),
    -- Servicing files keep reporting through foreclosure and loss settlement, so a
    -- defaulted loan carries several flagged rows. Cutting at the first terminating
    -- month is what keeps one event per loan.
    terminal AS (
        SELECT loan_identifier, MIN(age) AS terminal_age
        FROM perf WHERE defaulted OR prepaid GROUP BY loan_identifier
    ),
    truncated AS (
        SELECT p.*, t.terminal_age
        FROM perf p LEFT JOIN terminal t USING (loan_identifier)
        WHERE t.terminal_age IS NULL OR p.age <= t.terminal_age
    ),
    orig AS (
        SELECT
            loan_identifier,
            -- 9999 and 999 are the dataset's own "not available" markers. They are
            -- ordinary numbers, and left in place they produce a portfolio whose
            -- average credit score is several thousand.
            NULLIF(TRY_CAST(classic_fico AS DOUBLE), 9999)              AS credit_score,
            NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999)               AS orig_ltv,
            NULLIF(TRY_CAST(original_cltv AS DOUBLE), 999)              AS orig_cltv,
            NULLIF(TRY_CAST(original_dti AS DOUBLE), 999)               AS dti,
            TRY_CAST(original_upb AS DOUBLE)                            AS orig_upb,
            TRY_CAST(original_interest_rate AS DOUBLE)                  AS note_rate,
            TRY_CAST(original_loan_term AS INTEGER)                     AS orig_term,
            TRY_CAST(mortgage_insurance_percentage AS DOUBLE)           AS mi_percent,
            TRY_CAST(number_of_borrowers AS INTEGER)                    AS n_borrowers,
            CASE loan_purpose WHEN 'P' THEN 'purchase'
                              WHEN 'C' THEN 'refinance_cashout'
                              ELSE 'refinance_rate_term' END            AS purpose,
            CASE occupancy_status WHEN 'P' THEN 'owner_occupied'
                                  WHEN 'S' THEN 'second_home'
                                  ELSE 'investor' END                   AS occupancy,
            CASE channel WHEN 'R' THEN 'retail' WHEN 'B' THEN 'broker'
                         ELSE 'correspondent' END                       AS channel,
            CASE WHEN first_time_homebuyer_indicator = 'Y' THEN 'Y' ELSE 'N' END
                                                                        AS first_time_buyer,
            property_type,
            {_region_case()}
        FROM read_parquet(?)
    )
    SELECT
        t.age,
        t.period_key,
        t.eltv,
        COALESCE(t.defaulted AND t.age = t.terminal_age, FALSE)         AS event,
        COALESCE(t.prepaid AND t.age = t.terminal_age, FALSE)           AS prepaid,
        o.*
    FROM truncated t JOIN orig o USING (loan_identifier)
    WHERE o.credit_score IS NOT NULL AND o.orig_ltv IS NOT NULL AND o.dti IS NOT NULL
    """


#: Source expression for each continuous covariate, keyed by the name it takes.
_SOURCE: Final[dict[str, str]] = {
    "fico_s": "(credit_score - 700.0) / 50.0",
    "orig_ltv": "orig_ltv",
    "orig_cltv": "orig_cltv",
    "dti": "dti",
    "log_orig_upb": "ln(orig_upb)",
    # Freddie's own mark-to-market valuation, which is loan-specific. Falling back to
    # the original ratio where it is missing makes the drift zero rather than null,
    # which is the right default: no information about movement means no movement.
    "cltv_drift": "COALESCE(eltv, orig_ltv) - orig_ltv",
    "mi_percent": "mi_percent",
}

#: Categorical covariates and the SQL that produces them.
_CATEGORICAL: Final[dict[str, str]] = {
    "purpose": "purpose",
    "occupancy": "occupancy",
    "channel": "channel",
    "region": "region",
    "first_time_buyer": "first_time_buyer",
    "property_type": "property_type",
    "term_years": "CASE WHEN orig_term <= 190 THEN 15 ELSE 30 END",
    "has_mi": "CASE WHEN mi_percent > 0 THEN 'Y' ELSE 'N' END",
    "n_borrowers": "CASE WHEN n_borrowers >= 2 THEN 2 ELSE 1 END",
}


@dataclass(frozen=True)
class CellSpec:
    """Which covariates enter the aggregation, and how coarsely.

    Deliberately a parameter rather than a constant. The cell count is the product of
    every covariate's band count, so the specification *is* the cardinality, and it
    cannot be fixed before variable selection has said which covariates earn their
    place. Aggregating on everything available and selecting afterwards is the wrong
    order: it produces a table too large to fit, which is exactly what happened here
    on the first attempt.

    Measured on 1999Q1 (27.7 million loan-months):

    ==========================================  ==========  ============
    Specification                               Cells       Compression
    ==========================================  ==========  ============
    9 continuous + 9 categorical, monthly ages  12,752,331          2.2x
    same, quarterly ages                        12,752,331          2.2x
    same, banded ages                              919,634         30.1x
    4 coarse continuous + 3 categorical, banded     14,221      1,948.0x
    ==========================================  ==========  ============
    """

    continuous: dict[str, tuple[float, ...]]
    categorical: tuple[str, ...]
    age_bands: tuple[int, ...] = AGE_BANDS

    def validate(self) -> None:
        unknown = set(self.continuous) - set(_SOURCE)
        if unknown:
            message = f"Unknown continuous covariate(s): {sorted(unknown)}"
            raise ValueError(message)
        unknown = set(self.categorical) - set(_CATEGORICAL)
        if unknown:
            message = f"Unknown categorical covariate(s): {sorted(unknown)}"
            raise ValueError(message)


#: Coarse by design: five bands a covariate, three categoricals, banded ages. Chosen
#: from the measurement above as the point where the whole population fits in roughly
#: a million cells. Variable selection may replace it.
DEFAULT_SPEC: Final = CellSpec(
    continuous={
        "fico_s": (-2.4, -0.8, 0.0, 0.8, 1.6, 2.4),
        "orig_ltv": (30.0, 70.0, 80.0, 90.0, 100.0),
        "dti": (10.0, 28.0, 36.0, 45.0, 55.0),
        "cltv_drift": (-60.0, -15.0, -5.0, 5.0, 15.0, 80.0),
    },
    categorical=("purpose", "occupancy", "term_years"),
)


def _age_band_expression(bands: tuple[int, ...]) -> str:
    """Render the age bands as SQL, returning the band's lower edge in months.

    The lower edge rather than an index, so the value keeps the units of loan age and
    the episode bounds can be read straight off it.
    """
    clauses = []
    previous = 0
    for upper in bands:
        clauses.append(f"WHEN age < {upper} THEN {previous}")
        previous = upper
    return f"CASE {' '.join(clauses)} ELSE {previous} END AS age"


def _select_columns(spec: CellSpec) -> str:
    """Every covariate column of the SELECT, as SQL.

    Assembled from a list rather than interpolated as separate blocks: an empty
    continuous or categorical set would otherwise leave a dangling comma and fail
    with a parser error that says nothing about the specification that caused it.
    """
    columns = [
        _case_expression(_SOURCE[name], edges, name) for name, edges in spec.continuous.items()
    ]
    columns += [f"{_CATEGORICAL[name]} AS {name}" for name in spec.categorical]
    columns.append(_age_band_expression(spec.age_bands))
    columns.append("event")
    return ",\n            ".join(columns)


def _connect() -> duckdb.DuckDBPyConnection:
    """A connection that will not fill the working tree with spill files.

    DuckDB defaults its temporary directory to the process's working directory, so a
    query that spills leaves gigabytes inside the repository -- 20 GB, the first time
    this ran. Quarters are processed one at a time now and it should not spill at
    all, but the setting is cheap insurance against the next query that does.
    """
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{tempfile.gettempdir()}'")
    con.execute("SET memory_limit = '6GB'")
    return con


def _resolve(spec: PathSpec, kind: str) -> list[str]:
    """Turn a caller's argument into a concrete list of parquet paths."""
    if spec is None:
        return completed_files(kind)
    if isinstance(spec, str | Path):
        return [str(spec)]
    return [str(path) for path in spec]


def _cells_for_quarter(
    connection: duckdb.DuckDBPyConnection,
    perf_path: str,
    orig_path: str,
    vintage: str,
    spec: CellSpec,
) -> pd.DataFrame:
    """Aggregate one vintage quarter.

    Every loan appears in exactly one quarter's files -- verified, not assumed: the
    identifiers of 1999Q1 and 1999Q2 do not intersect at all. So a quarter can be
    collapsed on its own and the results concatenated, which keeps memory flat.

    The vintage is attached as a constant rather than derived, because it is already
    known from the file name. With vintage and age in the key the observation month
    follows, which is what lets the macro series stay out of the key entirely.
    """
    query = f"""
    WITH book AS ({_state_of_the_book_sql()}), classed AS (
        SELECT
            {_select_columns(spec)}
        FROM book
    )
    SELECT '{vintage}' AS vintage, *, COUNT(*) AS n
    FROM classed
    GROUP BY ALL
    """
    frame: pd.DataFrame = connection.execute(query, [perf_path, orig_path]).df()
    return frame


def build_cells(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Aggregate the parquet panel into weighted cells, one quarter at a time.

    The key is the coarse-classed covariates together with vintage quarter, loan age
    and the event flag. The weight is the loan-month count.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    spec.validate()
    con = connection or _connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        cells = _cells_for_quarter(con, perf_path, orig_path, vintage, spec)
        frames.append(cells)
        _LOGGER.info("%s: %d cells from %d loan-months", vintage, len(cells), int(cells["n"].sum()))

    # Vintage is in the key and constant within a quarter, so the pieces are already
    # disjoint: concatenating needs no second group-by.
    combined = pd.concat(frames, ignore_index=True)
    _LOGGER.info("Collapsed to %d cells", len(combined))
    return combined


def cardinality_report(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """How far the collapse actually gets, key by key.

    Run before fixing the grain, not after. The plan calls for reducing the vintage
    to a year, or the episode to a quarter, if the cells run past what a fit can
    carry — and that decision needs a number rather than an intuition.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    con = _connect()

    counted = con.execute(
        f"SELECT COUNT(*) FROM ({_state_of_the_book_sql()})", [perf, orig]
    ).fetchone()
    rows = int(counted[0]) if counted else 0
    cells = build_cells(perf, orig, spec=spec, connection=con)
    return pd.DataFrame(
        [
            {
                "loan_months": rows,
                "cells": len(cells),
                "compression": rows / max(len(cells), 1),
                "weight_total": int(cells["n"].sum()),
            }
        ]
    )
