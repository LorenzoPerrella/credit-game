"""What the book actually looks like, before any model is fitted to it.

A model is easier to trust when the reader has already seen the portfolio it was
estimated on: how many contracts were outstanding and when, how much was lent, what
was being written in which year, and what the economy was doing meanwhile. None of
this is modelling — it is the description that makes the modelling legible.

Everything reads the ingested parquet rather than the aggregated cells, because the
cells deliberately drop what this needs: balances are not in the grouping key, and
the loan count per period cannot be recovered from a table that has already collapsed
loans together.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.data.aggregate import PathSpec, _connect, _resolve

if TYPE_CHECKING:
    import duckdb

_LOGGER: Final = logging.getLogger(__name__)


def _run(
    query: str,
    perf: PathSpec,
    orig: PathSpec,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    per_quarter: bool = True,
) -> pd.DataFrame:
    """Run a descriptive query, quarter by quarter, and stack the results.

    Same reason the aggregation works this way: every loan lives in exactly one
    quarter's files, so the pieces stack and nothing has to be resident.
    """
    con = connection or _connect()
    perf_paths, orig_paths = _resolve(perf, "perf"), _resolve(orig, "orig")
    # Some descriptive queries read only the performance side. Passing both would
    # fail on the parameter count, so the query says how many it wants.
    wants_both = query.count("?") == 2

    if not per_quarter:
        arguments = [perf_paths, orig_paths] if wants_both else [perf_paths]
        return con.execute(query, arguments).df()

    frames = [
        con.execute(query, [[p], [o]] if wants_both else [[p]]).df()
        for p, o in zip(sorted(perf_paths), sorted(orig_paths), strict=True)
    ]
    return pd.concat(frames, ignore_index=True)


def outstanding_by_period(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Contracts outstanding and unpaid balance, month by month.

    The shape of the book over time: how many loans were on it, and how much was owed
    on them. A loan counts while it is still reporting a balance, which is the same
    condition the survival panel uses for being at risk.
    """
    query = """
    SELECT
        CAST(period AS INTEGER)                                 AS period_key,
        COUNT(*)                                                AS contracts,
        SUM(TRY_CAST(current_actual_upb AS DOUBLE))             AS balance
    FROM read_parquet(?)
    WHERE TRY_CAST(loan_age AS INTEGER) >= 0
      AND TRY_CAST(current_actual_upb AS DOUBLE) > 0
    GROUP BY 1
    """
    frame = _run(query, perf_source, orig_source, connection=connection)
    stacked = frame.groupby("period_key", as_index=False)[["contracts", "balance"]].sum()
    stacked["period"] = _to_period(stacked["period_key"])
    return stacked.sort_values("period").reset_index(drop=True)


def originations_by_period(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """New lending by month: how many loans were written, and for how much.

    Dated by first payment date, the only origination-side date the file carries. It
    falls one or two months after the loan was actually written, which matters for
    nothing here: the shape of a lending cycle is not changed by a constant shift.
    """
    query = """
    SELECT
        CAST(first_payment_date AS INTEGER)                     AS period_key,
        COUNT(*)                                                AS loans,
        SUM(TRY_CAST(original_upb AS DOUBLE))                   AS amount,
        AVG(NULLIF(TRY_CAST(classic_fico AS DOUBLE), 9999))     AS mean_score,
        AVG(NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999))      AS mean_ltv,
        AVG(NULLIF(TRY_CAST(original_dti AS DOUBLE), 999))      AS mean_dti,
        AVG(TRY_CAST(original_interest_rate AS DOUBLE))         AS mean_rate
    FROM read_parquet(?)
    WHERE first_payment_date IS NOT NULL
    GROUP BY 1
    """
    # Only the origination file is needed, but the runner's signature takes both.
    con = connection or _connect()
    frames = [con.execute(query, [[o]]).df() for o in sorted(_resolve(orig_source, "orig"))]
    frame = pd.concat(frames, ignore_index=True)

    weighted = frame.assign(
        score_weight=frame["mean_score"] * frame["loans"],
        ltv_weight=frame["mean_ltv"] * frame["loans"],
        dti_weight=frame["mean_dti"] * frame["loans"],
        rate_weight=frame["mean_rate"] * frame["loans"],
    )
    stacked = weighted.groupby("period_key", as_index=False)[
        ["loans", "amount", "score_weight", "ltv_weight", "dti_weight", "rate_weight"]
    ].sum()
    for name in ("score", "ltv", "dti", "rate"):
        stacked[f"mean_{name}"] = stacked[f"{name}_weight"] / stacked["loans"]
    stacked = stacked.drop(columns=[c for c in stacked.columns if c.endswith("_weight")])
    stacked["period"] = _to_period(stacked["period_key"])
    return stacked.sort_values("period").reset_index(drop=True)


def origination_mix(
    column: str,
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Share of new lending by category and vintage year.

    What was being written, and when. A mix that moves is the reason a model fitted on
    one decade can mislead about another, and it is worth seeing before the
    coefficients rather than after.
    """
    from creditsurv.data.aggregate import _CATEGORICAL

    expression = _CATEGORICAL[column]
    query = f"""
    SELECT
        CAST(first_payment_date AS INTEGER) // 100              AS year,
        {expression}                                            AS level,
        COUNT(*)                                                AS loans
    FROM read_parquet(?)
    WHERE first_payment_date IS NOT NULL
    GROUP BY 1, 2
    """
    con = connection or _connect()
    frames = [con.execute(query, [[o]]).df() for o in sorted(_resolve(orig_source, "orig"))]
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["level"])
    stacked: pd.DataFrame = (
        frame.groupby(["year", "level"], observed=True).agg(loans=("loans", "sum")).reset_index()
    )
    totals = stacked.groupby("year")["loans"].transform("sum")
    stacked["share"] = stacked["loans"] / totals
    ordered: pd.DataFrame = stacked.sort_values(["year", "level"]).reset_index(drop=True)
    return ordered


def covariate_evolution(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Quartiles of the underwriting covariates by vintage year.

    Distribution *and* drift in one table. A median that moves is a change in the book
    being written; a spread that narrows is a change in who is being lent to.
    """
    query = """
    SELECT
        CAST(first_payment_date AS INTEGER) // 100              AS year,
        QUANTILE_CONT(NULLIF(TRY_CAST(classic_fico AS DOUBLE), 9999), [0.25, 0.5, 0.75]) AS score,
        QUANTILE_CONT(NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999), [0.25, 0.5, 0.75])  AS ltv,
        QUANTILE_CONT(NULLIF(TRY_CAST(original_dti AS DOUBLE), 999), [0.25, 0.5, 0.75])  AS dti,
        COUNT(*)                                                AS loans
    FROM read_parquet(?)
    WHERE first_payment_date IS NOT NULL
    GROUP BY 1
    """
    con = connection or _connect()
    frames = [con.execute(query, [[o]]).df() for o in sorted(_resolve(orig_source, "orig"))]
    frame = pd.concat(frames, ignore_index=True)

    rows = []
    for year_label, group in frame.groupby("year"):
        weights = group["loans"].to_numpy(dtype=float)
        record: dict[str, object] = {
            "year": int(str(year_label)),
            "loans": int(weights.sum()),
        }
        for name in ("score", "ltv", "dti"):
            # A quarter where every value was a sentinel returns no quantiles at all,
            # so those rows are dropped before averaging rather than propagating.
            usable = [
                (quantiles, weight)
                for quantiles, weight in zip(group[name], weights, strict=True)
                if quantiles is not None and not pd.isna(pd.Series([quantiles])).all()
            ]
            if not usable:
                continue
            # Quantiles cannot be averaged exactly across quarters; weighting by loan
            # count is the usual approximation and is honest at this resolution.
            stacked = pd.DataFrame(
                [list(quantiles) for quantiles, _ in usable], columns=["q25", "q50", "q75"]
            )
            local = np.array([weight for _, weight in usable], dtype=float)
            for label in ("q25", "q50", "q75"):
                record[f"{name}_{label}"] = float(
                    (stacked[label].to_numpy(dtype=float) * local).sum() / local.sum()
                )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def default_rate_by_period(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Realised monthly default hazard across the book, by calendar month.

    The series the macro covariates are supposed to explain. Plotted against
    unemployment it is the whole argument for a time-varying model in one picture.
    """
    from creditsurv.data.aggregate import _state_of_the_book_sql

    query = f"""
    WITH book AS ({_state_of_the_book_sql()})
    SELECT period_key,
           COUNT(*)                                             AS loan_months,
           SUM(CASE WHEN event THEN 1 ELSE 0 END)               AS events
    FROM book GROUP BY 1
    """
    frame = _run(query, perf_source, orig_source, connection=connection)
    stacked = frame.groupby("period_key", as_index=False)[["loan_months", "events"]].sum()
    stacked["default_rate"] = stacked["events"] / stacked["loan_months"]
    stacked["period"] = _to_period(stacked["period_key"])
    return stacked.sort_values("period").reset_index(drop=True)


def _to_period(keys: pd.Series) -> pd.PeriodIndex:
    """YYYYMM integers to a monthly PeriodIndex."""
    text = keys.astype(int).astype(str)
    return pd.PeriodIndex(text.str[:4] + "-" + text.str[4:], freq="M")


def book_summary(
    lending: pd.DataFrame, outstanding: pd.DataFrame, defaults: pd.DataFrame
) -> dict[str, object]:
    """The numbers ``docs/portfolio.md`` opens with, from one pass and with their meaning.

    The validation's S7 crossed that document's default count, 1,906,460, with another
    document's 1,938,519. Each was right about a different run of a different panel, and
    neither said which. Every figure here comes from the same pass, and the two loan-month
    counts say what they count: every row the performance files report, against the
    loan-months the model is estimated on.
    """
    years = pd.PeriodIndex(lending["period"]).year
    return {
        "vintages": f"{int(years.min())} - {int(years.max())}",
        "loans_originated": int(lending["loans"].sum()),
        "amount_originated": float(lending["amount"].sum()),
        "loan_months_reported": int(outstanding["contracts"].sum()),
        "peak_contracts_outstanding": int(outstanding["contracts"].max()),
        "peak_balance_outstanding": float(outstanding["balance"].max()),
        "loan_months_modelled": int(defaults["loan_months"].sum()),
        "defaults_modelled": int(defaults["events"].sum()),
        "definition": (
            "Loan-months reported: every row of the performance files. Loan-months and "
            "defaults modelled: the book the cells are built from -- complete cases, cut at "
            "the first terminating month, with a moratorium not counted as a default."
        ),
    }
