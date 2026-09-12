"""Exploratory statistics, as functions rather than notebook cells.

Everything a notebook shows about the data is computed here and tested. A notebook
should read like a report — table, chart, conclusion — and an algorithm living in a
cell is an algorithm nobody runs twice and nobody checks.

**Every statistic is exposure-weighted.** The panel is aggregated into cells carrying
a loan-month count, so an unweighted mean over cells would weight a cell holding six
loan-months the same as one holding sixty thousand. The weight column is not optional
here; it is what the row means.

The default-rate convention follows `nmds`: events divided by exposure, which is a
monthly hazard rather than a share of loans. Two bands can hold the same number of
defaults and differ entirely in risk if one was watched ten times as long.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Default weight column, produced by the aggregation.
WEIGHT: Final = "n"

#: A category holding this much of the exposure leaves nothing to estimate from the
#: rest. Deliberately 99% rather than the 90% one might reach for: a covariate that is
#: 87% owner-occupied still has 13% of a very large book saying something.
DOMINANCE_THRESHOLD: Final = 0.99

#: A field missing this often cannot support a coefficient. Same rule as `nmds`.
MISSING_THRESHOLD: Final = 0.99


def fill_rate(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Share of rows where each column is present, worst first.

    Run before anything else. Several fields in this dataset are empty in every
    vintage, and a covariate that is absent is not a covariate.
    """
    selected = list(columns) if columns is not None else list(frame.columns)
    rows = [
        {
            "column": name,
            "filled": float(frame[name].notna().mean()),
            "missing": float(frame[name].isna().mean()),
        }
        for name in selected
    ]
    table = pd.DataFrame(rows).sort_values("filled").reset_index(drop=True)
    table["verdict"] = np.where(
        table["missing"] > MISSING_THRESHOLD, "drop: mostly missing", "usable"
    )
    return table


def frequency_table(frame: pd.DataFrame, column: str, *, weight: str = WEIGHT) -> pd.DataFrame:
    """Exposure share of each level, including a row for missing.

    Missing is always shown, even when it is zero, so its absence is visible rather
    than inferred — the convention `nmds` uses, and a good one: a table that omits
    the row leaves the reader unable to tell "none missing" from "not checked".
    """
    exposure = frame[weight].to_numpy(dtype=float)
    values = frame[column].astype("object").where(frame[column].notna(), "<missing>")

    grouped = (
        pd.DataFrame({"level": values, "exposure": exposure})
        .groupby("level", observed=True)["exposure"]
        .sum()
    )
    if "<missing>" not in grouped.index:
        grouped.loc["<missing>"] = 0.0

    table = grouped.reset_index(name="loan_months")
    table["share"] = table["loan_months"] / table["loan_months"].sum()
    ordered = table[table["level"] != "<missing>"].sort_values("share", ascending=False)
    missing = table[table["level"] == "<missing>"]
    return pd.concat([ordered, missing], ignore_index=True)


def concentration_report(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    weight: str = WEIGHT,
    threshold: float = DOMINANCE_THRESHOLD,
) -> pd.DataFrame:
    """How much exposure sits in each covariate's largest level.

    A covariate concentrated past the threshold is degenerate: there is nothing left
    outside the dominant level to estimate a contrast from. Below it, a lopsided
    covariate is merely lopsided, which is not the same thing and not a reason to
    drop it.
    """
    rows = []
    for name in columns:
        table = frequency_table(frame, name, weight=weight)
        populated = table[table["level"] != "<missing>"]
        largest = float(populated["share"].max()) if len(populated) else 1.0
        rows.append(
            {
                "column": name,
                "levels": len(populated),
                "largest_share": largest,
                "largest_level": (
                    str(populated.loc[populated["share"].idxmax(), "level"])
                    if len(populated)
                    else "<none>"
                ),
                "degenerate": largest >= threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("largest_share", ascending=False).reset_index(drop=True)


def default_rate_by_band(
    frame: pd.DataFrame,
    column: str,
    *,
    weight: str = WEIGHT,
    event: str = "event",
) -> pd.DataFrame:
    """Monthly default hazard by level of a covariate.

    Events over exposure, not defaults over loans. Two bands can hold the same number
    of defaults and differ entirely in risk if one was watched ten times as long, and
    on a panel spanning twenty-five vintages that difference is the rule.
    """
    exposure = frame[weight].to_numpy(dtype=float)
    events = exposure * frame[event].to_numpy(dtype=bool)

    grouped = (
        pd.DataFrame({"level": frame[column].to_numpy(), "exposure": exposure, "events": events})
        .groupby("level", observed=True)[["exposure", "events"]]
        .sum()
    )
    grouped["default_rate"] = grouped["events"] / grouped["exposure"]
    grouped["share"] = grouped["exposure"] / grouped["exposure"].sum()
    return grouped.reset_index().sort_values("level").reset_index(drop=True)


def weighted_correlation(
    frame: pd.DataFrame, columns: Sequence[str], *, weight: str = WEIGHT
) -> pd.DataFrame:
    """Exposure-weighted Pearson correlation between continuous covariates.

    Weighted because the rows are cells: an unweighted matrix would describe the
    distribution of *cells*, which is an artefact of the binning, rather than the
    distribution of loan-months, which is the data.
    """
    exposure = frame[weight].to_numpy(dtype=float)
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    total = exposure.sum()

    means = (values * exposure[:, None]).sum(axis=0) / total
    centred = values - means
    covariance = (centred * exposure[:, None]).T @ centred / total
    deviations = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(deviations, deviations)
    return pd.DataFrame(correlation, index=list(columns), columns=list(columns))


def collinear_pairs(correlation: pd.DataFrame, *, threshold: float = 0.8) -> pd.DataFrame:
    """Pairs correlated past the threshold, strongest first.

    Reported rather than resolved. Which of two collinear covariates to keep is a
    judgement about what the model is *for*, and `nmds` makes it explicitly with a
    stated priority order rather than letting a procedure pick.
    """
    pairs = []
    names = [str(name) for name in correlation.columns]
    values = correlation.to_numpy(dtype=float)
    for i, first in enumerate(names):
        for j, second in enumerate(names[i + 1 :], start=i + 1):
            value = float(values[i, j])
            if abs(value) >= threshold:
                pairs.append({"first": first, "second": second, "correlation": value})
    table = pd.DataFrame(pairs, columns=["first", "second", "correlation"])
    if table.empty:
        return table
    return table.reindex(table["correlation"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def survival_by_stratum(
    frame: pd.DataFrame,
    column: str,
    *,
    weight: str = WEIGHT,
    event: str = "event",
    age: str = "age",
) -> pd.DataFrame:
    """Survival curve per level, chained from the exposure-weighted hazards.

    Kaplan-Meier over individual loans is not available here — the panel is
    aggregated — but the construction is the same one: an empirical hazard per age
    band, chained into a survival curve. On weighted cells that is exactly what the
    estimator reduces to.

    Its purpose is to answer one question before any model is fitted: do the strata's
    curves **cross**? Curves that separate and stay separated are what a single model
    with covariates is for. Curves that cross cannot be reconciled by scaling one into
    another, and would need looking at before going further.
    """
    exposure = frame[weight].to_numpy(dtype=float)
    working = pd.DataFrame(
        {
            "stratum": frame[column].to_numpy(),
            "age": frame[age].to_numpy(),
            "exposure": exposure,
            "events": exposure * frame[event].to_numpy(dtype=bool),
        }
    )
    grouped = (
        working.groupby(["stratum", "age"], observed=True)[["exposure", "events"]]
        .sum()
        .reset_index()
    )
    grouped["hazard"] = grouped["events"] / grouped["exposure"]
    grouped = grouped.sort_values(["stratum", "age"])
    grouped["survival"] = (
        (1.0 - grouped["hazard"]).groupby(grouped["stratum"], observed=True).cumprod()
    )
    return grouped.reset_index(drop=True)


#: Loan-months a stratum must still have at an age for that age to count in the
#: crossing test. A thirty-year book has ages nobody reached: at the far tail a
#: stratum's exposure falls to single digits, where a survival estimate is noise and
#: two curves cross and re-cross freely.
#:
#: Without the floor the test returned "they cross" for **every** stratum of this book,
#: on tails of 2, 4 and 109 loan-months -- a check that always fires answers nothing,
#: and this one exists to be able to refuse the single-survival-function commitment.
MIN_EXPOSURE_FOR_CROSSING: Final = 10_000


#: How far apart two survival curves must be for their order to mean anything, in
#: survival units. A tenth of a percentage point.
#:
#: The default was 1e-9, which treats any difference at all as an ordering. On this book
#: that reported a crossing for every stratum, and every one of them sat at **two or
#: three months of loan age with a gap of 0.00000 to 0.0002** -- because a default needs
#: ninety days of delinquency, so before month three every curve is at 1.0 by
#: construction and they differ only by rounding.
CROSSING_TOLERANCE: Final = 0.001


def curves_cross(
    curves: pd.DataFrame,
    *,
    tolerance: float = CROSSING_TOLERANCE,
    min_exposure: float = MIN_EXPOSURE_FOR_CROSSING,
) -> bool:
    """Whether any two strata's survival curves change their order.

    The check behind :func:`survival_by_stratum`. If the ranking of two strata flips at
    any age, no single scale factor maps one onto the other -- and the project's
    commitment to a single survival function would have to be revisited.

    Two floors, for two different failures, and the check was useless without either:

    * **Exposure.** A thirty-year book has ages nobody reached. At the far tail a
      stratum falls to single-digit loan-months, where a survival estimate is noise.
    * **Materiality.** Before month three no loan can default at all -- ninety days of
      delinquency does not fit -- so every curve is at 1.0 there and they differ by
      rounding. A tolerance of 1e-9 called that a crossing.

    Between them they leave the check able to fire on a difference that would actually
    matter, which is the point: this is the test that can refuse the project's
    commitment to a single survival function, and a test that always fires refuses
    nothing.
    """
    if "exposure" in curves.columns:
        thin = curves.loc[curves["exposure"] < min_exposure, "age"].unique()
        curves = curves[~curves["age"].isin(thin)]

    wide = curves.pivot(index="age", columns="stratum", values="survival").dropna()
    if wide.shape[1] < 2 or wide.empty:
        return False

    names = list(wide.columns)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            difference = wide[first].to_numpy() - wide[second].to_numpy()
            meaningful = difference[np.abs(difference) > tolerance]
            if len(meaningful) and not ((meaningful > 0).all() or (meaningful < 0).all()):
                return True
    return False
