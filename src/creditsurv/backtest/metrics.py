"""Backtest metrics: discrimination, calibration and stability, on weighted cells.

Discrimination and calibration answer different questions, and a model can be
excellent at one while useless at the other. A model that ranks every loan
correctly but predicts three times the observed default rate discriminates
perfectly and would still misprice the book. Both are reported, always.

Everything is **exposure-weighted**, because the unit here is a cell rather than a
loan. A cell stands for a number of loan-months, so a statistic taken over rows would
describe the binning instead of the book -- weighting a cell holding six loan-months
the same as one holding sixty thousand.

That has one consequence worth naming. A concordance index is not available: it needs
pairs of *subjects*, and a cell is not a subject. The exposure-weighted Lorenz curve
asks the equivalent question of the data that does exist, and is reported in its place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from scipy.stats import beta

if TYPE_CHECKING:
    from collections.abc import Sequence


def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    n_bins: int = 10,
    reference_weights: pd.Series | None = None,
    comparison_weights: pd.Series | None = None,
) -> float:
    """How far a variable's distribution has moved between two samples.

    Handles categorical and numeric variables differently, because they are
    different problems. A categorical splits on its own levels -- that is the
    original use of the index, comparing the mix of business written. A numeric
    variable has no natural levels, so it is binned on quantiles of the reference
    sample, which keeps each bin populated where the data actually is rather than
    where an equal-width grid would put it.

    Standard credit monitoring, and the piece that explains *why* performance
    degrades rather than only reporting that it did. Convention: below 0.1 is
    stable, 0.1 to 0.25 warrants attention, above 0.25 is a material shift.

    The weights are how this survives aggregation. Shares taken over rows of a cell
    table describe the mix of *covariate combinations*, which nobody has a view on;
    weighting by the loan-months each cell stands for gives back the mix of
    business, which is the quantity the thresholds above were calibrated on.
    """
    reference_weight = _weights(reference, reference_weights)
    comparison_weight = _weights(comparison, comparison_weights)

    if isinstance(reference.dtype, pd.CategoricalDtype) or reference.dtype == object:
        levels = sorted(set(reference.dropna().unique()) | set(comparison.dropna().unique()))
        reference_share = np.array(
            [_share(reference == level, reference_weight) for level in levels]
        )
        comparison_share = np.array(
            [_share(comparison == level, comparison_weight) for level in levels]
        )
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        reference_share = (
            np.histogram(reference.to_numpy(dtype=float), bins=edges, weights=reference_weight)[0]
            / reference_weight.sum()
        )
        comparison_share = (
            np.histogram(comparison.to_numpy(dtype=float), bins=edges, weights=comparison_weight)[0]
            / comparison_weight.sum()
        )

    # A zero share makes the logarithm infinite, so empty bins are floored.
    floor = 1e-6
    reference_share = np.clip(reference_share, floor, None)
    comparison_share = np.clip(comparison_share, floor, None)

    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def _weights(values: pd.Series, weights: pd.Series | None) -> np.ndarray:
    """Weights as a plain array, defaulting to one per row."""
    if weights is None:
        return np.ones(len(values), dtype=float)
    return weights.to_numpy(dtype=float)


def _share(mask: pd.Series, weights: np.ndarray) -> float:
    """Weighted share of the rows a mask selects."""
    total = weights.sum()
    return float(weights[mask.to_numpy(dtype=bool)].sum() / total) if total > 0 else 0.0


def stability_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    covariates: Sequence[str],
    *,
    time_varying: Sequence[str] = (),
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Population stability index per covariate, worst first.

    The usual reading -- below 0.1 stable, 0.1 to 0.25 monitor, above 0.25 shifted
    -- was devised for **application characteristics**: credit score, loan size,
    the mix of business being written. Those should look much the same this quarter
    as last, and a jump means something has changed about who is applying.

    It does not transfer to macro-driven covariates. ``ltv_change`` and
    ``unemployment_change`` are *designed* to move with the economy, and train and test come
    from different calendar periods by construction, so their index is large
    whenever anything happened. This backtest returns 9.23 for ``ltv_change``
    against 0.03 for ``credit_score``: the first number says the economy changed, which
    is not news and is not a defect, while the second says the book being written
    did not, which is the thing PSI was built to detect.

    Those covariates are therefore labelled rather than dropped, so the alarming
    figure is visible and correctly discounted instead of quietly triggering a
    model review.
    """
    varying = set(time_varying)
    train_weights = None if weights_col is None else train[weights_col]
    test_weights = None if weights_col is None else test[weights_col]
    rows = [
        {
            "covariate": name,
            "psi": population_stability_index(
                train[name],
                test[name],
                reference_weights=train_weights,
                comparison_weights=test_weights,
            ),
            "kind": "time-varying" if name in varying else "static",
        }
        for name in covariates
        if name in train.columns and name in test.columns
    ]
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    table["interpretation"] = np.where(
        table["kind"] == "time-varying",
        "expected to move",
        pd.cut(
            table["psi"],
            bins=[-np.inf, 0.1, 0.25, np.inf],
            labels=["stable", "monitor", "shifted"],
        ).astype(str),
    )
    return table


# --------------------------------------------------------------------------------------
# Metrics for an aggregated panel
# --------------------------------------------------------------------------------------


def weighted_calibration(
    predicted: pd.Series,
    observed: pd.Series,
    exposure: pd.Series,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Predicted against realised default rate, by bucket of predicted risk.

    Quantities are over exposure rather than shares of loans, and buckets are weighted by
    exposure so a bucket is a comparable slice of the book rather than of the cell
    table.

    The columns are named as that function's are, deliberately: the two tables say
    the same thing about different units, and everything that reads one -- the chart,
    the report -- should not have to know which it was handed. They diverged once and
    the report simply stopped rendering.
    """
    frame = pd.DataFrame(
        {
            "predicted": predicted.to_numpy(dtype=float),
            "events": observed.to_numpy(dtype=float),
            "exposure": exposure.to_numpy(dtype=float),
        }
    )
    frame["bucket"] = exposure_buckets(
        frame["predicted"].to_numpy(), frame["exposure"].to_numpy(), n_buckets=n_buckets
    )
    # The expected rate is the expected defaults over the loan-months, as the actual rate is.
    # A plain mean of the cells' hazards gave a cell of three loan-months the say of one of
    # thirty thousand; on the test window it put the riskiest decile at 0.929 where it is
    # 1.064, and every decile below one.
    frame["expected_defaults"] = frame["predicted"] * frame["exposure"]

    grouped = frame.groupby("bucket", observed=True).agg(
        loan_months=("exposure", "sum"),
        events=("events", "sum"),
        expected_defaults=("expected_defaults", "sum"),
    )
    grouped["expected"] = grouped.pop("expected_defaults") / grouped["loan_months"]
    grouped["actual"] = grouped["events"] / grouped["loan_months"]
    grouped["difference"] = grouped["actual"] - grouped["expected"]
    grouped["ratio"] = np.where(
        grouped["expected"] > 0, grouped["actual"] / grouped["expected"], np.nan
    )
    return grouped.reset_index()


def exposure_buckets(
    predicted: np.ndarray, exposure: np.ndarray, *, n_buckets: int = 10
) -> np.ndarray:
    """The bucket of predicted risk each row falls in, buckets holding equal exposure.

    Returned in the rows' own order, so a caller can group anything else by it. One
    definition for the backtest's decile table and for every view that opens a
    calibration by decile: two versions of the same bucketing would, sooner or later,
    put the same loan-month in different deciles.
    """
    order = np.argsort(predicted, kind="stable")
    cumulative = np.cumsum(exposure[order])
    total = cumulative[-1] if len(cumulative) else 0.0
    sorted_buckets = np.minimum(
        (cumulative / total * n_buckets).astype(np.int64) if total > 0 else 0, n_buckets - 1
    )
    buckets = np.empty(len(predicted), dtype=np.int64)
    buckets[order] = sorted_buckets
    return buckets


def weighted_gini(predicted: pd.Series, observed: pd.Series, exposure: pd.Series) -> float:
    """Discrimination on an aggregated panel, from the Lorenz curve.

    An aggregated panel has no loans to rank, so a concordance index is not
    available: it needs pairs of subjects, and a cell is not a subject. The
    exposure-weighted Lorenz curve asks the equivalent question of the data that does
    exist -- order the cells by predicted risk, and see how much of the realised
    default falls in the riskiest slice of exposure.

    Returns twice the area between that curve and the diagonal, so it lands on the
    same 0-to-1 scale as a Gini from a concordance index, and means the same thing:
    zero is no ordering, one is perfect.
    """
    frame = pd.DataFrame(
        {
            "predicted": predicted.to_numpy(dtype=float),
            "events": observed.to_numpy(dtype=float),
            "exposure": exposure.to_numpy(dtype=float),
        }
    ).sort_values("predicted", ascending=False)

    total_events = float(frame["events"].sum())
    total_exposure = float(frame["exposure"].sum())
    if total_events <= 0 or total_exposure <= 0:
        return float("nan")

    share_exposure = np.concatenate([[0.0], (frame["exposure"].cumsum() / total_exposure)])
    share_events = np.concatenate([[0.0], (frame["events"].cumsum() / total_events)])

    area = float(np.trapezoid(share_events, share_exposure))
    return 2.0 * area - 1.0


def actual_versus_expected(
    predicted: pd.Series,
    observed: pd.Series,
    exposure: pd.Series,
    by: pd.Series,
) -> pd.DataFrame:
    """Realised against predicted default, grouped by anything.

    Usually by vintage or by reporting period. The single number a credit committee
    reads is the ratio: above one the model under-predicts, below one it
    over-predicts.
    """
    # Totals by bincount on the group codes. A group-by first builds a frame of four
    # columns of doubles -- 2 GB on the training half of the exact key -- to fill a table
    # of a few dozen rows.
    codes, groups = pd.factorize(by, sort=True)
    present = codes >= 0
    codes = codes[present]
    weights = exposure.to_numpy(dtype=float)[present]

    def total(values: np.ndarray) -> np.ndarray:
        summed: np.ndarray = np.bincount(codes, weights=values, minlength=len(groups))
        return summed

    grouped = pd.DataFrame(
        {
            "group": groups,
            "expected": total(predicted.to_numpy(dtype=float)[present] * weights),
            "events": total(observed.to_numpy(dtype=float)[present]),
            "exposure": total(weights),
        }
    )
    grouped["expected_rate"] = grouped["expected"] / grouped["exposure"]
    grouped["actual_rate"] = grouped["events"] / grouped["exposure"]
    grouped["actual_over_expected"] = np.where(
        grouped["expected"] > 0, grouped["events"] / grouped["expected"], np.nan
    )
    return grouped


#: A month of the test window holding less than this share of the median month's exposure
#: is one the data release barely covers.
THIN_MONTH_SHARE: Final = 0.10


def covered_months(
    over_time: pd.DataFrame, *, share: float = THIN_MONTH_SHARE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The months of the test window the data covers, and the ones it barely does.

    The last months of a release are reported for a sliver of the book: in the published
    backtest March 2026 carried 51,341 loan-months and April 8, against about twelve million
    in a full month. Their realised rates are noise -- April's was zero -- and drawn beside the
    rest they read as the model diverging. A month below ``share`` of the median month's
    exposure is set aside and named, never dropped silently; the totals keep it, since at
    that weight it moves nothing.
    """
    floor = share * float(over_time["exposure"].median())
    thin = over_time["exposure"] < floor
    return over_time.loc[~thin].reset_index(drop=True), over_time.loc[thin].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The master scale: twelve-month PD by grade
# --------------------------------------------------------------------------------------

#: Floor of the second grade. Each grade's floor is twice the previous one's, so eight
#: grades reach from under 5 basis points to over 3.2%, which is the span a prime agency
#: book actually occupies: the published model's twelve-month PD runs from a handful of
#: basis points on the best vintages to a few percent on 2007.
#:
#: Declared in `docs/rules.md` before the run that scores it, with the count of grades and
#: the rule for passing. A scale drawn after seeing the distribution is a scale drawn to
#: pass.
GRADE_FLOOR: Final = 0.0005

#: Grades on the master scale.
GRADES: Final = 8

#: How many of them must pass for the scale to pass.
GRADES_TO_PASS: Final = 7


def master_scale(*, floor: float = GRADE_FLOOR, grades: int = GRADES) -> np.ndarray:
    """The lower edge of every grade: geometric, each twice the one before.

    The first grade has no floor and the last no ceiling, so the scale covers the line and
    a loan can always be graded -- the returned array is the ``grades - 1`` internal
    boundaries.
    """
    return floor * 2.0 ** np.arange(grades - 1, dtype=float)


def grade_of(twelve_month_pd: np.ndarray, *, scale: np.ndarray | None = None) -> np.ndarray:
    """Which grade each predicted twelve-month PD falls in, 1 the safest.

    Bands closed on the right, as every band in this project is: a PD exactly on an edge
    belongs to the grade below it.
    """
    edges = master_scale() if scale is None else scale
    return np.searchsorted(edges, np.asarray(twelve_month_pd, dtype=float), side="left") + 1


def annualised(hazard: np.ndarray | pd.Series) -> np.ndarray:
    """A monthly hazard as the probability of defaulting within twelve months.

    ``1 - (1 - h)^12``: the loan's risk over the coming year at today's hazard, which is
    what the realised measure it is compared against also is -- defaults per obligor-year.
    Projecting each loan-month's covariate path forward twelve months would be the fuller
    answer and is out of reach at sixty million rows; the approximation is stated rather
    than hidden, and it is the same on both sides of the comparison.
    """
    monthly = np.asarray(hazard, dtype=float)
    return 1.0 - (1.0 - monthly) ** 12


def jeffreys_interval(
    defaults: np.ndarray | float, exposure: np.ndarray | float, *, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """The Jeffreys interval for a default rate: Beta(d + 1/2, n - d + 1/2).

    Jeffreys rather than a normal approximation because the top grades hold few defaults
    and sometimes none, where a normal interval is either meaningless or of zero width --
    and a criterion that no grade can fail is not a criterion. It is the interval the
    validation asked for by name.
    """
    d = np.asarray(defaults, dtype=float)
    n = np.asarray(exposure, dtype=float)
    tail = (1.0 - confidence) / 2.0
    lower = beta.ppf(tail, d + 0.5, np.maximum(n - d, 0.0) + 0.5)
    upper = beta.isf(tail, d + 0.5, np.maximum(n - d, 0.0) + 0.5)
    return np.nan_to_num(lower, nan=0.0), np.nan_to_num(upper, nan=1.0)


def grade_backtest(
    predicted_hazard: pd.Series,
    observed: pd.Series,
    exposure: pd.Series,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Twelve-month PD by grade, against what the grade's loans actually did.

    One row per populated grade: its exposure in obligor-years, the predicted PD it was
    graded on, the realised rate, the Jeffreys interval around the realised rate, and
    whether the prediction falls inside it.

    **Obligor-years, not obligors at a date.** The cells carry no loan identity, so a
    cohort taken at a reporting date cannot be followed through them; the denominator is
    loan-months over twelve, which is the exposure the defaults were earned on and the
    unit the predicted rate is already in. The two coincide when the hazard is flat over
    the year and differ by the loans that leave, which is the same approximation the
    annualisation makes.
    """
    frame = pd.DataFrame(
        {
            "hazard": predicted_hazard.to_numpy(dtype=float),
            "events": observed.to_numpy(dtype=float),
            "exposure": exposure.to_numpy(dtype=float),
        }
    )
    frame["predicted"] = annualised(frame["hazard"].to_numpy())
    frame["grade"] = grade_of(frame["predicted"].to_numpy())
    frame["predicted_defaults"] = frame["predicted"] * frame["exposure"] / 12.0

    grouped = frame.groupby("grade", observed=True).agg(
        loan_months=("exposure", "sum"),
        defaults=("events", "sum"),
        predicted_defaults=("predicted_defaults", "sum"),
    )
    grouped["obligor_years"] = grouped["loan_months"] / 12.0
    grouped["predicted_pd"] = grouped.pop("predicted_defaults") / grouped["obligor_years"]
    grouped["actual_pd"] = grouped["defaults"] / grouped["obligor_years"]
    lower, upper = jeffreys_interval(
        grouped["defaults"].to_numpy(), grouped["obligor_years"].to_numpy(), confidence=confidence
    )
    grouped["lower"] = lower
    grouped["upper"] = upper
    grouped["passed"] = (grouped["predicted_pd"] >= lower) & (grouped["predicted_pd"] <= upper)
    return grouped.reset_index()


def master_scale_passed(table: pd.DataFrame, *, required: int = GRADES_TO_PASS) -> bool:
    """Whether enough grades hold, by the rule declared before the run.

    "Seven of the eight" counted as **at most one failing grade**, over the grades that
    hold any exposure. A grade nobody is in can neither pass nor fail, and counting it as a
    failure would make the rule turn on how much of the scale a window happens to populate
    rather than on the model.
    """
    return bool(int((~table["passed"]).sum()) <= GRADES - required)
