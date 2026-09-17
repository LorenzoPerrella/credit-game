"""The views that need the fitted model: calibration, backtest, coefficients, projections.

Every table here is computed from one fit, scored once on the training half and once on the
test window, and opened by the segments of :mod:`creditsurv.views.segments`. Each table
carries a ``segment`` column -- ``all`` for the whole book -- and a ``group`` column, the level
within it, so the site can offer any of them as a menu of sub-items without knowing which
segments exist.

In-sample views (``ae_by_*``, ``km_vs_model``) describe the data the model was fitted to;
``backtest_*`` and ``acceptance_by_segment`` describe the months after the reporting date,
which it never saw. The site keeps the two apart, as the backtest report does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.backtest.metrics import exposure_buckets, weighted_gini
from creditsurv.backtest.runner import ACCEPTANCE
from creditsurv.data.panel import EVENT, LOAN_ID, WEIGHT
from creditsurv.models.aft import coefficient_table
from creditsurv.models.lifetime_pd import (
    conditional_pd,
    extend_macro,
    project_panel,
    scenario_lifetime_pd,
    survival_along_path,
)
from creditsurv.views.calibration import WHOLE_BOOK, actual_expected, survival_by_age
from creditsurv.views.segments import age_bands, available, calendar_years
from creditsurv.views.tables import View

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from creditsurv.backtest.splits import Split
    from creditsurv.models.aft import FitResult

#: The ``segment`` value of a row describing the whole book.
ALL: Final = "all"


def deciles(hazard: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Decile of predicted risk, 1 to 10, over the exposure of ``frame``."""
    return exposure_buckets(hazard, frame[WEIGHT].to_numpy(dtype=float)) + 1


def _by_segment(
    frame: pd.DataFrame, compute: Callable[[pd.Series | None], pd.DataFrame]
) -> pd.DataFrame:
    pieces = [compute(None).assign(segment=ALL)]
    for segment in available(frame):
        pieces.append(compute(segment.label(frame)).assign(segment=segment.name))
    stacked = pd.concat(pieces, ignore_index=True)
    leading = ["segment", "group"]
    return stacked[[*leading, *(c for c in stacked.columns if c not in leading)]]


def _actual_expected_by_segment(
    frame: pd.DataFrame,
    hazard: np.ndarray,
    name: str,
    dimension: pd.Series | pd.Categorical | np.ndarray,
) -> pd.DataFrame:
    def compute(groups: pd.Series | None) -> pd.DataFrame:
        if groups is None:
            return actual_expected(frame, hazard, {name: dimension}).assign(group=WHOLE_BOOK)
        return actual_expected(frame, hazard, {"group": groups, name: dimension})

    return _by_segment(frame, compute)


def _vintage_years(frame: pd.DataFrame) -> np.ndarray:
    return pd.PeriodIndex(frame["orig_period"]).year.to_numpy()


def _months(frame: pd.DataFrame) -> pd.Series:
    """The month of observation as a categorical of ``YYYY-MM`` labels.

    Built from the period ordinals: formatting sixty million periods as text is sixty million
    Python strings, where the distinct months number a few hundred.
    """
    index = pd.PeriodIndex(frame["period"])
    ordinals = index.year.to_numpy() * 12 + index.month.to_numpy() - 1
    codes, found = pd.factorize(ordinals, sort=True)
    labels = [f"{int(value) // 12}-{int(value) % 12 + 1:02d}" for value in found]
    months = pd.Categorical.from_codes(codes, categories=pd.Index(labels))
    return pd.Series(months, index=frame.index, name="month")


def acceptance_by_segment(test: pd.DataFrame, hazard: np.ndarray) -> pd.DataFrame:
    """The declared acceptance criteria, applied to every group of every segment.

    Deciles are re-bucketed within each group, so "every decile" means what it means for the
    whole book: ten equal slices of that group's exposure, ranked by its own predicted risk.
    """
    weight = test[WEIGHT].to_numpy(dtype=float)
    events = weight * test[EVENT].to_numpy(dtype=bool)
    labelled: list[tuple[str, pd.Series | None]] = [(ALL, None)]
    labelled += [(segment.name, segment.label(test)) for segment in available(test)]

    rows = []
    for name, labels in labelled:
        levels = [WHOLE_BOOK] if labels is None else [str(v) for v in labels.cat.categories]
        codes = None if labels is None else labels.cat.codes.to_numpy()
        for position, level in enumerate(levels):
            mask = np.ones(len(test), dtype=bool) if codes is None else codes == position
            if not mask.any():
                continue
            exposure, happened, predicted = weight[mask], events[mask], hazard[mask]
            expected = float((predicted * exposure).sum())
            ratio = float(happened.sum()) / expected if expected > 0 else np.nan
            buckets = exposure_buckets(predicted, exposure)
            decile_expected = np.bincount(buckets, weights=predicted * exposure, minlength=10)
            decile_events = np.bincount(buckets, weights=happened, minlength=10)
            with np.errstate(divide="ignore", invalid="ignore"):
                decile_ratios = np.where(
                    decile_expected > 0, decile_events / decile_expected, np.nan
                )
            gini = weighted_gini(pd.Series(predicted), pd.Series(happened), pd.Series(exposure))
            low, high = float(np.nanmin(decile_ratios)), float(np.nanmax(decile_ratios))
            rows.append(
                {
                    "segment": name,
                    "group": level,
                    "loan_months": float(exposure.sum()),
                    "defaults": float(happened.sum()),
                    "expected": expected,
                    "actual_over_expected": ratio,
                    "gini": gini,
                    "decile_low": low,
                    "decile_high": high,
                    "overall_passed": bool(ACCEPTANCE.ae_low <= ratio <= ACCEPTANCE.ae_high),
                    "gini_passed": bool(gini > ACCEPTANCE.gini_min),
                    "deciles_passed": bool(ACCEPTANCE.ae_low <= low and high <= ACCEPTANCE.ae_high),
                }
            )
    return pd.DataFrame(rows)


def calibration_views(
    split: Split,
    *,
    train_hazard: np.ndarray,
    test_hazard: np.ndarray,
    families: Mapping[str, np.ndarray] | None = None,
) -> list[View]:
    """Non-parametric against parametric, in-sample and out of time, by every segment."""
    train, test = split.train, split.test
    in_sample = "On the training half, the data the model was fitted to."
    out_of_time = f"On the months after {split.as_of}, which the model never saw."
    views = [
        View(
            "km_vs_model",
            "Kaplan-Meier against the model, by segment",
            f"Survival by loan age, observed and predicted along each loan's realised "
            f"covariate path, with the Greenwood band. {in_sample}",
            _by_segment(train, lambda groups: survival_by_age(train, train_hazard, groups=groups)),
            source="fit",
        ),
        View(
            "ae_by_year",
            "Actual against expected by calendar year",
            f"Defaults against the model's expectation, year of observation by segment. "
            f"{in_sample}",
            _actual_expected_by_segment(train, train_hazard, "year", calendar_years(train)),
            source="fit",
        ),
        View(
            "ae_by_vintage",
            "Actual against expected by vintage year",
            f"Defaults against expectation by year of origination. {in_sample}",
            _actual_expected_by_segment(train, train_hazard, "vintage_year", _vintage_years(train)),
            source="fit",
        ),
        View(
            "ae_by_age_band",
            "Actual against expected by loan age",
            f"Defaults against expectation by seasoning band. {in_sample}",
            _actual_expected_by_segment(train, train_hazard, "age_band", age_bands(train)),
            source="fit",
        ),
        View(
            "ae_by_decile",
            "Actual against expected by decile of predicted risk",
            f"Deciles of the whole training half's exposure. {in_sample}",
            _actual_expected_by_segment(
                train, train_hazard, "decile", deciles(train_hazard, train)
            ),
            source="fit",
        ),
        View(
            "backtest_by_month",
            "Backtest by month",
            f"Defaults against expectation by month of observation. {out_of_time}",
            _actual_expected_by_segment(test, test_hazard, "month", _months(test)),
            source="fit",
        ),
        View(
            "backtest_by_decile",
            "Backtest by decile of predicted risk",
            f"Deciles of the test window's exposure. {out_of_time}",
            _actual_expected_by_segment(test, test_hazard, "decile", deciles(test_hazard, test)),
            source="fit",
        ),
        View(
            "acceptance_by_segment",
            "Acceptance criteria by segment",
            f"Actual over expected 0.80 to 1.25 overall and in every decile, Gini above 0.45, "
            f"for every group. {out_of_time}",
            acceptance_by_segment(test, test_hazard),
            source="fit",
        ),
    ]
    if families:
        pieces = [
            survival_by_age(train, hazard).assign(distribution=name)
            for name, hazard in families.items()
        ]
        views.append(
            View(
                "families_vs_km",
                "Distribution families against Kaplan-Meier",
                f"Each family's fit of the same specification, chained along the realised "
                f"covariate paths. {in_sample}",
                pd.concat(pieces, ignore_index=True),
                source="fit",
            )
        )
    return views


def coefficient_view(fitted: FitResult, train: pd.DataFrame, continuous: Sequence[str]) -> View:
    """Coefficients with their intervals, and the effect of one standard deviation."""
    table = coefficient_table(fitted).reset_index()
    table.columns = [
        {
            "param": "parameter",
            "covariate": "term",
            "se(coef)": "se",
            "coef lower 95%": "lower",
            "coef upper 95%": "upper",
        }.get(str(column), str(column))
        for column in table.columns
    ]
    weights = train[WEIGHT].to_numpy(dtype=float)
    steps = {}
    for name in continuous:
        if name in train.columns:
            values = train[name].to_numpy(dtype=float)
            mean = float(np.average(values, weights=weights))
            steps[name] = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    table["one_sd"] = table["term"].map(steps)
    table["effect_1sd"] = table["coef"] * table["one_sd"]
    return View(
        "coefficients",
        "Coefficients",
        "On log survival time: positive lengthens survival and lowers risk. The effect of one "
        "standard deviation, exposure-weighted on the training half, puts covariates measured in "
        "different units on one scale.",
        table,
        source="fit",
    )


def covariates_over_time(split: Split, names: Sequence[str]) -> View:
    """The exposure-weighted mean of each time-varying covariate, month by month."""
    pieces = []
    for frame in (split.train, split.test):
        if frame.empty:
            continue
        months = _months(frame)
        codes = months.cat.codes.to_numpy().astype(np.int64)
        found = list(months.cat.categories)
        weight = frame[WEIGHT].to_numpy(dtype=float)
        total = np.bincount(codes, weights=weight, minlength=len(found))
        for name in names:
            if name not in frame.columns:
                continue
            summed = np.bincount(
                codes, weights=weight * frame[name].to_numpy(dtype=float), minlength=len(found)
            )
            pieces.append(pd.DataFrame({"month": found, "covariate": name, "mean": summed / total}))
    table = pd.concat(pieces, ignore_index=True)
    return View(
        "covariates_over_time",
        "Macro covariates over time",
        "The exposure-weighted mean of each time-varying covariate across the loans observed in "
        "each month, training half and test window together.",
        table,
        source="fit",
    )


def projection_views(
    fitted: FitResult,
    book: pd.DataFrame,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    *,
    horizon_months: int = 60,
) -> list[View]:
    """The PD term structure and the scenarios, on the origination book, by segment.

    The book is the commonest origination profiles dated to today, each weighted by how much
    of the lending it stands for (:func:`creditsurv.models.lifetime_pd.origination_book`).
    """
    weights = book.set_index(LOAN_ID)[WEIGHT].astype(float)
    extended = extend_macro(macro, horizon_months + 2)
    panel = project_panel(book, extended, horizon_months=horizon_months)
    survival = survival_along_path(fitted, panel, covariates)
    aligned = weights.reindex(survival.index)

    scenarios = scenario_lifetime_pd(
        fitted, book, macro, covariates, horizon_months=min(horizon_months, 36)
    )
    twelve = conditional_pd(survival, horizon_months=12)

    labelled: list[tuple[str, pd.Series | None]] = [(ALL, None)]
    labelled += [
        (segment.name, segment.label(book).set_axis(book[LOAN_ID])) for segment in available(book)
    ]
    structure, scenario_rows = [], []
    for name, labels in labelled:
        groups = (
            pd.Series(WHOLE_BOOK, index=survival.index)
            if labels is None
            else labels.astype(str).reindex(survival.index)
        )
        for level, members in groups.groupby(groups):
            index = members.index
            share = aligned.loc[index] / aligned.loc[index].sum()
            mean_survival = (survival.loc[index].mul(share, axis=0)).sum(axis=0)
            cumulative = 1.0 - mean_survival
            structure.append(
                pd.DataFrame(
                    {
                        "segment": name,
                        "group": level,
                        "month": mean_survival.index.astype(int),
                        "survival": mean_survival.to_numpy(),
                        "cumulative_pd": cumulative.to_numpy(),
                        "marginal_pd": cumulative.diff().fillna(cumulative.iloc[0]).to_numpy(),
                    }
                )
            )
            scenario_rows.append(
                {
                    "segment": name,
                    "group": level,
                    "book_weight": float(aligned.loc[index].sum()),
                    "pd_12m": float((twelve.loc[index] * share).sum()),
                    **{
                        f"lifetime_pd_{scenario}": float(
                            (scenarios.loc[index, scenario] * share).sum()
                        )
                        for scenario in scenarios.columns
                    },
                }
            )
    scenario_table = pd.DataFrame(scenario_rows)
    if {"lifetime_pd_baseline", "lifetime_pd_adverse"} <= set(scenario_table.columns):
        scenario_table["adverse_multiple"] = (
            scenario_table["lifetime_pd_adverse"] / scenario_table["lifetime_pd_baseline"]
        )
    return [
        View(
            "term_structure_by_segment",
            "PD term structure by segment",
            f"Cumulative and marginal PD over {horizon_months} months for the origination book "
            "dated to today, on the baseline path.",
            pd.concat(structure, ignore_index=True),
            source="fit",
        ),
        View(
            "scenarios_by_segment",
            "Scenarios by segment",
            "Twelve-month PD, and lifetime PD over 36 months under the baseline and the adverse "
            "path, weighted by the lending each profile stands for.",
            scenario_table,
            source="fit",
        ),
    ]
