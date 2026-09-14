"""The variable selection, run end to end on the training half of the whole population.

The validation's F1. The procedure in ``docs/variable_selection.md`` was carried out a step
at a time and its outcome copied into ``config.TIME_VARYING_CONTINUOUS`` and
``config.ELIMINATED``: every function it used is tested, and nothing ran them in sequence,
so the specification could be believed but not regenerated. :func:`run_selection` is that
sequence, ``creditsurv select`` runs it, and a test fails when the configuration and the
last record it wrote part.

5. **Weighted correlation** between the continuous candidates; pairs above 0.8 reported.
6. **Variance inflation**, dropping the worst above 10 in the priority fixed in config.
7. **Univariate screening**: each candidate fitted beside the loan block.
8. **Backward elimination**: a backwards sign goes first, then a p-value above 0.05 --
   although at this sample size no p-value is above anything.
9. **Stability**: the survivors fitted on two halves of the book. A covariate whose sign
   differs between them, beside a larger covariate of the same economic dimension, is not
   identified and goes: the rule the first run arrived at, now applied instead of argued.

Steps 1-4 describe one covariate at a time and belong to ``creditsurv profile``.

Two departures from the first run, both forced by keeping the backtest honest:

* the **training half**, not the whole population, which includes the months the model
  is judged on -- a specification chosen on them has already seen the test;
* stability across **loans originated in even and odd years**, rather than the whole
  population against its first 94%, a comparison that needs the test window. Each half
  covers every calendar month, so a sign that moves between them moves with the sample
  and not with the economy.

Every fit is saved the moment it lands, under the cell table, the sample and the formula,
so a run that stops resumes where it stopped. Each starts from the coefficients of the
nearest model already fitted, which changes how long it takes and not where it ends.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from creditsurv.config import (
    CATEGORICAL_REFERENCE,
    ECONOMIC_DIMENSION,
    MACRO_CANDIDATES,
    MACRO_ELIMINATION_PRIORITY,
    ORDINAL,
    STATIC_CONTINUOUS,
)
from creditsurv.data.panel import WEIGHT
from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
from creditsurv.explore import collinear_pairs
from creditsurv.models.aft import FitResult, fit_aft
from creditsurv.models.blocks import DEFAULT_BLOCK_ROWS
from creditsurv.models.selection import (
    EXPECTED_SIGNS,
    PVALUE_THRESHOLD,
    VIF_THRESHOLD,
    stepwise_vif,
    weighted_covariance,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

#: Loan characteristics in the cell key that no earlier specification screened, with their
#: reference levels. Admitted to the key for the validation's M3; step 7 is where they meet
#: the same test as every other candidate, whatever it says.
CANDIDATE_CATEGORICAL: Final[dict[str, str]] = {"has_mi": "N", "first_time_buyer": "N"}

#: Correlation above which a pair is reported at step 5.
CORRELATION_THRESHOLD: Final = 0.8


@dataclass(frozen=True)
class Specification:
    """The covariates a model reads, and the formula that says so."""

    continuous: tuple[str, ...]
    categorical: tuple[tuple[str, str], ...] = ()

    @property
    def covariates(self) -> list[str]:
        return [*self.continuous, *(name for name, _ in self.categorical)]

    @property
    def formula(self) -> str:
        terms = [*self.continuous]
        terms += [f"C({name}, Treatment('{level}'))" for name, level in self.categorical]
        return " + ".join(terms)

    def plus(self, name: str, *, reference: str | None = None) -> Specification:
        if reference is not None:
            return replace(self, categorical=(*self.categorical, (name, reference)))
        return replace(self, continuous=(*self.continuous, name))

    def minus(self, name: str) -> Specification:
        return replace(
            self,
            continuous=tuple(term for term in self.continuous if term != name),
            categorical=tuple(pair for pair in self.categorical if pair[0] != name),
        )


@dataclass
class Fits:
    """Every fit the selection makes: saved as it lands, started from the nearest model.

    ``identity`` names the cell table the rows came from -- its file, size and time of
    writing -- so a cached fit is never reused for a table that has since been rebuilt.
    """

    train: pd.DataFrame
    identity: str
    as_of: str
    moratorium: str
    block_rows: int = DEFAULT_BLOCK_ROWS
    record: list[dict[str, object]] = field(default_factory=list)

    def fit(
        self,
        spec: Specification,
        *,
        sample: str = "training half",
        where: np.ndarray | None = None,
        start: FitResult | None = None,
    ) -> FitResult:
        described: dict[str, object] = {
            "purpose": "selection",
            "cells": self.identity,
            "as_of": self.as_of,
            "moratorium": self.moratorium,
            "sample": sample,
            "formula": spec.formula,
            "distribution": "weibull",
            "likelihood": "interval_censored",
        }
        fingerprint = fit_fingerprint(**described)
        cached = load_fit(fingerprint)
        if isinstance(cached, FitResult):
            log.info("cached: %s on the %s", spec.formula, sample)
            self.record.append({**described, "minutes": 0.0, "evaluations": None, "cached": True})
            return cached

        log.info("fitting: %s on the %s", spec.formula, sample)
        started = time.perf_counter()
        result = fit_aft(
            self.train,
            spec.covariates,
            spec.formula,
            weights_col=WEIGHT,
            block_rows=self.block_rows,
            initial_point=None if start is None else start.fitter.params_,
            where=where,
        )
        minutes = (time.perf_counter() - started) / 60
        save_fit(result, fingerprint, {**described, "minutes": minutes})
        evaluations = None if result.blocks is None else result.blocks.evaluations
        self.record.append(
            {**described, "minutes": minutes, "evaluations": evaluations, "cached": False}
        )
        return result


@dataclass
class SelectionRecord:
    """Everything a selection run found, in the order it found it."""

    as_of: str
    moratorium: str
    rows: int
    loan_months: float
    correlation: pd.DataFrame
    collinear: pd.DataFrame
    inflation: pd.DataFrame
    screening: pd.DataFrame
    elimination: pd.DataFrame
    stability: pd.DataFrame
    selected: Specification
    eliminated: dict[str, str]
    fits: list[dict[str, object]]

    def summary(self) -> dict[str, object]:
        """What the configuration should say, as data a test can compare it with."""
        continuous = self.selected.continuous
        return {
            "as_of": self.as_of,
            "moratorium": self.moratorium,
            "static_continuous": [name for name in continuous if name in STATIC_CONTINUOUS],
            "ordinal": [name for name in continuous if name in ORDINAL],
            "time_varying_continuous": [name for name in continuous if name in MACRO_CANDIDATES],
            "categorical": dict(self.selected.categorical),
            "eliminated": self.eliminated,
            "formula": self.selected.formula,
        }


def run_selection(
    train: pd.DataFrame,
    fits: Fits,
    *,
    static: Sequence[str] = STATIC_CONTINUOUS,
    ordinal: Sequence[str] = ORDINAL,
    macro: Sequence[str] = MACRO_CANDIDATES,
    base_categorical: Mapping[str, str] = CATEGORICAL_REFERENCE,
    candidate_categorical: Mapping[str, str] = CANDIDATE_CATEGORICAL,
    halves: np.ndarray | None = None,
) -> SelectionRecord:
    """Run steps 5 to 9 and return what each found. ``halves`` marks one half for step 9.

    The candidate lists default to the configuration and are parameters so a test can run
    the whole sequence on a handful of covariates.
    """
    loan = [*static, *ordinal]
    continuous = [*loan, *macro]
    eliminated: dict[str, str] = {}

    # 5. Pairs that say the same thing -- reported, not resolved.
    log.info("step 5: weighted correlation of %d candidates", len(continuous))
    covariance = weighted_covariance(train, continuous, weight=WEIGHT)
    deviations = pd.Series(np.sqrt(np.diag(covariance.to_numpy())), index=covariance.index)
    correlation = pd.DataFrame(
        covariance.to_numpy() / np.outer(deviations, deviations),
        index=covariance.index,
        columns=covariance.columns,
    )
    collinear = collinear_pairs(correlation, threshold=CORRELATION_THRESHOLD)

    # 6. Collinearity beyond pairs. The loan block is protected above every macro series,
    # and among the macro series the order is the one config fixed before any result.
    log.info("step 6: variance inflation")
    priority = [
        *(name for name in MACRO_ELIMINATION_PRIORITY if name in macro),
        *(name for name in macro if name not in MACRO_ELIMINATION_PRIORITY),
        *ordinal,
        *static,
    ]
    inflation, surviving = stepwise_vif(
        train, continuous, weight=WEIGHT, threshold=VIF_THRESHOLD, priority=priority
    )
    for removed, vif in zip(
        inflation["removed"].astype(str), inflation["vif"].to_numpy(dtype=float), strict=True
    ):
        eliminated[removed] = f"step 6: variance inflation {vif:.1f}, above {VIF_THRESHOLD:g}"

    base = Specification(
        continuous=tuple(name for name in loan if name in surviving),
        categorical=tuple(base_categorical.items()),
    )

    # 7. Each candidate beside the loan block, started from the loan block's fit.
    log.info("step 7: univariate screening")
    base_fit = fits.fit(base)
    screened: list[dict[str, object]] = []
    kept: list[tuple[str, str | None]] = []
    candidates: list[tuple[str, str | None]] = [
        *((name, None) for name in macro if name in surviving),
        *candidate_categorical.items(),
    ]
    for name, reference in candidates:
        result = fits.fit(base.plus(name, reference=reference), start=base_fit)
        rows = _screen(name, reference, result, base_fit, deviations)
        screened.extend(rows)
        worst_p = max(float(str(row["p"])) for row in rows)
        if worst_p > PVALUE_THRESHOLD:
            eliminated[name] = f"step 7: p = {worst_p:.3g} beside the loan block"
        else:
            kept.append((name, reference))
    screening = pd.DataFrame(screened, columns=_SCREENING_COLUMNS)

    # 8. Backward elimination, one covariate a step.
    log.info("step 8: backward elimination")
    current = base
    for name, reference in kept:
        current = current.plus(name, reference=reference)
    previous = base_fit
    steps: list[dict[str, object]] = []
    while True:
        result = fits.fit(current, start=previous)
        worst = _worst(current, result)
        if worst is None:
            break
        name, reason, coefficient, p_value = worst
        steps.append(
            {
                "step": len(steps) + 1,
                "removed": name,
                "coef": coefficient,
                "p": p_value,
                "reason": reason,
                "remaining": len(current.covariates) - 1,
            }
        )
        eliminated[name] = f"step 8: {reason}"
        current = current.minus(name)
        previous = result
    elimination = pd.DataFrame(steps, columns=_ELIMINATION_COLUMNS)

    # 9. Stability, on two halves of the book.
    rounds: list[pd.DataFrame] = []
    if halves is not None:
        log.info("step 9: stability on two halves")
        while True:
            whole = fits.fit(current, start=previous)
            even = fits.fit(current, sample="even origination years", where=halves, start=whole)
            odd = fits.fit(current, sample="odd origination years", where=~halves, start=whole)
            table = _stability(current, whole, even, odd, deviations).assign(round=len(rounds) + 1)
            rounds.append(table)
            verdict = _not_identified(table)
            if verdict is None:
                break
            name, partner = verdict
            indexed = table.set_index("covariate")
            eliminated[name] = (
                f"step 9: 1 sd effect {_number(indexed, name, 'effect_even'):+.3f} on even "
                f"and {_number(indexed, name, 'effect_odd'):+.3f} on odd origination years, "
                f"beside {partner} ({_number(indexed, partner, 'effect_all'):+.3f}), both "
                f"{indexed.loc[name, 'dimension']}"
            )
            current = current.minus(name)
            previous = whole
    stability = pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame()

    return SelectionRecord(
        as_of=fits.as_of,
        moratorium=fits.moratorium,
        rows=len(train),
        loan_months=float(train[WEIGHT].sum()),
        correlation=correlation,
        collinear=collinear,
        inflation=inflation,
        screening=screening,
        elimination=elimination,
        stability=stability,
        selected=current,
        eliminated=eliminated,
        fits=fits.record,
    )


_SCREENING_COLUMNS: Final = [
    "covariate",
    "term",
    "coef",
    "se",
    "z",
    "p",
    "effect_1sd",
    "expected_sign",
    "sign_agrees",
    "lr_statistic",
]

_ELIMINATION_COLUMNS: Final = ["step", "removed", "coef", "p", "reason", "remaining"]


def _number(frame: pd.DataFrame, row: str, column: str) -> float:
    """One cell of a table as a float; the stubs type ``.loc`` as any scalar at all."""
    return float(cast("float", frame.loc[row, column]))


def _terms(result: FitResult) -> pd.DataFrame:
    """The coefficient table of the scale parameter, where every covariate enters."""
    summary: pd.DataFrame = result.fitter.summary.loc[result.fitter._primary_parameter_name]
    return summary


def _screen(
    name: str,
    reference: str | None,
    result: FitResult,
    base: FitResult,
    deviations: pd.Series,
) -> list[dict[str, object]]:
    """One row per coefficient a candidate adds, and the likelihood ratio it earns."""
    summary = _terms(result)
    keys = (
        [name]
        if reference is None
        else [str(key) for key in summary.index if str(key).startswith(f"C({name},")]
    )
    statistic = 2.0 * (result.log_likelihood - base.log_likelihood)
    expected = EXPECTED_SIGNS.get(name, 0)
    rows = []
    for key in keys:
        coefficient = _number(summary, key, "coef")
        error = _number(summary, key, "se(coef)")
        effect = coefficient * float(deviations[name]) if name in deviations.index else np.nan
        rows.append(
            {
                "covariate": name,
                "term": key,
                "coef": coefficient,
                "se": error,
                "z": coefficient / error,
                "p": _number(summary, key, "p"),
                "effect_1sd": effect,
                "expected_sign": expected,
                "sign_agrees": expected == 0 or coefficient * expected > 0,
                "lr_statistic": statistic,
            }
        )
    return rows


def _worst(spec: Specification, result: FitResult) -> tuple[str, str, float, float] | None:
    """The covariate step 8 removes next, and why -- or ``None`` when every one stays.

    A backwards sign outranks any p-value, because it says the specification is wrong
    rather than that the evidence is thin. Of several, the weakest -- smallest ``|z|`` --
    goes first, and only one per step: removing it moves every other coefficient.
    Categorical terms carry no expected sign, and at this sample size no p-value.
    """
    summary = _terms(result)
    backwards: list[tuple[float, str, float, float]] = []
    thin: list[tuple[float, str, float, float]] = []
    for name in spec.continuous:
        coefficient = _number(summary, name, "coef")
        z = coefficient / _number(summary, name, "se(coef)")
        p_value = _number(summary, name, "p")
        expected = EXPECTED_SIGNS.get(name)
        if expected is not None and coefficient * expected < 0:
            backwards.append((abs(z), name, coefficient, p_value))
        elif p_value > PVALUE_THRESHOLD:
            thin.append((p_value, name, coefficient, p_value))
    if backwards:
        _, name, coefficient, p_value = min(backwards)
        direction = "+" if EXPECTED_SIGNS[name] > 0 else "-"
        return (
            name,
            f"wrong sign: {coefficient:+.4g} where {direction} is expected",
            coefficient,
            p_value,
        )
    if thin:
        _, name, coefficient, p_value = max(thin)
        return name, f"p = {p_value:.3g}", coefficient, p_value
    return None


def _effect(result: FitResult, name: str, deviations: pd.Series) -> float:
    """Log survival time per standard deviation of the covariate."""
    return _number(_terms(result), name, "coef") * float(deviations[name])


def _stability(
    spec: Specification,
    whole: FitResult,
    even: FitResult,
    odd: FitResult,
    deviations: pd.Series,
) -> pd.DataFrame:
    """Each continuous covariate's standardised effect on the whole and on each half."""
    rows = []
    for name in spec.continuous:
        effects = [_effect(result, name, deviations) for result in (whole, even, odd)]
        rows.append(
            {
                "covariate": name,
                "dimension": ECONOMIC_DIMENSION.get(name, name),
                "effect_all": effects[0],
                "effect_even": effects[1],
                "effect_odd": effects[2],
                "stable": len({bool(effect > 0) for effect in effects}) == 1,
            }
        )
    return pd.DataFrame(rows)


def _not_identified(table: pd.DataFrame) -> tuple[str, str] | None:
    """The smallest unstable covariate that sits beside a larger one of its dimension.

    Returns the covariate and the partner that carries its information, or ``None``. An
    unstable covariate with no such partner is kept and left visible in the table: the
    rule is about information entering twice, and there is no second entry to prefer.
    """
    ranked = table.assign(size=table["effect_all"].abs()).sort_values("size")
    for row in ranked.itertuples(index=False):
        if row.stable:
            continue
        partners = ranked[
            (ranked["dimension"] == row.dimension)
            & (ranked["covariate"] != row.covariate)
            & (ranked["size"] > row.size)
        ]
        if not partners.empty:
            return str(row.covariate), str(partners.iloc[-1]["covariate"])
    return None
