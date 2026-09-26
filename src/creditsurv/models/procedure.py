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
8. **Backward elimination**: a backwards sign goes first, then a sign reversed between the
   screen and the full model on a covariate with no declared prior, then a p-value above
   0.05 -- although at this sample size almost no p-value is above anything.
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
from lifelines import exceptions

from creditsurv.config import (
    DISTRIBUTION,
    ECONOMIC_DIMENSION,
    MACRO_CANDIDATES,
    MACRO_ELIMINATION_PRIORITY,
)
from creditsurv.data.panel import DEFAULT_CAUSE, WEIGHT, CellBlocks
from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
from creditsurv.explore import collinear_pairs
from creditsurv.models.aft import FitResult, fit_aft, fit_streamed
from creditsurv.models.blocks import DEFAULT_BLOCK_ROWS
from creditsurv.models.selection import (
    EXPECTED_SIGNS,
    PVALUE_THRESHOLD,
    VIF_THRESHOLD,
    Moments,
    stepwise_vif,
    weighted_moments,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

#: Loan characteristics in the cell key that no earlier specification screened, with their
#: reference levels. Admitted to the key for the validation's M3; step 7 is where they meet
#: the same test as every other candidate, whatever it says.
CANDIDATE_CATEGORICAL: Final[dict[str, str]] = {
    "mortgage_insurance": "uninsured",
    "buyer_type": "repeat",
}

#: The payment state is **not** a candidate, and the reason is a measurement rather than a
#: preference. It entered the key in September 2026 and was a candidate until the first
#: selection tried to fit it: SLSQP reached coefficients of 1e+80 and an objective of -4.7e275,
#: and no damped Newton step could lower the objective from the warm start.
#:
#: Why it cannot be fitted is arithmetic, not optimisation. Default is three missed payments,
#: so a loan that opens the month two payments behind is one month from the definition:
#:
#: =============  ===============  ==========  ==================
#: State a month   Loan-months      Defaults    Monthly rate
#: ago
#: =============  ===============  ==========  ==================
#: current         2,740,161,293       22,276   **0.0008%**
#: one month          23,871,219       20,206   0.085%
#: three or more       4,269,824      109,964   2.58%
#: two months          5,337,846    1,518,761   **28.45%**
#: =============  ===============  ==========  ==================
#:
#: Two months behind is 0.2% of the exposure and **91% of every default in the book**, at a
#: rate 35,000 times the current state's. A coefficient for it is a number the likelihood
#: pushes as far as the clipping allows, and the model that came out would answer "will this
#: loan default next month" -- which is a behavioural score, not a lifetime PD. A lifetime PD
#: also has to *project* its covariates over the remaining life, and there is no way to
#: project a payment state: it is the outcome, one month early.
#:
#: It stays in the cell key, where it costs 1.19x and pays for itself in the views: it says
#: where the defaults are, which is worth publishing. The model does not read it.
DELINQUENCY_STATE_NOT_A_CANDIDATE: Final = (
    "step 7: 91% of defaults occur at two months behind, a 28.45% monthly rate against "
    "0.0008% for a current loan. The state is the event one month early, and a lifetime PD "
    "cannot project it forward."
)

#: The loan block the selection starts from and protects from variance inflation, fixed
#: here as ``config.MACRO_CANDIDATES`` fixes the macro block. ``config.STATIC_CONTINUOUS``,
#: ``config.ORDINAL`` and ``config.CATEGORICAL_REFERENCE`` hold what survived. Were the
#: candidates read back from them, a covariate the selection once removed could never be
#: considered again, and ``mortgage_insurance`` and ``buyer_type``, admitted by the first run under
#: this rule, would enter the next one twice.
LOAN_CONTINUOUS: Final[tuple[str, ...]] = ("credit_score", "original_ltv", "debt_to_income")
LOAN_ORDINAL: Final[tuple[str, ...]] = ("term_years",)
#: ``harp`` is in the protected block rather than among the candidates, and that is not a
#: judgement about its coefficient. A HARP refinance reports no debt-to-income, so the fill
#: that lets it into the model at all is absorbed by this level: a selection free to drop it
#: would be free to turn the fill into an imputed ratio for 18% of a decade of vintages. See
#: ``features.NOT_REPORTED``.
BASE_CATEGORICAL: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
    "harp": "standard",
}

#: Correlation above which a pair is reported at step 5.
CORRELATION_THRESHOLD: Final = 0.8

#: Effect of one standard deviation on log survival time below which a macro covariate is
#: removed at step 10. Fixed in `docs/rules.md` before any fit of this branch: 0.02 of log
#: survival time is about a 2% change in expected time to default per standard deviation,
#: the smallest effect this data can tell from a difference in specification. Identification,
#: not significance, is the reason -- at 60 million episodes every p-value is zero.
MATERIALITY_THRESHOLD: Final = 0.02


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

    **Either the rows or a description of them.** ``train`` is an expanded episode frame, as
    the selection used to hold: 91.6 million cells expand to a frame of many gigabytes, and
    a selection makes twenty-odd fits beside it. ``blocks`` is the alternative -- the cell
    file, the window and the covariates -- and each fit then reads its own share of the
    parquet in ``workers`` processes and keeps it compactly. The same specification fitted
    the two ways agrees to 9.4e-07 standard errors, measured; the difference is 68.1 minutes
    at 4.7 GB against 91 at 15.
    """

    train: pd.DataFrame | None
    identity: str
    as_of: str
    moratorium: str
    #: The distribution family every fit of this run uses. An input, not a constant: rule 2
    #: of docs/rules.md takes both families through the whole selection and compares the two
    #: *selected* models, which cannot be done while the family is written into the procedure.
    distribution: str = DISTRIBUTION
    #: Which exit these fits are of. It is in every fingerprint, because the prepayment
    #: model reads the same cells, the same formulas and the same window as the default one.
    cause: str = DEFAULT_CAUSE
    block_rows: int = DEFAULT_BLOCK_ROWS
    #: Where the rows come from when they are not held: the cell file and its window.
    blocks: CellBlocks | None = None
    #: Processes the likelihood is evaluated in, when streaming. Each holds one batch at a
    #: time plus its share of the stored blocks.
    workers: int = 1
    #: The optimiser that last answered, tried first on the next fit. A design that defeats
    #: SLSQP is usually beside another one just like it -- the elimination refits nearly the
    #: same model at every step -- and walking the chain each time cost over an hour a fit on
    #: the prepayment model.
    preferred: str | None = None
    record: list[dict[str, object]] = field(default_factory=list)

    def fit(
        self,
        spec: Specification,
        *,
        sample: str = "training half",
        where: np.ndarray | None = None,
        parity: int | None = None,
        start: FitResult | None = None,
    ) -> FitResult:
        described = selection_description(
            identity=self.identity,
            as_of=self.as_of,
            moratorium=self.moratorium,
            formula=spec.formula,
            sample=sample,
            distribution=self.distribution,
            cause=self.cause,
        )
        fingerprint = fit_fingerprint(**described)
        cached = load_fit(fingerprint)
        # A log-likelihood of zero or more is no likelihood at all: a fit saved before the
        # polish refused such values may have stopped at one.
        if isinstance(cached, FitResult) and cached.log_likelihood < 0:
            log.info("cached: %s on the %s", spec.formula, sample)
            self.record.append({**described, "minutes": 0.0, "evaluations": None, "cached": True})
            return cached

        log.info("fitting: %s on the %s", spec.formula, sample)
        started = time.perf_counter()
        result = self._estimate(spec, where=where, parity=parity, start=start)
        minutes = (time.perf_counter() - started) / 60
        save_fit(result, fingerprint, {**described, "minutes": minutes})
        evaluations = None if result.blocks is None else result.blocks.evaluations
        self.record.append(
            {**described, "minutes": minutes, "evaluations": evaluations, "cached": False}
        )
        return result

    def _estimate(
        self,
        spec: Specification,
        *,
        where: np.ndarray | None,
        parity: int | None,
        start: FitResult | None,
    ) -> FitResult:
        """One fit, from the rows in hand or from the cell file.

        A warm start is an optimisation, not part of any rule here: it is what turns a
        45-minute cold fit into 13 minutes by beginning at the nearest model's coefficients.
        When it fails, it fails as a *starting point* -- the prepayment model's backward
        elimination walked six damped Newton steps from 144 standard errors out to 3.86e+03,
        with damping at 1e+12, and then SLSQP diverged -- so the answer is to start where
        lifelines would have started and pay for it, not to give up on the model.
        """
        try:
            return self._once(spec, where=where, parity=parity, start=start)
        except exceptions.ConvergenceError:
            if start is None:
                raise
            log.warning("warm start did not converge; refitting cold: %s", spec.formula)
            return self._once(spec, where=where, parity=parity, start=None)

    def _once(
        self,
        spec: Specification,
        *,
        where: np.ndarray | None,
        parity: int | None,
        start: FitResult | None,
    ) -> FitResult:
        """One attempt, from the rows in hand or from the cell file."""
        initial_point = None if start is None else start.fitter.params_
        if self.blocks is None:
            if self.train is None:
                message = "Fits needs either an episode frame or a CellBlocks to read."
                raise ValueError(message)
            return fit_aft(
                self.train,
                spec.covariates,
                spec.formula,
                distribution=self.distribution,
                weights_col=WEIGHT,
                block_rows=self.block_rows,
                initial_point=initial_point,
                where=where,
            )
        # The sample is a property of the source, not a mask over rows the caller holds: a
        # half of the book taken while reading is the same half, and it never exists twice.
        source = replace(
            self.blocks, covariates=tuple(spec.covariates), vintage_parity=parity
        ).prepared()
        fitted = fit_streamed(
            source,
            spec.covariates,
            spec.formula,
            distribution=self.distribution,
            weights_col=WEIGHT,
            initial_point=initial_point,
            workers=self.workers,
            prefer=self.preferred,
        )
        if fitted.blocks is not None and fitted.blocks.method not in {"newton", "warm"}:
            self.preferred = fitted.blocks.method
        return fitted


def selection_description(
    *,
    identity: str,
    as_of: str,
    moratorium: str,
    formula: str,
    sample: str = "training half",
    distribution: str = DISTRIBUTION,
    cause: str = DEFAULT_CAUSE,
) -> dict[str, object]:
    """What a selection fit is saved under, and so how it is found again.

    The cause enters only when it is not ``default``, so every fit made before the
    prepayment model existed keeps the name it has.
    """
    described: dict[str, object] = {
        "purpose": "selection",
        "cells": identity,
        "as_of": as_of,
        "moratorium": moratorium,
        "sample": sample,
        "formula": formula,
        "distribution": distribution,
        "likelihood": "interval_censored",
    }
    if cause != DEFAULT_CAUSE:
        described["cause"] = cause
    return described


def selected_fit(
    *,
    identity: str,
    as_of: str,
    moratorium: str,
    formula: str,
    distribution: str = DISTRIBUTION,
    cause: str = DEFAULT_CAUSE,
) -> FitResult | None:
    """The selection's own fit of ``formula`` on the training half, if it made one.

    The model ``creditsurv report`` fits is the specification the selection ended on, on
    the same rows, so the selection has already found its optimum. Started there, the fit
    takes Newton steps to the same point instead of SLSQP's whole path from lifelines' seed.
    """
    cached = load_fit(
        fit_fingerprint(
            **selection_description(
                identity=identity,
                as_of=as_of,
                moratorium=moratorium,
                formula=formula,
                distribution=distribution,
                cause=cause,
            )
        )
    )
    return cached if isinstance(cached, FitResult) and cached.log_likelihood < 0 else None


@dataclass
class SelectionRecord:
    """Everything a selection run found, in the order it found it."""

    as_of: str
    moratorium: str
    distribution: str
    cause: str
    rows: int
    loan_months: float
    correlation: pd.DataFrame
    collinear: pd.DataFrame
    inflation: pd.DataFrame
    screening: pd.DataFrame
    elimination: pd.DataFrame
    stability: pd.DataFrame
    materiality: pd.DataFrame
    selected: Specification
    eliminated: dict[str, str]
    fits: list[dict[str, object]]

    def summary(self) -> dict[str, object]:
        """What the configuration should say, as data a test can compare it with."""
        continuous = self.selected.continuous
        return {
            "as_of": self.as_of,
            "moratorium": self.moratorium,
            "distribution": self.distribution,
            "cause": self.cause,
            "static_continuous": [name for name in continuous if name in LOAN_CONTINUOUS],
            "ordinal": [name for name in continuous if name in LOAN_ORDINAL],
            "time_varying_continuous": [name for name in continuous if name in MACRO_CANDIDATES],
            "categorical": dict(self.selected.categorical),
            "eliminated": self.eliminated,
            "formula": self.selected.formula,
        }


def run_selection(
    train: pd.DataFrame | None,
    fits: Fits,
    *,
    static: Sequence[str] = LOAN_CONTINUOUS,
    ordinal: Sequence[str] = LOAN_ORDINAL,
    macro: Sequence[str] = MACRO_CANDIDATES,
    base_categorical: Mapping[str, str] = BASE_CATEGORICAL,
    candidate_categorical: Mapping[str, str] = CANDIDATE_CATEGORICAL,
    stability: np.ndarray | bool | None = None,
    moments: Moments | None = None,
    signs: Mapping[str, int] = EXPECTED_SIGNS,
) -> SelectionRecord:
    """Run steps 5 to 10 and return what each found.

    ``train`` is the expanded training half, or ``None`` when ``fits`` reads the cell file
    and ``moments`` carries what steps 5 and 6 need. ``moments`` is the weighted covariance
    of the candidates with the row and exposure counts beside it: the selection used to take
    that pass twice, once for its correlation table and once inside the variance-inflation
    step, and on a training half of this size a pass is not cheap.

    ``stability`` says how step 9 takes its two halves: a boolean mask over ``train``, or
    ``True`` to take loans originated in even and in odd years while reading. ``None`` skips
    the step.

    The candidate lists default to the configuration and are parameters so a test can run
    the whole sequence on a handful of covariates.

    ``signs`` is the map of declared priors steps 7 and 8 read. It is an argument because
    the prepayment model is a model of a different exit and has different priors -- a credit
    score that lengthens survival shortens the time to repayment -- and a procedure that
    read one map for both would eliminate every covariate of the second model for
    disagreeing with the first model's economics.
    """
    loan = [*static, *ordinal]
    continuous = [*loan, *macro]
    eliminated: dict[str, str] = {}

    # 5. Pairs that say the same thing -- reported, not resolved.
    log.info("step 5: weighted correlation of %d candidates", len(continuous))
    if moments is None:
        if train is None:
            message = "run_selection needs either the rows or their moments."
            raise ValueError(message)
        moments = weighted_moments(train, continuous, weight=WEIGHT)
    # Checked before the first fit rather than found by an index error after it: on the
    # production table the pass that produced these moments is twenty minutes of parquet.
    absent = [name for name in continuous if name not in moments.covariance.index]
    if absent:
        message = f"The moments given cover {list(moments.covariance.index)}, missing {absent}."
        raise ValueError(message)
    deviations = moments.deviations
    correlation = moments.correlation
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
        None,
        continuous,
        weight=WEIGHT,
        threshold=VIF_THRESHOLD,
        priority=priority,
        covariance=moments.covariance,
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
    alone: dict[str, float] = {}
    for name, reference in candidates:
        try:
            result = fits.fit(base.plus(name, reference=reference), start=base_fit)
        except exceptions.ConvergenceError as error:
            # A candidate that cannot be fitted beside the loan block is a **finding about the
            # candidate**, not a crash: that is what a screen is for. It cost five hours to
            # learn that once -- the payment state diverged to coefficients of 1e+80 and took
            # the whole run with it, seventeen fits from the end -- and the run has no business
            # ending on the seventeenth of eighteen candidates.
            log.warning("step 7: %s did not converge beside the loan block", name)
            eliminated[name] = f"step 7: did not converge beside the loan block ({error!s:.120})"
            continue
        rows = _screen(name, reference, result, base_fit, deviations, signs=signs)
        screened.extend(rows)
        if reference is None:
            alone[name] = float(str(rows[0]["coef"]))
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
        worst = _worst(current, result, alone=alone, signs=signs)
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
    if stability is not None and stability is not False:
        log.info("step 9: stability on two halves")
        mask = None if isinstance(stability, bool) else stability
        while True:
            whole = fits.fit(current, start=previous)
            even = fits.fit(
                current,
                sample="even origination years",
                where=mask,
                parity=0,
                start=whole,
            )
            odd = fits.fit(
                current,
                sample="odd origination years",
                where=None if mask is None else ~mask,
                parity=1,
                start=whole,
            )
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
    rounds_table = pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame()

    # 10. Materiality. A macro covariate whose effect of one standard deviation on log
    # survival time is under the threshold contributes its sign and little else, and the
    # validation's objection was that covariates this small change sign when the sample
    # does. The threshold is declared in docs/rules.md, before any of these fits.
    #
    # One at a time, refitting between, for the reason step 8 gives: removing a covariate
    # moves every other coefficient, so a list of what was immaterial beside the whole model
    # is not a list of what is immaterial beside what remains.
    log.info("step 10: materiality")
    material: list[dict[str, object]] = []
    while True:
        result = fits.fit(current, start=previous)
        smallest = _immaterial(current, result, deviations, macro=macro)
        if smallest is None:
            break
        name, effect = smallest
        material.append(
            {
                "step": len(material) + 1,
                "removed": name,
                "effect_1sd": effect,
                "threshold": MATERIALITY_THRESHOLD,
                "remaining": len(current.covariates) - 1,
            }
        )
        eliminated[name] = (
            f"step 10: 1 sd effect {effect:+.4f} on log survival time, under "
            f"{MATERIALITY_THRESHOLD:g}"
        )
        current = current.minus(name)
        previous = result
    materiality = pd.DataFrame(material, columns=_MATERIALITY_COLUMNS)

    return SelectionRecord(
        as_of=fits.as_of,
        moratorium=fits.moratorium,
        distribution=fits.distribution,
        cause=fits.cause,
        rows=moments.rows,
        loan_months=moments.loan_months,
        correlation=correlation,
        collinear=collinear,
        inflation=inflation,
        screening=screening,
        elimination=elimination,
        stability=rounds_table,
        materiality=materiality,
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

_MATERIALITY_COLUMNS: Final = ["step", "removed", "effect_1sd", "threshold", "remaining"]


def _immaterial(
    spec: Specification,
    result: FitResult,
    deviations: pd.Series,
    *,
    macro: Sequence[str],
) -> tuple[str, float] | None:
    """The macro covariate step 10 removes next: the smallest effect under the threshold.

    The loan block is not eligible. A small coefficient on the credit score is a statement
    about this book; a small coefficient on a macro series is usually a statement about
    which of five correlated series happened to be left, and it is that instability the
    threshold is aimed at.
    """
    effects = {
        name: _number(_terms(result), name, "coef") * float(deviations[name])
        for name in spec.continuous
        if name in macro and name in deviations.index
    }
    below = {
        name: effect for name, effect in effects.items() if abs(effect) < MATERIALITY_THRESHOLD
    }
    if not below:
        return None
    name = min(below, key=lambda key: abs(below[key]))
    return name, below[name]


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
    *,
    signs: Mapping[str, int] = EXPECTED_SIGNS,
) -> list[dict[str, object]]:
    """One row per coefficient a candidate adds, and the likelihood ratio it earns."""
    summary = _terms(result)
    keys = (
        [name]
        if reference is None
        else [str(key) for key in summary.index if str(key).startswith(f"C({name},")]
    )
    statistic = 2.0 * (result.log_likelihood - base.log_likelihood)
    expected = signs.get(name, 0)
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


def _worst(
    spec: Specification,
    result: FitResult,
    *,
    alone: Mapping[str, float] | None = None,
    signs: Mapping[str, int] = EXPECTED_SIGNS,
) -> tuple[str, str, float, float] | None:
    """The covariate step 8 removes next, and why -- or ``None`` when every one stays.

    Three criteria, in order, and only one covariate per step, because removing it moves
    every other coefficient; of several meeting the same criterion the weakest -- smallest
    ``|z|`` -- goes first.

    * **A backwards sign** against a declared prior. It outranks everything else: it says
      the specification is wrong, not that the evidence is thin.
    * **A reversed sign** on a covariate with no declared prior: its coefficient in the
      full model points the other way from its coefficient ``alone`` beside the loan block at step
      7. This is the first run's marginal/conditional reversal rule, which removed
      ``corporate_bond_spread`` and ``yield_curve_slope`` and which ``docs/variable_selection.md``
      states "so it can be applied consistently rather than invoked when convenient". A covariate
      whose conditional effect contradicts its own is carrying something other than what its name
      says. The first version of this procedure did not run it; its first complete run kept three
      such covariates.
    * **A p-value above 0.05**, which at this sample size almost nothing reaches.

    Categorical terms carry no expected sign, are not screened alone, and at this sample
    size have no p-value.
    """
    summary = _terms(result)
    backwards: list[tuple[float, str, float, float]] = []
    reversed_: list[tuple[float, str, float, float]] = []
    thin: list[tuple[float, str, float, float]] = []
    for name in spec.continuous:
        coefficient = _number(summary, name, "coef")
        z = coefficient / _number(summary, name, "se(coef)")
        p_value = _number(summary, name, "p")
        expected = signs.get(name)
        own = None if alone is None else alone.get(name)
        if expected is not None and coefficient * expected < 0:
            backwards.append((abs(z), name, coefficient, p_value))
        elif expected is None and own is not None and coefficient * own < 0:
            reversed_.append((abs(z), name, coefficient, p_value))
        elif p_value > PVALUE_THRESHOLD:
            thin.append((p_value, name, coefficient, p_value))
    if backwards:
        _, name, coefficient, p_value = min(backwards)
        direction = "+" if signs[name] > 0 else "-"
        return (
            name,
            f"wrong sign: {coefficient:+.4g} where {direction} is expected",
            coefficient,
            p_value,
        )
    if reversed_ and alone is not None:
        _, name, coefficient, p_value = min(reversed_)
        return (
            name,
            f"reversed sign: {coefficient:+.4g} in the full model, "
            f"{alone[name]:+.4g} beside the loan block alone",
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
