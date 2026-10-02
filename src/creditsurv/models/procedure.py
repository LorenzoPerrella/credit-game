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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd
from lifelines import exceptions

from creditsurv.config import (
    ECONOMIC_DIMENSION,
    MACRO_CANDIDATES,
    MACRO_ELIMINATION_PRIORITY,
)
from creditsurv.data.panel import (
    WEIGHT,
)
from creditsurv.explore import collinear_pairs
from creditsurv.models.fits import _terms
from creditsurv.models.selection import (
    EXPECTED_SIGNS,
    PVALUE_THRESHOLD,
    VIF_THRESHOLD,
    Moments,
    stepwise_vif,
    weighted_moments,
)
from creditsurv.models.specification import Specification

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from creditsurv.models.aft import FitResult
    from creditsurv.models.fits import Fits

log = logging.getLogger(__name__)

#: Loan characteristics in the cell key that no earlier specification screened, with their
#: reference levels. Admitted to the key for the validation's M3; step 7 is where they meet
#: the same test as every other candidate, whatever it says.
CANDIDATE_CATEGORICAL: Final[dict[str, str]] = {
    "mortgage_insurance": "uninsured",
    "buyer_type": "repeat",
}

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
    # What a reading has to cover, said before the first specification exists: a reading is
    # made once and serves every fit of the run, so it cannot be narrowed to the first
    # formula's columns. The order is the one the candidate lists were given in.
    fits.covering = tuple(dict.fromkeys([*continuous, *base_categorical, *candidate_categorical]))

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
    #
    # The candidate model is **fitted before it is adopted**, so a removal that leaves a model
    # nobody can fit is a finding rather than the end of the run: rule 11 of `docs/rules.md`
    # keeps the covariate, records why, and offers the next-worst instead. On the prepayment
    # model that is not hypothetical -- removing `unemployment_change` from the thirteen macro
    # survivors leaves a specification every optimiser walks away from.
    log.info("step 8: backward elimination")
    current = base
    for name, reference in kept:
        current = current.plus(name, reference=reference)
    previous = base_fit
    # Which specification `previous` is a fit *of*. Carried from here to the end of step 10,
    # because `_floor` and `_check_nested` need it and only step 8 used to say it: the bound
    # that keeps a nested fit out of lifelines' clipped region was wired into one call site
    # out of four. It is not always a parent -- on the first pass here `previous` is the fit
    # of `base`, a model *smaller* than `current` -- and `_floor` returns nothing when the
    # subset test fails, which is why the honest thing to pass is what it is a fit of.
    previous_spec = base
    steps: list[dict[str, object]] = []
    unfittable: set[str] = set()
    while True:
        result = fits.fit(current, start=previous, parent=previous_spec)
        worst = _worst(current, result, alone=alone, signs=signs, skip=unfittable)
        if worst is None:
            break
        name, reason, coefficient, p_value = worst
        candidate = current.minus(name)
        try:
            fitted = fits.fit(candidate, start=result, parent=current)
        except exceptions.ConvergenceError as error:
            log.warning("step 8: without %s the model cannot be fitted; keeping it", name)
            unfittable.add(name)
            steps.append(
                {
                    "step": len(steps) + 1,
                    "removed": f"{name} (refused)",
                    "coef": coefficient,
                    "p": p_value,
                    "reason": f"kept: without it the model cannot be fitted ({error!s:.90})",
                    "remaining": len(current.covariates),
                }
            )
            continue
        steps.append(
            {
                "step": len(steps) + 1,
                "removed": name,
                "coef": coefficient,
                "p": p_value,
                "reason": reason,
                "remaining": len(candidate.covariates),
            }
        )
        eliminated[name] = f"step 8: {reason}"
        current, previous, previous_spec = candidate, fitted, candidate
    elimination = pd.DataFrame(steps, columns=_ELIMINATION_COLUMNS)

    # 9. Stability, on two halves of the book.
    rounds: list[pd.DataFrame] = []
    if stability is not None and stability is not False:
        log.info("step 9: stability on two halves")
        mask = None if isinstance(stability, bool) else stability
        while True:
            whole = fits.fit(current, start=previous, parent=previous_spec)
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
            previous, previous_spec = whole, current
            current = current.minus(name)
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
        result = fits.fit(current, start=previous, parent=previous_spec)
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
        previous, previous_spec = result, current
        current = current.minus(name)
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
    skip: Collection[str] = (),
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

    ``skip`` names covariates whose removal step 8 has tried and found to leave a model that
    cannot be fitted. They stay in the model and are not offered again.
    """
    summary = _terms(result)
    backwards: list[tuple[float, str, float, float]] = []
    reversed_: list[tuple[float, str, float, float]] = []
    thin: list[tuple[float, str, float, float]] = []
    for name in (term for term in spec.continuous if term not in skip):
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
