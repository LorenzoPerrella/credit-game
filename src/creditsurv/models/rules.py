"""What decides an elimination, as pure functions of a coefficient table.

Every rule here takes a table and returns a verdict. None of them reads a file, a cell, or a
fitted model: the one line that stood between them and that was `_terms`, which reduces a
`FitResult` to the coefficient table of its scale parameter, and it stays in
:mod:`creditsurv.models.fits` where the fitting lives. So these can be tested with a table
written by hand -- and several of them were already being tested that way, through a faked
`FitResult` built only to be reduced to one.

**Each rule writes its own reason.** It did not, and the two halves drifted: step 8's sentence
came from the rule while step 9's was composed in the orchestration, re-reading the table the
rule had just handed back, with the step number added outside the string in one place and
inside it in the other. The step number is the only thing the procedure adds now.

The thresholds are here because they are the rules': declared in `docs/rules.md` before the
fits they govern, and read nowhere else.

What the rules are *for* is in `docs/variable_selection.md`. In short: at 60 million episodes
every p-value is zero, so significance decides almost nothing and the criteria that do the work
are **identification** -- a declared sign, a sign that reverses between the marginal and the
conditional fit, a standardised effect too small to be about anything but which of five
correlated series survived, and an effect whose sign changes when the sample does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from creditsurv.config import ECONOMIC_DIMENSION
from creditsurv.models.selection import EXPECTED_SIGNS, PVALUE_THRESHOLD

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from creditsurv.models.specification import Specification

#: Above this, in absolute Pearson correlation, two covariates are reported as saying the same
#: thing. Reported, not resolved: which of two collinear covariates to keep is a judgement
#: about what the model is for, and `nmds` makes it explicitly with a stated priority order.
CORRELATION_THRESHOLD: Final = 0.8

#: The smallest effect on log survival time, per standard deviation of the covariate, that
#: step 10 counts as material. Identification, not significance, is the reason: at 60 million
#: episodes every p-value is zero, and a macro covariate this small is usually a statement
#: about which of five correlated series happened to be left rather than about the book.
MATERIALITY_THRESHOLD: Final = 0.02


@dataclass(frozen=True)
class Removal:
    """A covariate a rule removes, and the sentence that says why.

    The sentence is the rule's own, and the procedure adds only the step number. What else a
    rule fills in is what the record publishes for that step: the coefficient and the p-value
    at step 8, the standardised effect at step 10, the partner carrying the information at
    step 9.
    """

    covariate: str
    reason: str
    coefficient: float = float("nan")
    p_value: float = float("nan")
    effect: float = float("nan")
    partner: str | None = None


def number(frame: pd.DataFrame, row: str, column: str) -> float:
    """One cell of a table as a float; the stubs type ``.loc`` as any scalar at all."""
    return float(cast("float", frame.loc[row, column]))


def effect_of(terms: pd.DataFrame, name: str, deviations: pd.Series) -> float:
    """Log survival time per standard deviation of the covariate."""
    return number(terms, name, "coef") * float(deviations[name])


def screen(
    name: str,
    reference: str | None,
    terms: pd.DataFrame,
    statistic: float,
    deviations: pd.Series,
    *,
    signs: Mapping[str, int] = EXPECTED_SIGNS,
) -> list[dict[str, object]]:
    """One row per coefficient a candidate adds, and the likelihood ratio it earns.

    ``statistic`` is twice the gain in log-likelihood over the model without this candidate,
    which the caller has because it holds both fits. The rule does not: it sees a table.
    """
    keys = (
        [name]
        if reference is None
        else [str(key) for key in terms.index if str(key).startswith(f"C({name},")]
    )
    expected = signs.get(name, 0)
    rows = []
    for key in keys:
        coefficient = number(terms, key, "coef")
        error = number(terms, key, "se(coef)")
        effect = coefficient * float(deviations[name]) if name in deviations.index else np.nan
        rows.append(
            {
                "covariate": name,
                "term": key,
                "coef": coefficient,
                "se": error,
                "z": coefficient / error,
                "p": number(terms, key, "p"),
                "effect_1sd": effect,
                "expected_sign": expected,
                "sign_agrees": expected == 0 or coefficient * expected > 0,
                "lr_statistic": statistic,
            }
        )
    return rows


def worst(
    terms: pd.DataFrame,
    spec: Specification,
    *,
    alone: Mapping[str, float] | None = None,
    signs: Mapping[str, int] = EXPECTED_SIGNS,
    skip: Collection[str] = (),
) -> Removal | None:
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
    backwards: list[tuple[float, str, float, float]] = []
    reversed_: list[tuple[float, str, float, float]] = []
    thin: list[tuple[float, str, float, float]] = []
    for name in (term for term in spec.continuous if term not in skip):
        coefficient = number(terms, name, "coef")
        z = coefficient / number(terms, name, "se(coef)")
        p_value = number(terms, name, "p")
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
        return Removal(
            covariate=name,
            reason=f"wrong sign: {coefficient:+.4g} where {direction} is expected",
            coefficient=coefficient,
            p_value=p_value,
        )
    if reversed_ and alone is not None:
        _, name, coefficient, p_value = min(reversed_)
        return Removal(
            covariate=name,
            reason=(
                f"reversed sign: {coefficient:+.4g} in the full model, "
                f"{alone[name]:+.4g} beside the loan block alone"
            ),
            coefficient=coefficient,
            p_value=p_value,
        )
    if thin:
        _, name, coefficient, p_value = max(thin)
        return Removal(
            covariate=name,
            reason=f"p = {p_value:.3g}",
            coefficient=coefficient,
            p_value=p_value,
        )
    return None


def stability_table(
    spec: Specification,
    whole: pd.DataFrame,
    even: pd.DataFrame,
    odd: pd.DataFrame,
    deviations: pd.Series,
) -> pd.DataFrame:
    """Each continuous covariate's standardised effect on the whole and on each half."""
    rows = []
    for name in spec.continuous:
        effects = [effect_of(terms, name, deviations) for terms in (whole, even, odd)]
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


def not_identified(table: pd.DataFrame) -> Removal | None:
    """The smallest unstable covariate that sits beside a larger one of its dimension.

    An unstable covariate with no such partner is kept and left visible in the table: the
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
        if partners.empty:
            continue
        name = str(row.covariate)
        partner = str(partners.iloc[-1]["covariate"])
        indexed = table.set_index("covariate")
        return Removal(
            covariate=name,
            reason=(
                f"1 sd effect {number(indexed, name, 'effect_even'):+.3f} on even "
                f"and {number(indexed, name, 'effect_odd'):+.3f} on odd origination years, "
                f"beside {partner} ({number(indexed, partner, 'effect_all'):+.3f}), both "
                f"{indexed.loc[name, 'dimension']}"
            ),
            partner=partner,
        )
    return None


def immaterial(
    spec: Specification,
    terms: pd.DataFrame,
    deviations: pd.Series,
    *,
    macro: Sequence[str],
) -> Removal | None:
    """The macro covariate step 10 removes next: the smallest effect under the threshold.

    The loan block is not eligible. A small coefficient on the credit score is a statement
    about this book; a small coefficient on a macro series is usually a statement about
    which of five correlated series happened to be left, and it is that instability the
    threshold is aimed at.
    """
    effects = {
        name: number(terms, name, "coef") * float(deviations[name])
        for name in spec.continuous
        if name in macro and name in deviations.index
    }
    below = {
        name: effect for name, effect in effects.items() if abs(effect) < MATERIALITY_THRESHOLD
    }
    if not below:
        return None
    name = min(below, key=lambda key: abs(below[key]))
    return Removal(
        covariate=name,
        reason=(
            f"1 sd effect {below[name]:+.4f} on log survival time, under {MATERIALITY_THRESHOLD:g}"
        ),
        effect=below[name],
    )
