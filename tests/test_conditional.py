"""Rule 18: a monthly episode's likelihood written as what it is, and why it is the same estimator.

A row is at risk from age ``a`` and either reaches ``a + 1`` or fails before it, so conditional on
having entered it contributes ``-dH`` or ``log(1 - exp(-dH))``, with ``dH = H(a+1) - H(a)``.
lifelines writes the same number as ``log(clip(S(start) - S(stop))) + H(entry)``, with the
truncation term outside the clip, and on 13 November 2026 that asymmetry refused 401
evaluations of the prepayment model as "not a likelihood". These tests hold the conditional form
to three things: the exact value, computed in 60-digit arithmetic; lifelines, wherever lifelines
does not clip; and a likelihood everywhere else.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest

from creditsurv.data.panel import to_interval_censored
from creditsurv.models.engine.contract import POLISH_TOLERANCE_SE
from creditsurv.models.kernel.likelihood import (
    _MAX_AGE,
    log_times,
    row_likelihood,
    row_likelihood_conditional,
    step_constants,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from creditsurv.models.kernel.likelihood import _Jet

FAMILIES = ("weibull", "loglogistic")


def _exact(distribution: str, age: int, eta: float, shape: float, failed: bool) -> float:
    """The episode's log-probability in 60-digit arithmetic, as the textbook writes it."""
    getcontext().prec = 60
    rate = Decimal(repr(shape)).exp()
    scale = Decimal(repr(eta))

    def cumulative(time: int) -> Decimal:
        if time == 0:
            return Decimal(0)
        grown = (rate * (Decimal(time).ln() - scale)).exp()
        return (1 + grown).ln() if distribution == "loglogistic" else grown

    change = cumulative(age + 1) - cumulative(age)
    return float((1 - (-change).exp()).ln()) if failed else float(-change)


def _conditional(
    distribution: str, ages: np.ndarray, failed: np.ndarray, eta: np.ndarray, shape: float
) -> _Jet:
    log_age, step, zero = step_constants(np.arange(_MAX_AGE + 1))
    return row_likelihood_conditional(
        distribution,
        log_age=log_age[ages],
        step=step[ages],
        zero=zero[ages],
        event=failed,
        eta=eta,
        shape=shape,
    )


def _lifelines(
    distribution: str, ages: np.ndarray, failed: np.ndarray, eta: np.ndarray, shape: float
) -> _Jet:
    entry, following, far = log_times(distribution, np.arange(_MAX_AGE + 1))
    opens, closes = entry[ages], following[ages]
    return row_likelihood(
        distribution,
        log_entry=opens,
        log_start=np.where(failed, opens, closes),
        log_stop=np.where(failed, closes, far),
        truncated=(ages > 0).astype(np.float64),
        eta=eta,
        shape=shape,
    )


@pytest.mark.parametrize("distribution", FAMILIES)
def test_the_conditional_row_is_the_exact_episode_probability(distribution: str) -> None:
    """To the last digit, at the ages and predictors a mortgage book has.

    Measured against 60-digit arithmetic the conditional form is within 1.5e-15 everywhere,
    where lifelines' own is 8e-8 out on the log-logistic at age one: two survivals within
    1e-5 of one, subtracted. That is why lifelines is not the reference here.
    """
    for age in (0, 1, 2, 3, 12, 60, 200, 350):
        for eta in (4.0, 6.0, 8.0):
            for shape in (-0.3, 0.2, 0.6):
                for failed in (False, True):
                    ages = np.array([age])
                    value = float(
                        np.asarray(
                            _conditional(
                                distribution, ages, np.array([failed]), np.array([eta]), shape
                            ).v
                        )[0]
                    )
                    exact = _exact(distribution, age, eta, shape, failed)
                    assert value == pytest.approx(exact, rel=1e-14), (age, eta, shape, failed)


@pytest.mark.parametrize("distribution", FAMILIES)
def test_where_no_clip_binds_the_two_formulations_are_one_number(distribution: str) -> None:
    """The identity, read on rows where lifelines computes its own expression well.

    Twenty thousand rows inside the data, every age and both exits. lifelines' expression
    subtracts two survivals, so on a survivor whose month costs 1e-6 of log-likelihood it keeps
    fewer digits: up to 2.3e-11 relative on the Weibull, which is 5e-17 absolute, and 8e-8 on the
    log-logistic, whose survival is also clipped to 1 - 1e-12. The tolerances sit just above
    those, measured, and the test above holds the conditional form to the exact value, so what
    is checked here is that the two are one estimator, not which is closer.
    """
    rng = np.random.default_rng(5)
    count = 20_000
    ages = rng.integers(0, 330, count)
    failed = rng.random(count) < 0.3
    eta = rng.uniform(4.0, 9.0, count)
    shape = 0.4
    tolerance = 1e-10 if distribution == "weibull" else 1e-6
    np.testing.assert_allclose(
        np.asarray(_conditional(distribution, ages, failed, eta, shape).v),
        np.asarray(_lifelines(distribution, ages, failed, eta, shape).v),
        rtol=tolerance,
    )


@pytest.mark.parametrize("distribution", FAMILIES)
@pytest.mark.parametrize("failed", [False, True])
def test_the_derivatives_are_the_derivatives_of_the_value(distribution: str, failed: bool) -> None:
    """The jet against central differences, first and second order, both scalars."""
    rng = np.random.default_rng(11)
    count = 2_000
    ages = rng.integers(0, 330, count)
    outcome = np.full(count, failed)
    eta = rng.uniform(4.5, 8.0, count)
    shape = 0.3
    step = 1e-6

    def at(e: np.ndarray, s: float) -> _Jet:
        return _conditional(distribution, ages, outcome, e, s)

    jet = at(eta, shape)

    def numbers(value: np.ndarray | float) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=float), (count,))

    pairs = [
        ("de", (np.asarray(at(eta + step, shape).v) - np.asarray(at(eta - step, shape).v))),
        ("dr", (np.asarray(at(eta, shape + step).v) - np.asarray(at(eta, shape - step).v))),
        ("dee", numbers(at(eta + step, shape).de) - numbers(at(eta - step, shape).de)),
        ("drr", numbers(at(eta, shape + step).dr) - numbers(at(eta, shape - step).dr)),
        ("der", numbers(at(eta + step, shape).dr) - numbers(at(eta - step, shape).dr)),
    ]
    for name, difference in pairs:
        numeric = difference / (2 * step)
        analytic = numbers(getattr(jet, name))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-9, err_msg=name)


def _kernels(distribution: str) -> tuple[object, object]:
    from dataclasses import replace as copied

    from test_kernel import _kernel

    lifelines, _ = _kernel(distribution)
    return lifelines, copied(lifelines, formulation="conditional")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", FAMILIES)
def test_far_from_the_data_only_the_conditional_form_is_still_a_likelihood(
    distribution: str,
) -> None:
    """Where lifelines' objective goes negative, the conditional one stays at or above zero.

    `eta = -500` with the shape on its bound is the point the backend tests call "over the
    cliff": every hazard explodes, lifelines' interval probability is clipped at 1e-25 while its
    truncation term keeps growing, and its mean negative log-likelihood falls below zero -- which
    no likelihood can do. The conditional one is a mean of terms that are each at least zero.
    """
    from creditsurv.models.kernel.terms import Kernel

    lifelines, conditional = _kernels(distribution)
    assert isinstance(lifelines, Kernel) and isinstance(conditional, Kernel)
    x = np.array([-500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    assert lifelines(x).value < 0, "the fixture no longer reaches lifelines' clipped region"
    value = conditional(x).value
    assert value >= 0 or np.isinf(value)
    inside = np.array([4.6, -0.3, 0.21, 0.14, -0.9, 0.35, 0.28])
    assert conditional(inside).value == pytest.approx(lifelines(inside).value, rel=1e-6)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", FAMILIES)
@pytest.mark.parametrize(
    "point",
    [
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.6, -0.3, 0.21, 0.14, -0.9, 0.35, 0.28],
        [6.0, 0.1, -0.2, 0.0, 0.3, 0.0, -3.0],
    ],
)
def test_the_two_backends_write_the_same_conditional_objective(
    distribution: str, point: list[float]
) -> None:
    """The crate's conditional loop against the NumPy one, value, gradient and Hessian.

    1e-12, the tolerance the lifelines formulation is held to: the two sum in different orders
    and rule 13 asks each to be deterministic rather than identical to the other.
    """
    from creditsurv.models.kernel import terms

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    _, conditional = _kernels(distribution)
    assert isinstance(conditional, terms.Kernel)
    x = np.array(point)
    written = conditional._written(x, curvature=True)
    compiled = conditional._compiled(x, curvature=True)
    assert compiled.value == pytest.approx(written.value, rel=1e-12)
    np.testing.assert_allclose(compiled.gradient, written.gradient, rtol=1e-12, atol=1e-300)
    assert compiled.curvature is not None and written.curvature is not None
    np.testing.assert_allclose(compiled.curvature, written.curvature, rtol=1e-12, atol=1e-300)


def test_an_unknown_formulation_is_refused_by_name() -> None:
    from creditsurv.models.kernel.terms import Kernel

    lifelines, _ = _kernels("weibull")
    assert isinstance(lifelines, Kernel)
    with pytest.raises(ValueError, match="formulation"):
        replace(lifelines, formulation="clipped")


PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=-0.06,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
)
COVARIATES = ["credit_score", "ltv_change", "unemployment_change", "purpose"]
FORMULA = "credit_score + ltv_change + unemployment_change + C(purpose)"
CALENDAR = ["ltv_change", "unemployment_change"]


@pytest.fixture(scope="module")
def weighted(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    """Left-truncated episodes with frequency weights, as `tests/test_blocks.py` builds them."""
    panel, _ = build_panel(book_dir, macro_module, n_loans=900, seed=23, params=PARAMS)
    encoded = to_interval_censored(panel)
    encoded["loan_months"] = np.random.default_rng(5).integers(1, 6, len(encoded))
    encoded["purpose"] = encoded["purpose"].astype("category")
    encoded["credit_score"] = (encoded["credit_score"] - 700.0) / 50.0
    return encoded.sort_values(["purpose", "age"], kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("distribution", FAMILIES)
def test_the_optimum_is_the_one_lifelines_objective_certifies(
    weighted: pd.DataFrame, distribution: str
) -> None:
    """One estimator: the same rows fitted under both formulations land in the same place.

    Each fit is polished to within a thousandth of a standard error of its own optimum, so two
    optima of one objective sit within two thousandths of each other; the log-likelihoods agree
    to the digits lifelines' subtraction keeps.
    """
    from creditsurv.models.aft import fit_aft

    fits = {
        formulation: fit_aft(
            weighted,
            COVARIATES,
            FORMULA,
            weights_col="loan_months",
            calendar=CALENDAR,
            distribution=distribution,
            formulation=formulation,
        )
        for formulation in ("lifelines", "conditional")
    }
    lifelines, conditional = fits["lifelines"], fits["conditional"]

    assert conditional.formulation == "conditional"
    assert conditional.blocks is not None and conditional.blocks.formulation == "conditional"
    moved = (
        conditional.fitter.params_ - lifelines.fitter.params_
    ).abs() / lifelines.fitter.standard_errors_
    assert moved.max() < 2 * POLISH_TOLERANCE_SE, f"{moved.max()} standard errors apart"
    assert conditional.log_likelihood == pytest.approx(lifelines.log_likelihood, rel=1e-8)


def test_the_conditional_form_is_refused_where_autograd_would_trace_lifelines(
    weighted: pd.DataFrame,
) -> None:
    """Without the factorised rows the fit would trace lifelines' own function under this name."""
    from creditsurv.models.aft import fit_aft

    with pytest.raises(ValueError, match="written-out kernel"):
        fit_aft(weighted, COVARIATES, FORMULA, weights_col="loan_months", formulation="conditional")


def test_the_formulation_is_in_the_fingerprint_only_when_it_is_not_lifelines() -> None:
    """So no fit made before the option existed changes its name, and no two are confused."""
    from creditsurv.data.store import fit_fingerprint
    from creditsurv.models.fits import fit_description, selection_description

    plain = selection_description(identity="t", as_of="2021-12", moratorium="exclude", formula="x")
    marked = selection_description(
        identity="t", as_of="2021-12", moratorium="exclude", formula="x", formulation="conditional"
    )
    assert "formulation" not in plain
    assert marked["formulation"] == "conditional"
    assert fit_fingerprint(**plain) != fit_fingerprint(**marked)

    report = fit_description((10, 20), "x", as_of="2021-12", moratorium="exclude")
    report_marked = fit_description(
        (10, 20), "x", as_of="2021-12", moratorium="exclude", formulation="conditional"
    )
    assert "formulation" not in report
    assert report_marked["formulation"] == "conditional"
