"""The written-out likelihood is the one autograd differentiates.

`creditsurv.models.kernel` is the second implementation of lifelines' interval-censored
likelihood that the block engine was built to avoid, so the evidence that it is the same
likelihood has to be a comparison against autograd on the same expression -- value, gradient
and Hessian, at ordinary points and at every point where one of lifelines' clips binds.

The clips are the reason the comparison has to reach the extremes. `safe_exp` caps its argument
and still reports the derivative of the uncapped exponential; the Weibull's survival is
unclipped where the log-logistic's is clipped with a derivative of zero; the interval
probability is clipped and the left-truncation term beside it is not. Each of those is a wall
the optimiser backtracks from, and a wall that moves by an epsilon moves
`blocks._outside_the_domain` with it and changes which fits are refused and which are cached.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, hessian
from lifelines.utils.safe_exp import safe_exp

from creditsurv.models.kernel import (
    INTERVAL_CEILING,
    INTERVAL_FLOOR,
    MAX_EXPONENT,
    SURVIVAL_CEILING,
    SURVIVAL_FLOOR,
    log_times,
    row_likelihood,
)


def _weibull(
    params: np.ndarray,
    log_entry: np.ndarray,
    log_start: np.ndarray,
    log_stop: np.ndarray,
    left: np.ndarray,
) -> Any:
    """lifelines' Weibull AFT, as its own two methods write it.

    `_cumulative_hazard` is `safe_exp(rho * (log T - log lambda))` with `rho = safe_exp(...)`,
    and `WeibullAFTFitter` overrides `_survival_function` with `safe_exp(-ch)` -- no clip.
    """
    eta, coefficient = params[0], params[1]
    shape = safe_exp(coefficient)

    def cumulative(log_time: np.ndarray) -> Any:
        return safe_exp(shape * (log_time - eta))

    def survival(log_time: np.ndarray) -> Any:
        return safe_exp(-cumulative(log_time))

    interval = anp.clip(survival(log_start) - survival(log_stop), INTERVAL_FLOOR, INTERVAL_CEILING)
    return (anp.log(interval) + left * cumulative(log_entry))[0]


def _loglogistic(
    params: np.ndarray,
    log_entry: np.ndarray,
    log_start: np.ndarray,
    log_stop: np.ndarray,
    left: np.ndarray,
) -> Any:
    """lifelines' log-logistic AFT, as its own method and the base class write it.

    `_cumulative_hazard` is `logaddexp(beta * (log T - log(alpha)), 0)` with
    `alpha = safe_exp(...)` and `beta` a **plain** exponential, and this fitter does not
    override `_survival_function`, so the survival is `clip(exp(-ch), 1e-12, 1 - 1e-12)`.
    """
    eta, coefficient = params[0], params[1]
    log_scale = anp.log(safe_exp(eta))
    shape = anp.exp(coefficient)

    def cumulative(log_time: np.ndarray) -> Any:
        return anp.logaddexp(shape * (log_time - log_scale), 0.0)

    def survival(log_time: np.ndarray) -> Any:
        return anp.clip(anp.exp(-cumulative(log_time)), SURVIVAL_FLOOR, SURVIVAL_CEILING)

    interval = anp.clip(survival(log_start) - survival(log_stop), INTERVAL_FLOOR, INTERVAL_CEILING)
    return (anp.log(interval) + left * cumulative(log_entry))[0]


REFERENCE = {"weibull": _weibull, "loglogistic": _loglogistic}


def _agree(mine: np.ndarray, theirs: np.ndarray, where: str) -> None:
    """The kernel's derivatives against autograd's, judged the way the engine reads them.

    Two allowances, both measured rather than assumed.

    **The tolerance is against the size of the matrix, not of each entry.** The curvature in
    `eta` is a difference of two nearly equal survival functions, so on a row with a wide
    interval it is 1e-6 beside entries of 15, and two orderings of the same subtraction
    disagree in the last bit -- 1.1e-14 absolute, which is 1.6e-9 of that entry and 1e-15 of
    the matrix it sits in.

    **And where autograd's own answer is not a matrix, what has to agree is the verdict.** At
    eta = -500 with the shape on its bound, the log-logistic's plain exponential overflows and
    autograd returns `[[inf, -20.09], [nan, 10125]]` -- not finite and not even symmetric,
    which is why `blocks._symmetric` exists. The engine's response to such a curvature is a
    Cholesky that fails, a step that is refused and damping ten times larger, and a `nan`
    earns that exactly as an `inf` does. So the requirement there is that the kernel also
    refuses to call it a number.
    """
    if np.isfinite(theirs).all():
        scale = max(1.0, float(np.max(np.abs(theirs))))
        np.testing.assert_allclose(mine, theirs, rtol=1e-10, atol=1e-10 * scale, err_msg=where)
        return
    assert not np.isfinite(mine).all(), (
        f"{where}: autograd's answer is not a number here and the kernel's is, so a step this "
        "engine would refuse would be taken"
    )


def _rows(distribution: str) -> list[tuple[int, bool, float, float]]:
    """(age, exit, eta, shape coefficient) -- the ordinary and then each wall in turn."""
    ordinary = [
        (a, event, eta, coefficient)
        for a in (0, 1, 7, 120, 275)
        for event in (True, False)
        for eta in (5.0, 7.5, 9.0)
        for coefficient in (-0.5, 0.0, 0.35, 1.2)
    ]
    walls = [
        # The interval probability under its floor: a hazard so small that the two survivals
        # are the same number, which is where the clip starts and the objective stops falling.
        (120, False, 60.0, 0.35),
        (120, True, 80.0, 0.35),
        # And over its ceiling: a hazard so large that the interval is the whole of survival.
        (12, True, -40.0, 0.35),
        (1, True, -80.0, 1.2),
        # `safe_exp` capped: the shape at the bound the engine allows it, and a scale far
        # enough out that rho * (log t - eta) passes 634.78.
        (60, False, -500.0, 3.0),
        (60, True, -500.0, 3.0),
        (60, False, 500.0, 3.0),
        # The log-logistic's survival clip, which the Weibull does not have.
        (240, False, -30.0, 1.0),
        (240, True, 40.0, 1.0),
        # Age zero, where the time floor is what keeps the logarithm finite and the
        # truncation term is dropped.
        (0, True, 6.0, 0.35),
        (0, False, 6.0, 0.35),
    ]
    return ordinary + walls


# The reference raises these at the extreme rows: autograd's own VJP for division computes
# `-g * x / y**2` and the clipped interval probability makes `y` zero there. They are the
# reference's warnings, not the kernel's, and the rows are in the list on purpose.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
def test_the_written_out_likelihood_is_what_autograd_differentiates(distribution: str) -> None:
    reference = REFERENCE[distribution]
    gradient = grad(reference)
    curvature = hessian(reference)

    for age, event, eta, coefficient in _rows(distribution):
        entry, following, far = log_times(distribution, np.array([age]))
        log_entry = float(entry[0])
        log_start = log_entry if event else float(following[0])
        log_stop = float(following[0]) if event else far
        truncated = float(age > 0)

        # Both sides see length-one arrays, because numpy's `exp` and `log` take a different
        # code path for a scalar than for an array and the two disagree in the last bit: on a
        # log-likelihood of -9.449 that is 1.4e-12, which is nothing about the formula and
        # would hide a change that was.
        times = (np.array([log_entry]), np.array([log_start]), np.array([log_stop]))
        jet = row_likelihood(
            distribution,
            log_entry=times[0],
            log_start=times[1],
            log_stop=times[2],
            truncated=np.array([truncated]),
            eta=np.array([eta]),
            shape=coefficient,
        )
        params = np.array([eta, coefficient])
        where = f"{distribution} age={age} exit={event} eta={eta} r={coefficient}"

        expected = float(reference(params, *times, np.array([truncated])))
        assert float(np.asarray(jet.v)[0]) == expected, where

        first = np.asarray(gradient(params, *times, np.array([truncated])), dtype=float)
        got = np.array([np.asarray(jet.de + 0.0).ravel()[0], np.asarray(jet.dr + 0.0).ravel()[0]])
        _agree(got, first, where)

        second = np.asarray(curvature(params, *times, np.array([truncated])), dtype=float)
        mine = np.array(
            [
                [np.asarray(jet.dee + 0.0).ravel()[0], np.asarray(jet.der + 0.0).ravel()[0]],
                [np.asarray(jet.der + 0.0).ravel()[0], np.asarray(jet.drr + 0.0).ravel()[0]],
            ]
        )
        _agree(mine, second, where)


def test_the_time_floor_is_the_familys_own_and_age_zero_stays_finite() -> None:
    """Age zero is a real cell -- the month a loan is written -- and `log 0` is not a number.

    lifelines floors the time inside `_cumulative_hazard`, at `1e-100` for the Weibull and at
    `1e-25` for the log-logistic. Two different numbers, and both are visible in the answer, so
    neither can be replaced by the other or by a shared constant.
    """
    ages = np.array([0, 1, 2])
    weibull_entry, weibull_next, weibull_far = log_times("weibull", ages)
    logistic_entry, logistic_next, logistic_far = log_times("loglogistic", ages)

    assert weibull_entry[0] == pytest.approx(np.log(1e-100))
    assert logistic_entry[0] == pytest.approx(np.log(1e-25))
    assert weibull_entry[0] != logistic_entry[0]
    # Everything above the floor is the age itself, and the same for both families.
    np.testing.assert_allclose(weibull_entry[1:], np.log([1.0, 2.0]))
    np.testing.assert_allclose(logistic_entry[1:], np.log([1.0, 2.0]))
    np.testing.assert_allclose(weibull_next, np.log([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(logistic_next, np.log([1.0, 2.0, 3.0]))
    assert weibull_far == logistic_far == pytest.approx(np.log(1e25))


def test_a_family_the_kernel_does_not_write_out_is_refused_by_name() -> None:
    """The log-normal is in `FITTERS` and has no written-out form here, so it says so."""
    with pytest.raises(ValueError, match="not 'lognormal'"):
        row_likelihood(
            "lognormal",
            log_entry=np.zeros(1),
            log_start=np.zeros(1),
            log_stop=np.ones(1),
            truncated=np.zeros(1),
            eta=np.zeros(1),
            shape=0.0,
        )


def test_the_safe_exponential_reports_the_derivative_lifelines_reports() -> None:
    """Capped, and still differentiated as though it had not been.

    `defvjp(safe_exp, lambda ans, x: lambda g: g * ans)`. Mathematically the derivative above
    the cap is zero; autograd says it is the capped value, and the optimiser's path is drawn on
    autograd's surface, not on the mathematical one. Reproducing the cap but not its derivative
    would move a wall this engine's guards are calibrated against.
    """
    from creditsurv.models.kernel import _Jet, _safe_exp

    above = MAX_EXPONENT + 10.0
    jet = _safe_exp(_Jet(np.array([above]), de=1.0))
    capped = float(np.exp(MAX_EXPONENT))

    assert float(np.asarray(jet.v)[0]) == pytest.approx(capped)
    assert float(np.asarray(jet.de)[0]) == pytest.approx(capped), "not zero, as autograd has it"
    assert float(np.asarray(grad(lambda x: safe_exp(x))(above))) == pytest.approx(capped)
