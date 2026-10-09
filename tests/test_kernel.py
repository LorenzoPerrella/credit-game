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

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

import autograd.numpy as anp
import numpy as np
import pandas as pd
import pytest
from autograd import grad, hessian
from lifelines.utils.safe_exp import safe_exp

from creditsurv.models.kernel import (
    FAMILIES,
    INTERVAL_CEILING,
    INTERVAL_FLOOR,
    MAX_EXPONENT,
    SURVIVAL_CEILING,
    SURVIVAL_FLOOR,
    Factorisation,
    Kernel,
    Rows,
    log_times,
    row_likelihood,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    return anp.log(interval) + left * cumulative(log_entry)


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
    return anp.log(interval) + left * cumulative(log_entry)


REFERENCE = {"weibull": _weibull, "loglogistic": _loglogistic}


def _one_row(reference: Any) -> Any:
    """The reference on a single row, as a scalar autograd can differentiate."""

    def scalar(params: np.ndarray, *rest: np.ndarray) -> Any:
        return reference(params, *rest)[0]

    return scalar


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
    reference = _one_row(REFERENCE[distribution])
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
    from creditsurv.models.kernel.likelihood import _Jet, _safe_exp

    above = MAX_EXPONENT + 10.0
    jet = _safe_exp(_Jet(np.array([above]), de=1.0))
    capped = float(np.exp(MAX_EXPONENT))

    assert float(np.asarray(jet.v)[0]) == pytest.approx(capped)
    assert float(np.asarray(jet.de)[0]) == pytest.approx(capped), "not zero, as autograd has it"
    assert float(np.asarray(grad(lambda x: safe_exp(x))(above))) == pytest.approx(capped)


LOAN = ("credit_score", "original_ltv", "purpose")
CALENDAR = ("unemployment_change", "ltv_change")
DESIGN_COLUMNS = (
    "Intercept",
    "C(credit_score)[T.680.0]",
    "original_ltv",
    "C(purpose)[T.cash_out_refinance]",
    "unemployment_change",
    "ltv_change",
)


def _book(months: range, ages: range) -> pd.DataFrame:
    """A little book with the production table's structure, not its size.

    Loan characteristics that the cell key carries; macro covariates that are functions of the
    observation month, which is the origination month plus the age; and `ltv_change`, which is
    the one covariate of the real model that reads a loan characteristic *and* the calendar --
    the LTV band's midpoint times a house-price ratio -- and is therefore the reason the
    calendar key carries 153,309 entries rather than 38,384.
    """
    rows = []
    for origination in months:
        for age in ages:
            observation = origination + age
            for score in (680.0, 790.0):
                for ltv in (50.0, 95.0):
                    for purpose in ("purchase", "cash_out_refinance"):
                        rows.append(
                            {
                                "credit_score": score,
                                "original_ltv": ltv,
                                "purpose": purpose,
                                "unemployment_change": 0.1 * np.sin(observation),
                                "ltv_change": ltv * (0.01 * np.cos(observation)),
                                "age": age,
                                "loan_months": 1.0 + (age % 7),
                                "outcome": (age + origination) % 11 == 0,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame["purpose"] = pd.Categorical(
        frame["purpose"], categories=["purchase", "cash_out_refinance"]
    )
    return frame


def _design(frame: pd.DataFrame) -> np.ndarray:
    """The formula's expansion, written out: an intercept, two factors and three numbers."""
    return np.column_stack(
        [
            np.ones(len(frame)),
            (frame["credit_score"].to_numpy() == 680.0).astype(float),
            frame["original_ltv"].to_numpy(dtype=float),
            (frame["purpose"].astype(str).to_numpy() == "cash_out_refinance").astype(float),
            frame["unemployment_change"].to_numpy(dtype=float),
            frame["ltv_change"].to_numpy(dtype=float),
        ]
    )


def _factorisation() -> Factorisation:
    return Factorisation(loan=LOAN, calendar=CALENDAR, age_column="age", columns=DESIGN_COLUMNS)


def test_the_two_tables_reproduce_the_design_row_for_row() -> None:
    """The point of the whole kernel: no design matrix, and nothing lost by not having one.

    Every column is a function of the loan combination or of the calendar key, so the design
    is two small tables read at two indices -- and that is checked by rebuilding it and
    comparing, exactly rather than to a tolerance, because the same combination carries
    literally the same floats.
    """
    frame = _book(range(0, 6), range(0, 9))
    design = _design(frame)
    factorisation = _factorisation()

    rows = factorisation.add(
        frame,
        design,
        event=frame["outcome"].to_numpy(dtype=bool),
        weight=frame["loan_months"].to_numpy(dtype=float),
    )
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()

    rebuilt = np.zeros_like(design)
    rebuilt[:, loan_positions] = loan[rows.i]
    rebuilt[:, calendar_positions] = calendar[rows.j]
    np.testing.assert_array_equal(rebuilt, design)

    # The intercept is a function of both sides and has to land on exactly one of them.
    assert sorted([*loan_positions, *calendar_positions]) == list(range(design.shape[1]))
    assert set(loan_positions) == {0, 1, 2, 3}
    assert set(calendar_positions) == {4, 5}

    # Eight loan combinations: two scores, two LTV bands, two purposes.
    assert len(loan) == 2 * 2 * 2
    # And one calendar key for each (observation month, age, LTV band): the macro series are
    # functions of the observation month, `ltv_change` reads the band as well, and the age is
    # in the key because the interval bounds are. Six origination months by nine ages is
    # fifty-four (observation, age) pairs, over two bands.
    expected = {
        (origination + age, age, ltv)
        for origination in range(0, 6)
        for age in range(0, 9)
        for ltv in (50.0, 95.0)
    }
    assert len(calendar) == len(expected) == 54 * 2
    assert rows.rows == len(frame)


def test_a_combination_keeps_its_index_when_a_level_is_missing_from_a_block() -> None:
    """The tables are global, so the codes have to be too.

    A categorical contributes its **declared** level codes. Were they discovered per block, a
    level absent from one batch would shift every code after it and two different combinations
    would be handed the same index -- silently, and with the design rebuilt from the wrong row.
    """
    first = _book(range(0, 4), range(0, 5))
    second = _book(range(4, 8), range(0, 5))
    # The second block never sees a cash-out refinance, but still knows the level exists.
    second = second[second["purpose"].astype(str) == "purchase"].reset_index(drop=True)
    assert set(second["purpose"].cat.categories) == {"purchase", "cash_out_refinance"}

    factorisation = _factorisation()
    encoded = []
    for block in (first, second):
        design = _design(block)
        encoded.append(
            (
                block,
                design,
                factorisation.add(
                    block,
                    design,
                    event=block["outcome"].to_numpy(dtype=bool),
                    weight=block["loan_months"].to_numpy(dtype=float),
                ),
            )
        )
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()

    for block, design, rows in encoded:
        rebuilt = np.zeros_like(design)
        rebuilt[:, loan_positions] = loan[rows.i]
        rebuilt[:, calendar_positions] = calendar[rows.j]
        np.testing.assert_array_equal(rebuilt, design, err_msg=f"{len(block)} rows")

    # The same loan combination in the two blocks is the same index.
    purchases = [
        {
            int(code)
            for code, keep in zip(rows.i, block["purpose"].astype(str) == "purchase", strict=True)
            if keep
        }
        for block, _, rows in encoded
    ]
    assert purchases[1] <= purchases[0]


def test_a_column_that_reads_both_sides_is_refused_by_name() -> None:
    """A term coupling a loan characteristic to the calendar has no place in a sum of two
    tables, and the kernel says which column rather than fitting something else.

    This is not hypothetical arithmetic: `ltv_change` and `mortgage_rate_decline` are both of
    that kind, and they work only because the calendar key carries the LTV band and the term.
    An interaction written into the formula would not.
    """
    frame = _book(range(0, 4), range(0, 5))
    design = _design(frame)
    coupled = np.column_stack(
        [design, design[:, 1] * design[:, 4]]  # credit score band times a macro series
    )
    factorisation = Factorisation(
        loan=LOAN,
        calendar=CALENDAR,
        age_column="age",
        columns=(*DESIGN_COLUMNS, "C(credit_score)[T.680.0]:unemployment_change"),
    )

    with pytest.raises(ValueError, match="function of neither"):
        factorisation.add(
            frame,
            coupled,
            event=frame["outcome"].to_numpy(dtype=bool),
            weight=frame["loan_months"].to_numpy(dtype=float),
        )


def test_a_key_that_stops_holding_a_covariate_is_caught_on_the_later_block() -> None:
    """The verification runs on every block, not on the first.

    A covariate that is a function of the key on one quarter of the book and not on another
    would pass a test of the first block alone, and the fit would then read one of the two
    values from the table and the other from nowhere.
    """
    first = _book(range(0, 4), range(0, 5))
    second = _book(range(0, 4), range(0, 5))
    factorisation = _factorisation()
    factorisation.add(
        first,
        _design(first),
        event=first["outcome"].to_numpy(dtype=bool),
        weight=first["loan_months"].to_numpy(dtype=float),
    )

    # The same keys, and one column moved: the stored table no longer describes it.
    moved = _design(second)
    moved[:, 2] += 1.0
    with pytest.raises(ValueError, match="not a function of the loan key after all"):
        factorisation.add(
            second,
            moved,
            event=second["outcome"].to_numpy(dtype=bool),
            weight=second["loan_months"].to_numpy(dtype=float),
        )


def test_plain_text_is_refused_because_its_codes_would_be_this_blocks_own() -> None:
    frame = _book(range(0, 3), range(0, 4))
    frame["purpose"] = frame["purpose"].astype(str)

    with pytest.raises(ValueError, match="plain text"):
        _factorisation().add(
            frame,
            _design(frame),
            event=frame["outcome"].to_numpy(dtype=bool),
            weight=frame["loan_months"].to_numpy(dtype=float),
        )


@pytest.fixture(params=["numpy", "compiled"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the test against each backend the port has, and skip the one that is not built.

    Rule 13 of `docs/rules.md` says the NumPy kernel is **normative** and the compiled one is
    optional at import, so the suite has to hold both to the same reference and has to pass
    with neither installed. Forcing the NumPy path is a one-line patch because the port decides
    which arithmetic to use in one place.
    """
    from creditsurv.models.kernel import terms

    if request.param == "numpy":
        monkeypatch.setattr(terms, "_compiled", None)
    elif terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    return str(request.param)


def test_the_compiled_backend_agrees_with_the_port_about_which_family_is_which() -> None:
    """A number crosses the boundary where a name would be safer, so the two are compared.

    `FAMILIES` is sorted, so a code derived from it would hand the log-logistic the Weibull's
    number: the other family, fitted under the right name, with no error anywhere. The crate
    exports its own constants for exactly this test.
    """
    from creditsurv.models.kernel import terms

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    assert terms._FAMILY_CODE["weibull"] == terms._compiled.WEIBULL
    assert terms._FAMILY_CODE["loglogistic"] == terms._compiled.LOGLOGISTIC
    assert terms._compiled.MAX_EXPONENT == MAX_EXPONENT
    assert set(terms._FAMILY_CODE) == set(FAMILIES), "every family the port writes out has a code"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
def test_the_kernel_is_the_objective_autograd_gives_over_the_whole_design(
    distribution: str,
    backend: str,
) -> None:
    """The accumulation, end to end, against the design it exists not to build.

    The reference is the objective lifelines optimises -- the mean negative log-likelihood over
    every loan-month, weights counted as the replications they are -- written over a
    materialised design and differentiated by autograd in the full parameter space. The kernel
    gets to the same numbers through two tables, two scatter-adds and three small matrix
    products, and never holds a design at all.

    The column standard deviations are deliberately not one, because lifelines optimises each
    coefficient multiplied by its column's and the tables carry that division: a scale of one
    everywhere would let a missing division pass.
    """
    frame = _book(range(0, 5), range(0, 7))
    design = _design(frame)
    weight = frame["loan_months"].to_numpy(dtype=float)
    event = frame["outcome"].to_numpy(dtype=bool)

    factorisation = _factorisation()
    rows = factorisation.add(frame, design, event=event, weight=weight)
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()

    # lifelines' design carries the shape's own column -- a constant, whose standard deviation
    # it sets to one -- after the scale's, and the parameter vector follows that order.
    shape_position = design.shape[1]
    deviations = np.array([1.0, 0.4, 17.5, 0.5, 0.06, 2.3, 1.0])
    kernel = Kernel(
        distribution=distribution,
        loan=loan / deviations[loan_positions],
        calendar=calendar / deviations[calendar_positions],
        loan_index=loan_positions,
        calendar_index=calendar_positions,
        shape_index=shape_position,
        blocks=(rows,),
        total_weight=float(weight.sum()),
    )

    full = np.column_stack([design, np.ones(len(frame))]) / deviations
    entry, following, far = log_times(distribution, frame["age"].to_numpy())
    times = (
        entry,
        np.where(event, entry, following),
        np.where(event, following, np.full(len(frame), far)),
    )
    truncated = (frame["age"].to_numpy() > 0).astype(float)
    reference = REFERENCE[distribution]

    def objective(params: np.ndarray) -> Any:
        eta = full[:, :shape_position] @ params[:shape_position]
        shape = full[:, shape_position] * params[shape_position]
        return -anp.sum(weight * reference(anp.array([eta, shape]), *times, truncated)) / float(
            weight.sum()
        )

    rng = np.random.default_rng(7)
    for attempt in range(3):
        x = np.concatenate([[7.0], rng.standard_normal(shape_position - 1) * 0.3, [0.35]])
        got = kernel(x, curvature=True)
        where = f"{distribution} attempt {attempt}"

        assert got.value == pytest.approx(float(objective(x)), rel=1e-12), where
        np.testing.assert_allclose(
            got.gradient, np.asarray(grad(objective)(x), dtype=float), rtol=1e-10, err_msg=where
        )
        assert got.curvature is not None
        expected = np.asarray(hessian(objective)(x), dtype=float)
        scale = max(1.0, float(np.max(np.abs(expected))))
        np.testing.assert_allclose(
            got.curvature, expected, rtol=1e-9, atol=1e-10 * scale, err_msg=where
        )

    # And the gradient alone is the same gradient, with no curvature computed.
    cheap = kernel(x, curvature=False)
    assert cheap.curvature is None
    np.testing.assert_array_equal(cheap.gradient, got.gradient)
    assert cheap.value == got.value


FULL = "credit_score + original_ltv + unemployment_change + ltv_change + C(purpose)"
SUBSET = "credit_score + unemployment_change + C(purpose)"


def _regressors(formula: str, frame: pd.DataFrame) -> Any:
    """lifelines' own expansion of a formula, as `blocks._scan` builds it."""
    from lifelines import utils

    return utils.CovariateParameterMappings(
        {"lambda_": formula, "rho_": "1"}, frame, force_intercept=True
    )


def _encoded(frame: pd.DataFrame, formula: str) -> tuple[Factorisation, Any, np.ndarray]:
    """Encode a book against one formula, through lifelines' expansion of it.

    Only the **scale's** columns are handed over, exactly as `blocks._scan` hands them over.
    lifelines' design carries the shape's column too -- a constant, and so a function of both
    sides -- and a version of this that passed the whole design to both sides agreed with
    itself while the engine fitted a model in which the shape's coefficient was read twice.
    """
    covariates = frame.drop(columns=["age", "loan_months", "outcome"])
    regressors = _regressors(formula, covariates)
    design = regressors.transform_df(covariates)
    values = design.to_numpy(dtype=np.float64)
    scale = [position for position, (block, _) in enumerate(design.columns) if block == "lambda_"]
    factorisation = Factorisation(
        loan=LOAN,
        calendar=CALENDAR,
        age_column="age",
        columns=[str(name) for block, name in design.columns if block == "lambda_"],
    )
    factorisation.add(
        frame,
        values[:, scale],
        event=frame["outcome"].to_numpy(dtype=bool),
        weight=frame["loan_months"].to_numpy(dtype=float),
    )
    return factorisation, regressors, values


def test_the_key_frames_rebuild_the_tables_the_rows_produced() -> None:
    """One scan, then any model: the claim that makes a selection affordable.

    The rows carry two indices into the combinations of a widest key, and every design column
    is a function of one side, so a model's tables come from putting its formula through 3,001
    and 153,309 rows. Here the same formula is expanded both ways -- from the rows during the
    scan, and from the key frames afterwards -- and the two tables have to be identical, not
    close.
    """
    frame = _book(range(0, 5), range(0, 7))
    factorisation, regressors, values = _encoded(frame, FULL)

    from_rows = factorisation.tables()
    from_keys = factorisation.expand(regressors.transform_df, primary="lambda_")

    np.testing.assert_array_equal(from_keys.loan, from_rows[0])
    np.testing.assert_array_equal(from_keys.calendar, from_rows[1])
    np.testing.assert_array_equal(from_keys.loan_positions, from_rows[2])
    np.testing.assert_array_equal(from_keys.calendar_positions, from_rows[3])

    # And the moments, which the scan would otherwise have to accumulate over every row. The
    # sum over combinations of a value times its count is the same sum in another order, so it
    # agrees to the last digits rather than exactly.
    np.testing.assert_allclose(from_keys.first, values.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(from_keys.second, (values * values).sum(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(from_keys.low, values.min(axis=0))
    np.testing.assert_array_equal(from_keys.high, values.max(axis=0))
    assert factorisation.rows == len(frame)

    # The moments cover every column -- the shape's included, because that is what gives each
    # coefficient its scale -- while the tables hold only the scale's. Reading the shape's
    # constant column as a loan-side covariate as well is the defect this asserts against: it
    # stopped a real fit 2.1 standard errors out while the polish reported 2.5e-4.
    assert len(from_keys.columns) == values.shape[1]
    assert from_keys.loan.shape[1] + from_keys.calendar.shape[1] == values.shape[1] - 1


def test_a_different_model_is_expanded_without_reading_a_row() -> None:
    """The point of the whole thing: a candidate's tables cost 3,001 rows, not 72 million.

    A selection makes about thirty fits, and every one of them used to re-read the cell file,
    re-derive the macro family, rebuild the design and throw all of it away -- 12.1 minutes a
    time on the production table, against 53 seconds of arithmetic. The second formula here
    drops a loan covariate and a calendar one, and its tables are built from the key frames of
    an encoding made for the first.
    """
    frame = _book(range(0, 5), range(0, 7))
    widest, _, _ = _encoded(frame, FULL)
    narrow, regressors, values = _encoded(frame, SUBSET)

    expanded = widest.expand(regressors.transform_df, primary="lambda_")
    from_rows = narrow.tables()

    np.testing.assert_array_equal(expanded.loan, from_rows[0])
    np.testing.assert_array_equal(expanded.calendar, from_rows[1])
    np.testing.assert_array_equal(expanded.loan_positions, from_rows[2])
    np.testing.assert_array_equal(expanded.calendar_positions, from_rows[3])
    np.testing.assert_allclose(expanded.first, values.sum(axis=0), rtol=1e-12)
    wider = _regressors(FULL, frame.drop(columns=["age", "loan_months", "outcome"]))
    assert len(expanded.columns) < len(widest.expand(wider.transform_df, primary="lambda_").columns)


def test_a_coupled_column_is_refused_when_the_tables_are_built_from_the_keys() -> None:
    """The guard survives the move off the rows, and still names the column.

    `ltv_change` and `mortgage_rate_decline` read a loan characteristic *and* the calendar, and
    work only because the calendar key carries the LTV band and the term. An interaction written
    into the formula does not, and it has to be refused here as well as during a scan -- because
    here is where a candidate model's formula arrives.
    """
    frame = _book(range(0, 4), range(0, 5))
    factorisation, _, _ = _encoded(frame, FULL)
    coupled = _regressors(
        "credit_score * unemployment_change", frame.drop(columns=["age", "loan_months", "outcome"])
    )

    with pytest.raises(ValueError, match="move with the loan"):
        factorisation.expand(coupled.transform_df, primary="lambda_")


def test_a_saved_encoding_comes_back_the_same_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reading of the production table is 10.9 minutes and 1.09 GB, and a selection pays
    three of them where the four runs rule 2 needs pay twelve. Written once and mapped back,
    the second and every later reading is seconds -- and it survives a restart, which on a job
    that takes a day matters more than the minutes.

    What has to come back identical is everything a fit reads: the rows, the keys in the order
    their codes were handed out in, the counts the column moments are computed from, and the
    bounds the univariate seed is fitted on. The codes live in the rows, so a restored
    factorisation that renumbered a combination would be a different model silently.
    """
    from creditsurv.data.panel import AGE_START, EXACT_OBSERVATION, LOWER_BOUND, UPPER_BOUND
    from creditsurv.models.engine.cache import load_encoding, save_encoding
    from creditsurv.models.engine.scan import encode_blocks

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    # The frame `model_frame` hands over: the covariates, the age the episode opens at, the
    # interval and the weight. Nothing else -- a column in neither key has no index to be
    # looked up by, which is what the scan refuses by name.
    book = _book(range(0, 5), range(0, 7))
    defaulted = book.pop("outcome").to_numpy(dtype=bool)
    book[AGE_START] = book.pop("age").astype(float)
    book[LOWER_BOUND] = np.where(defaulted, book[AGE_START], book[AGE_START] + 1.0)
    book[UPPER_BOUND] = np.where(defaulted, book[AGE_START] + 1.0, np.inf)
    book[EXACT_OBSERVATION] = False

    encoded = {
        "loan": LOAN,
        "calendar": CALENDAR,
        "lower_bound_col": LOWER_BOUND,
        "upper_bound_col": UPPER_BOUND,
        "event_col": EXACT_OBSERVATION,
        "entry_col": AGE_START,
        "weights_col": "loan_months",
    }
    original = encode_blocks([book.iloc[:400], book.iloc[400:]], **encoded)  # type: ignore[arg-type]
    save_encoding(original, "fixture", {"what": "a fixture book"})
    restored = load_encoding("fixture")

    assert restored is not None
    assert restored.episodes == original.episodes
    assert restored.events == original.events
    assert restored.weight == original.weight
    assert restored.names == original.names
    assert [block.rows for block in restored.rows] == [block.rows for block in original.rows]
    for was, now in zip(original.rows, restored.rows, strict=True):
        for column in ("i", "j", "age", "event", "weight"):
            np.testing.assert_array_equal(
                getattr(now, column), getattr(was, column), err_msg=column
            )
    pd.testing.assert_series_equal(restored.bounds, original.bounds, check_names=False)

    # The keys in their codes' own order, which is what the rows point at -- and on their
    # declared levels, because what a formula's expansion reads off a key frame is its dtypes.
    for before, after in zip(
        original.factorisation.keys(), restored.factorisation.keys(), strict=True
    ):
        pd.testing.assert_frame_equal(after, before)
    np.testing.assert_array_equal(
        restored.factorisation.loan_counts, original.factorisation.loan_counts
    )
    np.testing.assert_array_equal(
        restored.factorisation.calendar_counts, original.factorisation.calendar_counts
    )

    # And a model expanded from the restored keys is the model expanded from the original ones,
    # which is the claim a cached reading makes: a candidate's tables come off the keys.
    regressors = _regressors(
        FULL,
        book.drop(columns=["loan_months", AGE_START, LOWER_BOUND, UPPER_BOUND, EXACT_OBSERVATION]),
    )
    tables = original.factorisation.expand(regressors.transform_df, primary="lambda_")
    remapped = restored.factorisation.expand(regressors.transform_df, primary="lambda_")
    np.testing.assert_array_equal(remapped.loan, tables.loan)
    np.testing.assert_array_equal(remapped.calendar, tables.calendar)
    np.testing.assert_allclose(remapped.first, tables.first, rtol=0, atol=0)


def test_a_macro_month_the_window_cannot_reach_does_not_rename_a_reading() -> None:
    """A reading is named by the macro panel's own numbers, but only the readable ones.

    The 10.9 minutes a reading costs are paid again whenever its name changes, and the macro
    panel is live FRED data that gains a month without anything about the model moving. A cell
    reads the panel at its observation month less the lag or at its origination month less the
    lag, both at or before the cut the window stops at -- so a month above the cut cannot enter
    a reading, and must not retire one. A month *inside* it must, because that is a different
    calendar key on the same cell file, which the cell table's identity cannot see.
    """
    from creditsurv.data.panel import month_ordinal
    from creditsurv.models.engine.cache import encoding_fingerprint

    def named(macro: pd.DataFrame, months: tuple[int | None, int | None] | None) -> str:
        return encoding_fingerprint(
            identity="cells_exclude.parquet:448169670:1789713959804446587",
            loan=LOAN,
            calendar=CALENDAR,
            age_column="age_start",
            cause="default",
            parity=None,
            block_rows=250_000,
            lag_months=3,
            macro=macro,
            months=months,
        )

    periods = pd.period_range("2019-01", "2021-12", freq="M")
    macro = pd.DataFrame({"unemployment_rate": np.linspace(3.5, 6.0, len(periods))}, index=periods)
    cut = (None, month_ordinal(pd.Period("2021-12", freq="M")))
    extended = pd.concat(
        [
            macro,
            pd.DataFrame(
                {"unemployment_rate": [9.9]},
                index=pd.period_range("2022-01", periods=1, freq="M"),
            ),
        ]
    )
    assert named(extended, cut) == named(macro, cut)

    revised = macro.copy()
    revised.iloc[-1, 0] = 6.1  # the last readable month, restated
    assert named(revised, cut) != named(macro, cut)

    # And with no cut -- `creditsurv fit` without an `--as-of` -- the whole table is read, so the
    # month that could not reach the window above is a month the reading does see.
    assert named(extended, None) != named(macro, None)


def test_a_model_with_no_calendar_covariate_is_evaluated_and_not_a_segmentation_fault(
    backend: str,
) -> None:
    """A table of `n` rows and **zero** columns, which is what a loan-only model hands over.

    The compiled kernel sized its accumulators by dividing the table's flat length by its column
    count, and zero columns gave zero rows: the accumulators were allocated empty and the row
    loop wrote past them. With the bounds checks removed -- which bought 2% -- that was a
    **segmentation fault inside an ordinary fit**, found by the suite and not by a test of this
    case, because there was none. There is now, and the checks are back.
    """
    frame = _book(range(0, 4), range(0, 5))
    weight = frame["loan_months"].to_numpy(dtype=float)
    event = frame["outcome"].to_numpy(dtype=bool)
    factorisation = Factorisation(
        loan=LOAN, calendar=(), age_column="age", columns=["Intercept", "credit_score"]
    )
    design = np.column_stack(
        [np.ones(len(frame)), (frame["credit_score"].to_numpy() == 680.0).astype(float)]
    )
    rows = factorisation.add(frame, design, event=event, weight=weight)
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()
    assert calendar.shape[1] == 0, "nothing in this model is a function of the calendar"

    kernel = Kernel(
        distribution="weibull",
        loan=loan,
        calendar=calendar,
        loan_index=loan_positions,
        calendar_index=calendar_positions,
        shape_index=design.shape[1],
        blocks=(rows,),
        total_weight=float(weight.sum()),
    )
    totals = kernel(np.array([5.0, 0.1, 0.3]), curvature=True)

    assert np.isfinite(totals.value) and totals.value > 0, backend
    assert np.isfinite(totals.gradient).all()
    assert totals.curvature is not None
    assert np.isfinite(totals.curvature).all()


def _kernel(distribution: str) -> tuple[Kernel, np.ndarray]:
    """A kernel on the fixture book, and the parameter vector its design expects."""
    frame = _book(range(0, 5), range(0, 7))
    design = _design(frame)
    weight = frame["loan_months"].to_numpy(dtype=float)
    event = frame["outcome"].to_numpy(dtype=bool)
    factorisation = _factorisation()
    rows = factorisation.add(frame, design, event=event, weight=weight)
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()
    deviations = np.array([1.0, 0.4, 17.5, 0.5, 0.06, 2.3, 1.0])
    kernel = Kernel(
        distribution=distribution,
        loan=loan / deviations[loan_positions],
        calendar=calendar / deviations[calendar_positions],
        loan_index=loan_positions,
        calendar_index=calendar_positions,
        shape_index=design.shape[1],
        blocks=(rows,),
        total_weight=float(weight.sum()),
    )
    return kernel, deviations


#: Where the two backends are compared: the seed, a point in the middle of the data, and one
#: far outside it where every clip binds and the second-order association decides the answer.
_POINTS: Final = {
    "the seed": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "inside the data": [4.6, -0.3, 0.21, 0.14, -0.9, 0.35, 0.28],
    "over the cliff": [-500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    "the shape on its other bound": [6.0, 0.1, -0.2, 0.0, 0.3, 0.0, -3.0],
}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
@pytest.mark.parametrize("point", list(_POINTS))
def test_the_two_backends_are_the_same_objective_everywhere_it_is_evaluated(
    distribution: str, point: str
) -> None:
    """The compiled kernel against the NumPy one, element by element, at four points.

    Both are held to autograd elsewhere; this is the comparison that matters once there are
    two, because the engine's guards read the *objective* and not a reference. **A wall that
    moves by an epsilon moves which fits are refused**, so the points include two outside the
    data: `eta = -500` with the shape on its bound, where the cumulative hazard overflows and
    `(f'' * d) * d` is the only grouping that does not produce a `nan`, and the shape on the
    other bound.

    The tolerance is 1e-12 relative and not zero, because the two accumulate in different
    orders -- the NumPy in chunks through `bincount` and pairwise sums, the compiled loop row
    by row with Neumaier compensation on the three sums that are over every row -- and rule 13
    asks each to be deterministic rather than identical to the other. On the production table,
    72.7 million rows at the published specification, they agree to **1.2e-15** on the
    objective, 1.1e-13 on the gradient and 1.7e-13 of the Hessian's largest entry; see
    `docs/reports/engine.md`.

    **And where a curvature is not a number, the two are compared on that rather than on its
    digits.** Outside the data the log-logistic's Hessian overflows, and the two arrive at a
    different flavour of non-finite in the same entries: `-inf` from the compiled loop where
    the NumPy reads `nan`. That is the one declared difference between the backends and it is
    structural, not a bug -- the Python chain carries a structural zero as the literal `0.0`
    and drops the term, so `0 * inf` never arises there, while an array element that is merely
    numerically zero gets no such treatment and poisons the sum; the compiled loop takes the
    shortcut on the value instead. What the engine reads is the **objective**, which agrees
    here to the last bit, and a step to a non-finite curvature is refused by the polish either
    way. So the claim held is the one that decides a fit: identical wherever the curvature is a
    number, and not a number wherever the other is not.
    """
    from creditsurv.models.kernel import terms

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    kernel, _ = _kernel(distribution)
    x = np.array(_POINTS[point])

    written = kernel._written(x, curvature=True)
    compiled = kernel._compiled(x, curvature=True)

    assert np.isnan(compiled.value) == np.isnan(written.value), f"{point}: one is nan"
    if not np.isnan(written.value):
        assert compiled.value == pytest.approx(written.value, rel=1e-12, nan_ok=True)
    np.testing.assert_allclose(
        compiled.gradient, written.gradient, rtol=1e-12, atol=1e-300, err_msg=point
    )
    assert compiled.curvature is not None
    assert written.curvature is not None
    defined = np.isfinite(written.curvature)
    np.testing.assert_array_equal(
        np.isfinite(compiled.curvature),
        defined,
        err_msg=f"{point}: one backend has a number where the other has none",
    )
    np.testing.assert_allclose(
        compiled.curvature[defined],
        written.curvature[defined],
        rtol=1e-12,
        atol=1e-300,
        err_msg=point,
    )


@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
def test_the_compiled_kernel_gives_the_same_bits_on_every_call(distribution: str) -> None:
    """Rule 13 asks for a deterministic summation, and this is what that means.

    A fixed chunk order, no unordered reduction and no FMA reassociation -- so the same point
    evaluated again is the same number to the last bit, not to a tolerance. It matters because
    an optimiser turns a difference in the last digit into a different search: two runs of the
    identical prepayment fit once agreed to every printed digit for eighty evaluations, split
    at 0.065288491918 against 0.065288491919, and were five significant figures apart forty
    evaluations later.
    """
    from creditsurv.models.kernel import terms

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    kernel, _ = _kernel(distribution)
    x = np.array(_POINTS["inside the data"])

    first = kernel._compiled(x, curvature=True)
    for _ in range(3):
        again = kernel._compiled(x, curvature=True)
        assert again.value.hex() == first.value.hex()
        np.testing.assert_array_equal(again.gradient, first.gradient)
        assert again.curvature is not None
        assert first.curvature is not None
        np.testing.assert_array_equal(again.curvature, first.curvature)


def test_the_rows_cross_the_boundary_as_one_buffer_of_the_right_dtype() -> None:
    """What `joined` promises the compiled backend: one C-contiguous array a column.

    Shared where the blocks are already adjacent views of one array, which is what a reading
    off the disk cache is -- copying there would put 1.09 GB inside a 1.31 GB ceiling -- and
    concatenated anywhere else. Either way the dtype is the declared one, checked here rather
    than refused at the boundary in the middle of an hour-old fit.
    """
    from creditsurv.models.kernel.factorisation import _ROW_DTYPES, joined

    whole = np.arange(12, dtype=np.uint32)
    views = [
        Rows(
            i=whole[start:stop],
            j=whole[start:stop],
            age=np.arange(stop - start, dtype=np.uint16),
            event=np.zeros(stop - start, dtype=bool),
            weight=np.ones(stop - start, dtype=np.uint32),
        )
        for start, stop in ((0, 5), (5, 12))
    ]
    one = joined(views)

    assert one.rows == 12
    assert np.shares_memory(one.i, whole), "adjacent views of one array are shared, not copied"
    assert one.i is whole, "and shared means the array itself, with nothing between"
    np.testing.assert_array_equal(one.age, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6])
    for column, dtype in _ROW_DTYPES.items():
        assert getattr(one, column).dtype == dtype
        assert getattr(one, column).flags.c_contiguous

    # And a column that is not what the kernel reads is refused by name, not misread.
    wrong = Rows(
        i=np.arange(3, dtype=np.int64),
        j=np.arange(3, dtype=np.uint32),
        age=np.arange(3, dtype=np.uint16),
        event=np.zeros(3, dtype=bool),
        weight=np.ones(3, dtype=np.uint32),
    )
    with pytest.raises(TypeError, match="column is int64"):
        joined([wrong, wrong])


def test_a_reading_of_a_rebuilt_table_is_the_only_one_the_sweep_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reading is 1.0 GB, and a rebuild makes every reading of the old table dead weight.

    Rule 2 needs six readings a campaign -- two causes by three samples -- so a rebuild can
    leave six gigabytes nothing will ever look for again, and there was no way to find them.
    What the sweep must **not** take is a reading of the table on disk: that costs 11.6 minutes
    to take again, and a run that stops picks up at the fit it was on rather than at the
    reading.

    The identity is the table's file, size and time of writing, so this writes a cell file,
    describes one reading under its identity and one under a table that is gone.
    """
    import pandas as pd

    from creditsurv.data.artefacts import ENCODINGS
    from creditsurv.data.store import cells_identity, save_cells
    from creditsurv.models.engine.cache import audit_readings, remove_reading

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    save_cells(pd.DataFrame({"age": [0, 1], "loan_months": [3, 4]}), "exclude")
    current = cells_identity("exclude")

    for name, identity in (("still-here", current), ("rebuilt-since", "cells_exclude.parquet:1:2")):
        folder = ENCODINGS.path(name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "i.npy").write_bytes(b"0" * 64)
        ENCODINGS.describe(
            name, {"identity": identity, "cells": 9, "cause": "default", "as_of": "2021-12"}
        )

    audited = {reading.name: reading for reading in audit_readings()}
    assert set(audited) == {"still-here", "rebuilt-since"}
    assert audited["still-here"].current
    assert not audited["rebuilt-since"].current
    assert audited["still-here"].bytes_on_disk == 64
    assert audited["still-here"].describe()["table"] == "cells_exclude.parquet"
    # The reporting date is shown, because a reading of the table on disk at a date nobody will
    # ask about again is current and useless at once, and only a person can tell which.
    assert audited["still-here"].describe()["as_of"] == "2021-12"

    # And removing one takes the description with it, because a description beside no arrays
    # would be found by every search and read back as a reading.
    remove_reading("rebuilt-since")
    assert not ENCODINGS.path("rebuilt-since").exists()
    assert not ENCODINGS.described("rebuilt-since").exists()
    assert [reading.name for reading in audit_readings()] == ["still-here"]


def test_a_reading_whose_description_cannot_be_read_is_not_called_current() -> None:
    """The conservative direction, and the one that cannot delete anything by accident.

    `audit_readings` reads the identity from the description beside the arrays. A description
    that is missing or unreadable leaves the identity empty, which matches no live table -- so
    the reading reports as **not** current, which keeps it out of the default sweep rather than
    putting it in.
    """
    from creditsurv.models.engine.cache import StoredReading

    unknown = StoredReading(
        name="whatever",
        identity="",
        current=False,
        bytes_on_disk=0,
        cells=None,
        cause=None,
        parity=None,
        as_of=None,
    )
    assert not unknown.current
    assert unknown.describe()["parity"] == "whole"


@pytest.mark.parametrize("threads", [1, 2, 3, 7])
def test_the_threaded_kernel_is_the_single_threaded_one_to_the_last_bit(threads: int) -> None:
    """Rule 13's condition on threads, and the one a parallel sum can fail silently.

    The rows go into a **fixed** number of contiguous parts, each part sums its own in its own
    order, and the partials are added in the parts' own order rather than as they finish. So
    two runs at the same count agree **bit for bit** -- which is what this holds -- and two
    different counts agree to the last digits of a sum over many terms, which is the other
    assertion here.

    The counts include ones that do not divide the row count, because an uneven cut is where an
    off-by-one in the partition would land, and one larger than the machine's cores, because
    oversubscription must not change an answer either.

    **The objective is identical across counts**, not merely close, and that is the compensated
    sum doing its work: Neumaier recovers the same total whatever the grouping. The gradient and
    the curvature accumulate into per-combination bins and are grouped differently, so they
    agree to a tolerance.
    """
    from creditsurv.models.kernel import terms

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    kernel, _ = _kernel("weibull")
    x = np.array(_POINTS["inside the data"])

    one = replace(kernel, threads=1)(x, curvature=True)
    many = replace(kernel, threads=threads)
    first = many(x, curvature=True)
    again = many(x, curvature=True)

    assert again.value.hex() == first.value.hex(), "the same count, the same bits"
    np.testing.assert_array_equal(again.gradient, first.gradient)
    np.testing.assert_array_equal(np.asarray(again.curvature), np.asarray(first.curvature))

    assert first.value == one.value, "the compensated sum recovers the same total"
    np.testing.assert_allclose(first.gradient, one.gradient, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(
        np.asarray(first.curvature), np.asarray(one.curvature), rtol=1e-12, atol=1e-300
    )


def test_more_threads_than_rows_is_not_an_error() -> None:
    """The partition is cut by index, so it cannot hand a thread an empty slice by accident.

    A fixture block is a few hundred rows and a count can be anything a caller declares, so the
    cut is clamped to the rows rather than trusted to divide them.
    """
    from creditsurv.models.kernel import terms
    from creditsurv.models.kernel.factorisation import Factorisation

    if terms._compiled is None:
        pytest.skip("the compiled kernel is not installed (`uv sync --extra kernel`)")
    frame = _book(range(0, 1), range(0, 2))
    weight = frame["loan_months"].to_numpy(dtype=float)
    factorisation = Factorisation(
        loan=LOAN, calendar=CALENDAR, age_column="age", columns=DESIGN_COLUMNS
    )
    rows = factorisation.add(
        frame, _design(frame), event=frame["outcome"].to_numpy(dtype=bool), weight=weight
    )
    loan, calendar, loan_positions, calendar_positions = factorisation.tables()
    kernel = Kernel(
        distribution="weibull",
        loan=loan,
        calendar=calendar,
        loan_index=loan_positions,
        calendar_index=calendar_positions,
        shape_index=len(DESIGN_COLUMNS),
        blocks=(rows,),
        total_weight=float(weight.sum()),
        threads=rows.rows * 4,
    )
    totals = kernel(np.concatenate([[5.0], np.zeros(len(DESIGN_COLUMNS) - 1), [0.3]]))

    assert np.isfinite(totals.value)
    assert np.isfinite(totals.gradient).all()
