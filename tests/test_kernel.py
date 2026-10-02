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
import pandas as pd
import pytest
from autograd import grad, hessian
from lifelines.utils.safe_exp import safe_exp

from creditsurv.models.kernel import (
    INTERVAL_CEILING,
    INTERVAL_FLOOR,
    MAX_EXPONENT,
    SURVIVAL_CEILING,
    SURVIVAL_FLOOR,
    Factorisation,
    Kernel,
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
    from creditsurv.models.kernel import _Jet, _safe_exp

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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
def test_the_kernel_is_the_objective_autograd_gives_over_the_whole_design(
    distribution: str,
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
    """Encode a book against one formula, through lifelines' expansion of it."""
    covariates = frame.drop(columns=["age", "loan_months", "outcome"])
    regressors = _regressors(formula, covariates)
    design = regressors.transform_df(covariates)
    values = design.to_numpy(dtype=np.float64)
    factorisation = Factorisation(
        loan=LOAN,
        calendar=CALENDAR,
        age_column="age",
        columns=[str(name) for _, name in design.columns],
    )
    factorisation.add(
        frame,
        values,
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
    from_keys = factorisation.expand(regressors.transform_df)

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

    expanded = widest.expand(regressors.transform_df)
    from_rows = narrow.tables()

    np.testing.assert_array_equal(expanded.loan, from_rows[0])
    np.testing.assert_array_equal(expanded.calendar, from_rows[1])
    np.testing.assert_array_equal(expanded.loan_positions, from_rows[2])
    np.testing.assert_array_equal(expanded.calendar_positions, from_rows[3])
    np.testing.assert_allclose(expanded.first, values.sum(axis=0), rtol=1e-12)
    assert len(expanded.columns) < len(
        widest.expand(
            _regressors(FULL, frame.drop(columns=["age", "loan_months", "outcome"])).transform_df
        ).columns
    )


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
        factorisation.expand(coupled.transform_df)
