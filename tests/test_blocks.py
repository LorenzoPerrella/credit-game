"""A fit in blocks is the fit lifelines makes.

The block engine mirrors private lifelines code, so the evidence that it is equivalent
has to be a fit of the same rows both ways, compared on every number a report reads.
These tests are that evidence, and the first thing to re-run against a new lifelines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import (
    AGE_START,
    EXACT_OBSERVATION,
    LOWER_BOUND,
    UPPER_BOUND,
    model_blocks,
    model_frame,
    to_interval_censored,
)
from creditsurv.models.aft import FITTERS
from creditsurv.models.blocks import (
    POLISH_TOLERANCE_SE,
    StoredColumn,
    _polish,
    fit_interval_censoring_in_blocks,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    from lifelines.fitters import ParametericAFTRegressionFitter

COVARIATES = ["credit_score", "ltv_change", "unemployment_change", "purpose"]
FORMULA = "credit_score + ltv_change + unemployment_change + C(purpose)"

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=-0.06,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def weighted(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    """Left-truncated episodes with frequency weights and a categorical covariate.

    Sorted by purpose, so the first blocks hold a single level: the design built from
    them must still carry a column for every other one.
    """
    panel, _ = build_panel(book_dir, macro_module, n_loans=900, seed=23, params=PARAMS)
    encoded = to_interval_censored(panel)
    encoded["loan_months"] = np.random.default_rng(5).integers(1, 6, len(encoded))
    encoded["purpose"] = encoded["purpose"].astype("category")
    # Centred and scaled, because what is measured here is the blocks, not the optimiser. On
    # the score in points SLSQP stops 2.3e-8 apart from lifelines' own run -- the same
    # objective, a flatter valley along the intercept -- which would hide a block error of
    # that size; the polish closes it in a real fit.
    encoded["credit_score"] = (encoded["credit_score"] - 700.0) / 50.0
    return encoded.sort_values(["purpose", "age"], kind="stable").reset_index(drop=True)


@pytest.fixture
def through_the_optimiser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fit fall through to the method chain, which is now the fallback.

    Damped Newton from the starting point goes first -- 10.18 minutes against 43.79 for SLSQP
    on the production table, to the same log-likelihood -- so the chain is reached only when
    Newton declines. Everything the chain does is still reachable and still has to work, and a
    test of it that let Newton answer instead would pass without exercising anything.
    """
    from creditsurv.models import blocks

    monkeypatch.setattr(blocks, "_newton_steps_first", lambda objective, start: None)


def single_level_rows(frame: pd.DataFrame) -> int:
    """A block size that leaves the first two blocks with one purpose only."""
    rows = int((frame["purpose"] == frame["purpose"].iloc[0]).sum() // 2)
    assert frame.iloc[:rows]["purpose"].nunique() == 1
    return rows


def stock_fit(
    fitter: ParametericAFTRegressionFitter, frame: pd.DataFrame
) -> ParametericAFTRegressionFitter:
    fitter.fit_interval_censoring(
        model_frame(frame, COVARIATES).assign(loan_months=frame["loan_months"].to_numpy()),
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        formula=FORMULA,
    )
    return fitter


def block_fit(
    fitter: ParametericAFTRegressionFitter,
    frame: pd.DataFrame,
    rows: int,
    *,
    polish: bool = False,
) -> ParametericAFTRegressionFitter:
    """The block engine, by default stopping where lifelines stops, to be compared with it."""
    fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(frame, COVARIATES, rows=rows, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=polish,
    )
    return fitter


@pytest.mark.parametrize("distribution", ["weibull", "loglogistic"])
def test_a_fit_in_blocks_is_the_fit_lifelines_makes(
    weighted: pd.DataFrame, distribution: str
) -> None:
    rows = single_level_rows(weighted)
    stock = stock_fit(FITTERS[distribution](), weighted)
    blocked = block_fit(FITTERS[distribution](), weighted, rows)

    assert len(weighted) > 3 * rows, "the fixture must split into several blocks"
    # Measured on this fixture: coefficients 5.0e-10 apart for the Weibull and 6.1e-13 for
    # the log-logistic, standard errors 9.4e-11, log-likelihood 6.4e-16, predictions
    # 3.0e-10. The blocks account for 8.4e-13 of that (the next test); the rest is where
    # the optimiser starts and stops, since the seed is fitted on distinct bounds and SLSQP
    # stops on a 1e-10 change in the objective. The tolerances leave a margin of 20 or more.
    pd.testing.assert_index_equal(blocked.params_.index, stock.params_.index)
    np.testing.assert_allclose(blocked.params_, stock.params_, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(blocked.standard_errors_, stock.standard_errors_, rtol=1e-8)
    assert blocked.log_likelihood_ == pytest.approx(stock.log_likelihood_, rel=1e-12)
    blocked_aic, stock_aic = blocked.AIC_, stock.AIC_
    assert blocked_aic == pytest.approx(stock_aic, rel=1e-12)
    pd.testing.assert_index_equal(blocked.summary.columns, stock.summary.columns)

    rows_to_score = model_frame(weighted.sample(60, random_state=1), COVARIATES)
    times = [1.0, 12.0, 60.0, 180.0]
    np.testing.assert_allclose(
        blocked.predict_cumulative_hazard(rows_to_score, times=times).to_numpy(),
        stock.predict_cumulative_hazard(rows_to_score, times=times).to_numpy(),
        rtol=1e-8,
    )


def test_the_answer_does_not_depend_on_the_block_size(weighted: pd.DataFrame) -> None:
    rows = single_level_rows(weighted)
    coarse = block_fit(FITTERS["weibull"](), weighted, rows)
    fine = block_fit(FITTERS["weibull"](), weighted, rows // 3)

    # Measured 8.4e-13 apart, with identical log-likelihoods: splitting differently only
    # reorders additions.
    np.testing.assert_allclose(fine.params_, coarse.params_, rtol=1e-10, atol=1e-12)
    assert fine.log_likelihood_ == pytest.approx(coarse.log_likelihood_, rel=1e-14)


def test_a_block_fit_reports_what_it_saw(weighted: pd.DataFrame) -> None:
    record = fit_interval_censoring_in_blocks(
        FITTERS["weibull"](),
        model_blocks(
            weighted, COVARIATES, rows=single_level_rows(weighted), weights_col="loan_months"
        ),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
    )

    assert record.rows == len(weighted)
    assert record.loan_months == weighted["loan_months"].sum()
    assert record.events == weighted.loc[weighted["event"].astype(bool), "loan_months"].sum()
    assert record.blocks > 3
    # The point of storing compactly: well under the 8 bytes a float64 cell would take.
    assert record.stored_bytes < 8 * record.rows * 4


def test_a_text_covariate_is_refused(weighted: pd.DataFrame) -> None:
    """Categorised block by block, a text column would give each block its own dummies."""
    text = weighted.assign(purpose=weighted["purpose"].astype(str))

    with pytest.raises(TypeError, match="category"):
        block_fit(FITTERS["weibull"](), text, single_level_rows(weighted))


def test_blocks_declaring_different_levels_are_refused(weighted: pd.DataFrame) -> None:
    rows = single_level_rows(weighted)
    first = model_frame(weighted.iloc[:rows], COVARIATES).assign(loan_months=1)
    second = model_frame(weighted.iloc[rows:], COVARIATES).assign(loan_months=1)
    first["purpose"] = first["purpose"].cat.remove_unused_categories()

    with pytest.raises(ValueError, match="levels"):
        fit_interval_censoring_in_blocks(
            FITTERS["weibull"](),
            [first, second],
            formula=FORMULA,
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            weights_col="loan_months",
        )


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(np.array([0.0, 1.0, 1.0, 0.0]), id="indicator"),
        pytest.param(np.array([0.4, -1.6, 0.4, 1.8, 1e25]), id="binned-with-stand-in"),
        pytest.param(np.arange(400, dtype=float), id="loan-ages"),
        pytest.param(
            np.random.default_rng(3).normal(size=70_000).astype(np.float32).astype(float),
            id="float32-macro",
        ),
        pytest.param(np.random.default_rng(4).normal(size=70_000), id="float64"),
    ],
)
def test_a_stored_column_expands_to_exactly_the_column_it_was(column: np.ndarray) -> None:
    stored = StoredColumn.of(column)

    assert np.array_equal(stored.expand(), column)
    assert stored.expand().dtype == np.float64


def test_a_warm_start_ends_where_a_cold_one_does_in_fewer_steps(weighted: pd.DataFrame) -> None:
    """Backward elimination refits a model one covariate smaller at every step.

    Starting from the larger model's coefficients must change how long the smaller fit
    takes and nothing about where it ends. Polished, both end at the optimum; unpolished
    they stopped up to 1.8e-4 apart, each wherever SLSQP's tolerance happened to run out.
    """
    rows = single_level_rows(weighted)
    larger = block_fit(FITTERS["weibull"](), weighted, rows, polish=True)

    def smaller(initial_point: pd.Series | None) -> tuple[pd.Series, pd.Series, int]:
        fitter = FITTERS["weibull"]()
        record = fit_interval_censoring_in_blocks(
            fitter,
            model_blocks(weighted, COVARIATES, rows=rows, weights_col="loan_months"),
            formula="credit_score + ltv_change + unemployment_change",
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            weights_col="loan_months",
            initial_point=initial_point,
            polish=True,
        )
        return fitter.params_, fitter.standard_errors_, record.evaluations

    cold, errors, cold_steps = smaller(None)
    warm, _, warm_steps = smaller(larger.params_)

    assert float(((warm - cold).abs() / errors).max()) < 2 * POLISH_TOLERANCE_SE
    # SLSQP from the same warm start took 27 evaluations, as many as from a cold one.
    assert warm_steps < cold_steps, f"{warm_steps} evaluations warm against {cold_steps} cold"


def test_the_polish_reaches_the_optimum_the_optimiser_stops_short_of(
    weighted: pd.DataFrame,
) -> None:
    """SLSQP stops on a change of 1e-10 in the mean log-likelihood.

    A tolerance that takes no account of how precisely the data pin a coefficient down: on
    four quarters of the book it stopped up to 5.9 standard errors short of the optimum,
    and 2.0 on sixteen. Newton steps on lifelines' own gradient and Hessian finish the job,
    and the likelihood they reach is at least the one lifelines stops at.
    """
    stock = stock_fit(FITTERS["weibull"](), weighted)
    fitter = FITTERS["weibull"]()
    record = fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(
            weighted, COVARIATES, rows=single_level_rows(weighted), weights_col="loan_months"
        ),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=True,
    )

    assert record.residual_error_se < POLISH_TOLERANCE_SE
    assert record.residual_error_se <= record.stopping_error_se
    assert fitter.log_likelihood_ >= stock.log_likelihood_ - 1e-9 * abs(stock.log_likelihood_)


class _Cliff:
    """One parameter, whose Newton step from far out lands past a cliff.

    ``log cosh(x - 1)`` has its minimum at 1 and almost no curvature far from it, so the
    Newton step from -3 is some 750 long. Past 10 the objective is negative, as lifelines'
    clipped likelihood becomes once the parameters leave the region it is computed exactly
    in: impossible for a mean negative log-likelihood, and lower than the true minimum.
    """

    total_weight = 1e6

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if x[0] > 10:
            return -4604.0, np.zeros(1)
        return float(np.log(np.cosh(x[0] - 1.0)) + 0.01), np.array([np.tanh(x[0] - 1.0)])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if x[0] > 10:
            return np.zeros((1, 1))
        return np.array([[1.0 / np.cosh(x[0] - 1.0) ** 2]])


def test_the_polish_never_steps_into_values_no_likelihood_can_take() -> None:
    """On the training half a warm start's full Newton step went 8.31e5 standard errors, to
    an objective of -4604 -- lifelines' clipping, not a better fit. It lowered the objective,
    so it was taken; the fit fell back on SLSQP for 76 minutes, and with the gradient and the
    curvature both flat past the cliff it could as well have stopped there. Steps that refuse
    impossible values, damped until they lower the objective, reach the optimum instead."""
    objective = _Cliff()
    start = np.array([-3.0])
    value, gradient = objective(start)

    point, reached, _, _, began, left = _polish(
        cast("Any", objective), start, value, gradient, objective.hessian(start)
    )

    assert began > 1e3
    assert reached >= 0
    assert abs(float(point[0]) - 1.0) < 1e-6
    assert left < POLISH_TOLERANCE_SE


def test_the_slicer_hands_lifelines_the_columns_it_asks_for(macro: pd.DataFrame) -> None:
    """The design is handed to the likelihood through a slicer of our own, and the likelihood
    is lifelines' -- so the slicer has to answer exactly as lifelines' own does.

    It exists because pandas' answer was measured at 32% of a value-and-gradient, plus the
    copies it made: a parameter's columns are adjacent in the design, so asking for them is a
    view, and the masks the likelihood filters by are properties of the data, so a repeated
    filter is free. Together they made an evaluation 1.9 times faster.
    """
    from lifelines.utils import DataframeSlicer

    from creditsurv.models.blocks import _Slicer

    columns = pd.MultiIndex.from_tuples(
        [("lambda_", "Intercept"), ("lambda_", "credit_score"), ("rho_", "Intercept")]
    )
    design = np.asfortranarray(np.arange(30, dtype=float).reshape(10, 3))
    theirs = DataframeSlicer(pd.DataFrame(design, columns=columns))
    ours = _Slicer(design, columns)
    mask = np.zeros(10, dtype=bool)
    mask[[1, 4, 7]] = True

    for key in ("lambda_", "rho_"):
        np.testing.assert_array_equal(ours[key], theirs[key])
        np.testing.assert_array_equal(ours.filter(mask)[key], theirs.filter(mask)[key])
        np.testing.assert_array_equal(ours.filter(~mask)[key], theirs.filter(~mask)[key])
    assert ours.size == theirs.size
    assert ours.filter(mask).size == 3

    # The same mask twice is the same object, and the columns come back contiguous -- the
    # first version of this returned C-ordered rows and was 60% slower than the pandas it
    # replaced.
    assert ours.filter(mask) is ours.filter(mask.copy())
    assert ours["lambda_"].flags["F_CONTIGUOUS"]
    assert ours.filter(mask)["lambda_"].flags["F_CONTIGUOUS"]


def test_another_optimiser_is_tried_when_lifelines_own_stops_short(
    weighted: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    through_the_optimiser: None,
) -> None:
    """SLSQP solves a quadratic subproblem at each step, and on an ill-conditioned design it
    reports "Rank-deficient equality constraint subproblem" and gives up -- which is what the
    prepayment model did at step 8 of its selection, cold, at a finite objective of 56.58.

    The estimator is unchanged by trying another path: the same likelihood on the same rows has
    the same optimum, and the polish certifies the answer is at it. This holds the mechanism --
    the fit comes out of the fallback, is a real fit, and says which method found it.
    """
    from scipy import optimize

    from creditsurv.models import blocks

    calls: list[str] = []
    real = optimize.minimize

    def refuses_slsqp(*args: object, **kwargs: object) -> object:
        method = str(kwargs.get("method"))
        calls.append(method)
        if method.lower() == "slsqp":
            # Inside the likelihood but nowhere near the optimum, so the polish cannot finish
            # from it and the next method is tried. A point *outside* the likelihood is a
            # different matter and stops the chain -- that is a fact about the surface, not the
            # method, and the test below covers it.
            failed = real(*args, **{**kwargs, "options": {"maxiter": 1}})
            failed.success = False
            failed.message = "Rank-deficient equality constraint subproblem HFTI"
            failed.x = np.asarray(failed.x, dtype=float) + 40.0
            failed.fun = 1.0
            return failed
        return real(*args, **kwargs)

    monkeypatch.setattr(blocks, "minimize", refuses_slsqp)

    fitter = FITTERS["weibull"]()
    record = fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=True,
    )

    assert calls[0].lower() == "slsqp"
    assert calls[1] == "L-BFGS-B", "the first fallback is tried next"
    assert record.method == "l-bfgs-b"
    assert record.evaluations > 0
    assert fitter.log_likelihood_ < 0, "a real fit came out of the fallback"
    assert record.residual_error_se < 1e-3, "and the polish certifies it is at the optimum"


def test_the_optimisers_report_is_used_for_nothing_but_its_point(
    weighted: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    through_the_optimiser: None,
) -> None:
    """Each optimiser reports its value and gradient its own way: SLSQP's ``jac`` is the
    gradient, trust-constr's is shaped for its constraint machinery. Reading that field cost a
    run -- trust-constr solved a prepayment fit SLSQP had given up on, and the polish died on
    `LinAlgError: Incompatible dimensions`, an hour of fitting thrown away for a shape.

    So both are recomputed from the objective, and here the optimiser returns nonsense in
    those fields to prove nothing reads them.
    """
    from scipy import optimize

    from creditsurv.models import blocks

    real = optimize.minimize

    def reports_nonsense(*args: object, **kwargs: object) -> object:
        results = real(*args, **kwargs)
        results.jac = "not a gradient"
        results.fun = float(results.fun)  # kept finite: the engine checks convergence on it
        return results

    monkeypatch.setattr(blocks, "minimize", reports_nonsense)

    fitter = FITTERS["weibull"]()
    record = fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=True,
    )

    assert record.residual_error_se < 1e-3, "the polish ran on a recomputed gradient"
    assert fitter.log_likelihood_ < 0


def test_the_optimiser_that_worked_is_tried_first_next_time() -> None:
    """A design that defeats SLSQP is usually beside another one just like it: the backward
    elimination refits nearly the same model at every step. Walking the whole chain each time
    cost over an hour a fit on the prepayment model -- 25 minutes of SLSQP failing, 40 of
    L-BFGS-B, then an hour of trust-constr answering.
    """
    from creditsurv.models.blocks import _methods

    assert _methods("SLSQP", None) == ("SLSQP", "L-BFGS-B", "trust-constr")
    assert _methods("SLSQP", "slsqp") == ("SLSQP", "L-BFGS-B", "trust-constr")
    assert _methods("SLSQP", "trust-constr") == ("trust-constr", "SLSQP", "L-BFGS-B")
    assert _methods("SLSQP", "l-bfgs-b") == ("L-BFGS-B", "SLSQP", "trust-constr")
    # A method nobody offers is ignored rather than tried.
    assert _methods("SLSQP", "newton") == ("SLSQP", "L-BFGS-B", "trust-constr")


def test_the_shape_is_bounded_far_more_tightly_than_the_coefficients() -> None:
    """The two are not comparable, and the first bound I wrote was aimed at the wrong one. A
    scale coefficient of 100 is absurd but harmless to evaluate; the shape sits in an exponent,
    and the same number overflows the cumulative hazard and takes the objective with it.

    Three on the log scale is a shape between 0.05 and 20, where **125 converged fits** on this
    book put it between 1.07 and 1.62. A fit that ends on a bound is refused: that is the edge
    of where the likelihood can be evaluated, not a maximum.
    """
    from lifelines import exceptions

    from creditsurv.models.blocks import (
        _PARAMETER_BOUND,
        _SHAPE_BOUND,
        _check_interior,
        _limits,
    )

    columns = pd.MultiIndex.from_tuples(
        [("lambda_", "Intercept"), ("lambda_", "credit_score"), ("rho_", "Intercept")]
    )
    limits = _limits(columns, "lambda_")

    assert limits == [
        (-_PARAMETER_BOUND, _PARAMETER_BOUND),
        (-_PARAMETER_BOUND, _PARAMETER_BOUND),
        (-_SHAPE_BOUND, _SHAPE_BOUND),
    ]
    assert _SHAPE_BOUND < _PARAMETER_BOUND / 10, "the shape is bounded in a different league"

    _check_interior(np.array([0.5, -30.0, 0.4]), limits)  # interior: nothing happens
    with pytest.raises(exceptions.ConvergenceError, match="not identified"):
        _check_interior(np.array([0.5, 0.5, _SHAPE_BOUND]), limits)
    with pytest.raises(exceptions.ConvergenceError, match="Parameter\\(s\\) \\[0\\]"):
        _check_interior(np.array([-_PARAMETER_BOUND, 0.1, 0.4]), limits)


def test_a_method_that_stops_short_of_its_tolerance_is_finished_by_the_polish(
    weighted: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    through_the_optimiser: None,
) -> None:
    """lifelines caps SLSQP at 200 iterations, and a fit that reaches the cap comes back
    `success=False` although it is at the answer: on the prepayment model one had been stable
    to nine significant figures for twenty evaluations when it got there.

    Discarding it would have thrown away two and a half hours and started another method from
    scratch. What settles whether a method worked is the distance to the optimum, which only
    the polish measures.
    """
    from scipy import optimize

    from creditsurv.models import blocks

    calls: list[str] = []
    real = optimize.minimize

    def stops_short(*args: object, **kwargs: object) -> object:
        calls.append(str(kwargs.get("method")))
        results = real(*args, **kwargs)
        results.success = False
        results.message = "Iteration limit reached"
        return results

    monkeypatch.setattr(blocks, "minimize", stops_short)

    fitter = FITTERS["weibull"]()
    record = fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=True,
    )

    assert calls == ["SLSQP"], "no other method was needed"
    assert record.method == "slsqp"
    assert record.residual_error_se < 1e-3


def test_the_region_that_is_not_a_likelihood_is_a_wall() -> None:
    """The objective is a mean negative log-likelihood and cannot be negative. lifelines clips
    the interval probability but adds the truncation term unclipped, so beyond a ridge the
    surface falls away -- the worst point seen on the production table read -8.97e+69.

    Reported at face value that region is the most attractive place on the surface, and on the
    prepayment model **six attempts in a row** ended there: warm and cold, SLSQP, L-BFGS-B and
    trust-constr alike, with the coefficients bounded at 100 throughout. Reported as infinite
    it is a wall, which is how a domain boundary is told to an optimiser.
    """
    from creditsurv.models.blocks import _outside_the_domain

    x = np.array([1.0, -2.0, 0.5])

    assert _outside_the_domain(0.0176, x) is None, "a possible value is left alone"
    assert _outside_the_domain(0.0, x) is None, "zero is possible, if unlikely"
    for impossible in (-1e-9, -8.97e69, float("-inf"), float("nan"), float("inf")):
        answer = _outside_the_domain(impossible, x)
        assert answer is not None
        value, gradient = answer
        assert value == float("inf")
        np.testing.assert_array_equal(gradient, np.zeros_like(x))


def test_a_point_outside_the_likelihood_stops_the_chain(
    weighted: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    through_the_optimiser: None,
) -> None:
    """Ending outside the likelihood is a statement about the **surface**: the optimiser found
    nothing better than the wall, so the specification has the spurious minimum lifelines'
    unclipped truncation term creates -- and every other method finds it too, measured six times
    out of six on the prepayment model, warm and cold, with both bounds in place.

    Trying the rest costs hours and tells us what we already know, so the engine stops and lets
    the caller decide. Step 8 keeps the covariate: rule 11 of docs/rules.md.
    """
    from lifelines import exceptions
    from scipy import optimize

    from creditsurv.models import blocks

    calls: list[str] = []
    real = optimize.minimize

    def leaves_the_likelihood(*args: object, **kwargs: object) -> object:
        calls.append(str(kwargs.get("method")))
        results = real(*args, **{**kwargs, "options": {"maxiter": 1}})
        results.success = False
        results.fun = float("inf")
        results.message = "the wall"
        return results

    monkeypatch.setattr(blocks, "minimize", leaves_the_likelihood)

    with pytest.raises(exceptions.ConvergenceError, match="unbounded below"):
        fit_interval_censoring_in_blocks(
            FITTERS["weibull"](),
            model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
            formula=FORMULA,
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            weights_col="loan_months",
            polish=True,
        )

    assert calls == ["SLSQP"], "no other method was tried"


def test_a_floor_makes_the_unbounded_region_unreachable_while_the_fit_runs() -> None:
    """For a nested model there is a floor the objective cannot go below: its parameters are the
    parent's with a coefficient held at zero, so every point of the child is a point of the
    parent, and the parent's maximum bounds all of them.

    Handing that bound to the objective is the difference between a fit that converges and one
    that runs for three hours and is thrown away -- the prepayment model's step 8 had reached
    0.0122 against a parent's optimum of 0.0179, which is impossible, and was still going.
    """
    from creditsurv.models.blocks import _outside_the_domain

    x = np.array([0.5, -1.0])

    # With no floor, anything non-negative is allowed through.
    assert _outside_the_domain(0.0122, x) is None
    # With the parent's optimum as the floor, the impossible value is a wall.
    refused = _outside_the_domain(0.0122, x, 0.0179)
    assert refused is not None and refused[0] == float("inf")
    np.testing.assert_array_equal(refused[1], np.zeros_like(x))
    # At or above the floor it passes, and a negative value is still refused either way.
    assert _outside_the_domain(0.0179, x, 0.0179) is None
    assert _outside_the_domain(0.02, x, 0.0179) is None
    assert _outside_the_domain(-1e-9, x, None) is not None


def test_the_floor_and_the_progress_line_belong_to_the_objective_over_every_row(
    weighted: pd.DataFrame,
) -> None:
    """Inside a pool the local objective sees **a share of the rows**, so its value is a share of
    the objective: with four processes, a quarter.

    Comparing that quarter with a floor on the whole objective refused every nested fit that was
    perfectly good -- the prepayment model's step 8 turned down a candidate at 0.076 against a
    parent's 0.071, which is exactly what a nested model should look like. And the progress line,
    being in the same place, had been reporting a quarter of the objective for every run made with
    workers.
    """
    from creditsurv.models.blocks import _Objective, _scan, _seed_regressors, _set_censoring

    names = ("lower_bound", "upper_bound", "exact_observation", "age_start", "loan_months")
    fitter = FITTERS["weibull"]()
    _set_censoring(fitter, names)
    scan = _scan(
        fitter,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        seed=_seed_regressors(fitter, FORMULA, None),
        names=names,
    )
    assert scan.columns is not None

    pooled = _Objective(
        fitter,
        scan.blocks,
        scan.columns,
        np.ones(scan.columns.size),
        lambda x: {"lambda_": x},
        floor=0.5,
        pooled=True,
    )
    alone = _Objective(
        fitter,
        scan.blocks,
        scan.columns,
        np.ones(scan.columns.size),
        lambda x: {"lambda_": x},
        floor=0.5,
        pooled=False,
    )

    assert pooled.floor is None, "a share of the rows cannot be compared with a whole objective"
    assert alone.floor == 0.5, "on its own it sees every row, so the floor is its own"


def test_an_optimiser_pinned_against_the_floor_is_given_up_on() -> None:
    """Turned back this many times in a row, the only direction the optimiser can find an
    improvement in is the impossible one -- so the maximum of the likelihood as lifelines
    computes it lies in the region lifelines cannot compute.

    That is a conclusion about the specification, and waiting for the iteration cap to confirm
    it costs hours: the prepayment model's step 8 spent 85 minutes on 17 straight refusals
    without a single accepted point, with four more hours to go. It cannot change which model is
    chosen -- a fit ending this way is refused either way, and step 8 keeps the covariate under
    rule 11 -- only how long the run waits to say what the log already shows.
    """
    from lifelines import exceptions

    from creditsurv.models.blocks import _PINNED_REFUSALS, _check_pinned, _Pinned

    def turned_back(times: int) -> _Pinned:
        pinned = _Pinned()
        for _ in range(times):
            pinned.saw(refused=True, value=float("inf"))
        return pinned

    _check_pinned(turned_back(0), 0.07)
    _check_pinned(turned_back(_PINNED_REFUSALS - 1), 0.07)
    with pytest.raises(exceptions.ConvergenceError, match="below the parent's optimum"):
        _check_pinned(turned_back(_PINNED_REFUSALS), 0.07)
    with pytest.raises(exceptions.ConvergenceError, match="outside the likelihood"):
        _check_pinned(turned_back(_PINNED_REFUSALS), None)

    # One accepted point among them resets the run, which is the hole the window fills.
    mixed = _Pinned()
    for _ in range(_PINNED_REFUSALS + 5):
        mixed.saw(refused=True, value=float("inf"))
        mixed.saw(refused=False, value=0.07)
    assert mixed.refusals == 0, "an accepted point clears the consecutive count"


def test_an_optimiser_circling_the_floor_is_given_up_on() -> None:
    """The shape a consecutive count cannot see, and the one that cost four hours.

    The prepayment model's step 8 refused six or seven points a cycle with one accepted point
    among them, so the run of refusals never passed one and the fit went to 211 evaluations
    before being refused anyway. Over a window the states separate: that fit ran at 73% refused
    across the whole of SLSQP and 88% once the cycle set in, while its productive phase ran at
    0% and no window of forty fell below 35% afterwards.

    The second condition is what keeps it honest. A search can be refused most of the time and
    still be working -- the refusals are where it probes, not where it stands -- so the guard
    also asks that nothing inside the window improved on the best point already found.
    """
    from lifelines import exceptions

    from creditsurv.models.blocks import _PINNED_SHARE, _PINNED_WINDOW, _check_pinned, _Pinned

    def cycle(cycles: int, *, improving: bool) -> _Pinned:
        """Six refused, one accepted -- the prepayment model's own cadence."""
        pinned, best = _Pinned(), 0.072
        for _turn in range(cycles):
            for _ in range(6):
                pinned.saw(refused=True, value=0.0695)
            best = best - 1e-5 if improving else best
            pinned.saw(refused=False, value=best)
        return pinned

    short = cycle(3, improving=False)
    assert not short.circling, "not a full window yet"
    _check_pinned(short, 0.07)

    # 6 refused in 7 is 86%, past the threshold, and the accepted point never improves.
    stuck = cycle(2 * _PINNED_WINDOW // 7 + 2, improving=False)
    assert stuck.refusals < 2, "the cycle keeps clearing the consecutive count"
    assert stuck.circling
    with pytest.raises(exceptions.ConvergenceError, match="circling a boundary"):
        _check_pinned(stuck, 0.07)

    # The same cadence, but each cycle finds a better point: not pinned, whatever the share.
    working = cycle(2 * _PINNED_WINDOW // 7 + 2, improving=True)
    assert not working.circling
    _check_pinned(working, 0.07)

    # And a window mostly accepted is never pinned, however little it improves.
    quiet = _Pinned()
    for _ in range(_PINNED_WINDOW * 2):
        quiet.saw(refused=False, value=0.072)
    assert sum(quiet._window) / _PINNED_WINDOW < _PINNED_SHARE
    assert not quiet.circling


def test_a_polish_closing_too_slowly_to_finish_is_given_up() -> None:
    """The prepayment model's first backward-elimination candidate closed by a constant 0.925 a
    step with the damping stuck at 1e+02, which needs 177 steps to reach a thousandth of a
    standard error against a cap of 40. Its refusals were real -- every step long enough to
    make progress landed below the parent's optimum -- but they came one to a step, each
    followed by an accepted one, so `_PINNED_REFUSALS` counted no streak and stayed silent.

    The trajectory below is that run's, step by step. The guard has to sit out the wild early
    phase, where a step can land further out than the one before, and fire in the geometric
    tail: it counts four slow steps by step 10 and stops the polish on the eleventh, where the
    cap would have taken three more hours to reach the same verdict.
    """
    from itertools import pairwise

    from creditsurv.models.blocks import _POLISH_STEPS, _STALL_STEPS, _required_ratio, _too_slow

    observed = [144.0, 40.7, 81.8, 7.56e3, 3.45e3, 1.37e3, 1.24e3, 1.13e3, 1.05e3, 975.0]
    streaks, stalled = [], 0
    for step, (previous, remaining) in enumerate(pairwise(observed), 2):
        stalled = stalled + 1 if _too_slow(remaining, previous, step) else 0
        streaks.append(stalled)

    # Steps 3 and 4 overshoot and are forgiven; 5 and 6 halve and reset the count; 7 to 10 are
    # the tail, and a fifth of them would stop the polish.
    assert streaks == [0, 1, 2, 0, 0, 1, 2, 3, 4]
    assert max(streaks) < _STALL_STEPS

    # 975 standard errors out with 30 of the 40 steps left has to close by 0.631 a step.
    assert _required_ratio(975.0, 10) == pytest.approx(0.631, abs=5e-4)
    assert _too_slow(975.0, 1050.0, 10)
    assert not _too_slow(3.45e3, 7.56e3, 5)

    # At the tolerance there is nothing left to close, and past the cap no budget to close in.
    assert not _too_slow(POLISH_TOLERANCE_SE, 1.0, 2)
    assert _required_ratio(975.0, _POLISH_STEPS) == 0.0


def test_a_polish_the_floor_stalled_is_not_offered_to_another_optimiser() -> None:
    """Told apart by what turned the steps back, because only one of the two answers is "try
    something else".

    A polish that stalls against lifelines' clipped region has failed where it stood, and another
    method from another start may well stand somewhere better -- that is why the fallbacks exist.
    A polish that stalls because every step it wants is below the **parent's optimum** has found
    a boundary that is in the same place for everybody: on the prepayment model's step 8, SLSQP
    gave up 557 standard errors out, then L-BFGS-B from its own path gave up at 762, then
    trust-constr was started, and a cold attempt would have repeated all three. Over twelve hours
    for one answer, which rule 11 gives either way.
    """
    from lifelines import exceptions

    from creditsurv.models.blocks import _PINNED_WINDOW, Pinned, _Pinned

    #: A value a likelihood can take, refused: that is the floor's doing.
    below_floor = _Pinned()
    for _ in range(_PINNED_WINDOW):
        below_floor.saw(refused=True, value=0.0695)
    assert below_floor.against_the_floor

    #: A value no likelihood can take: lifelines' clipping, and another method may escape it.
    impossible = _Pinned()
    for _ in range(_PINNED_WINDOW):
        impossible.saw(refused=True, value=float("-inf"))
    assert not impossible.against_the_floor

    #: Mostly clipping with a floor refusal among them is still clipping.
    mixed = _Pinned()
    for turn in range(_PINNED_WINDOW):
        mixed.saw(refused=True, value=0.0695 if turn % 4 == 0 else float("inf"))
    assert not mixed.against_the_floor

    #: And an accepted point is neither.
    working = _Pinned()
    for _ in range(_PINNED_WINDOW):
        working.saw(refused=False, value=0.0717)
    assert not working.against_the_floor

    #: The cadence that matters, and the one a first version got wrong: a damped Newton takes a
    #: refused step and then an accepted one, so the refusals are half of what it does. This is
    #: the prepayment model's own warm phase -- 22 evaluations, 9 refused by the floor, 2 by the
    #: clipping -- and asking for a majority of the evaluations would have wanted 12 and let the
    #: fit run three hours longer to the same answer.
    warm = _Pinned()
    for kind in [True] * 9 + [False] * 2 + [None] * 11:
        if kind is None:
            warm.saw(refused=False, value=0.0717)
        else:
            warm.saw(refused=True, value=0.0695 if kind else float("inf"))
    assert warm.against_the_floor, "9 of 11 refusals are the floor's"

    assert issubclass(Pinned, exceptions.ConvergenceError)


def test_a_worker_that_dies_ends_the_fit_instead_of_blocking_it() -> None:
    """The parent used to wait for an answer that was never coming.

    `_collect`'s `get` had no timeout, so a worker killed by the memory pressure this engine
    exists to manage left the parent blocked for ever, with nothing in the log after the last
    evaluation. There is no honest fixed deadline -- one logged fit spent 457 minutes on four
    evaluations -- so the wait polls and looks at whether the processes that owe answers are
    still alive. A missing answer from a live worker is patience; a missing answer from a
    process that has exited is the end of the fit, because its share of the rows is gone and a
    sum over the parts that remain is a different likelihood.
    """
    import queue
    from types import SimpleNamespace

    from creditsurv.models.blocks import _Workers

    pool = cast("Any", object.__new__(_Workers))
    pool._results = queue.Queue()
    pool._processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=None)]

    # Both alive and both answered: the parts come back in their own order, not in arrival
    # order, which is what keeps a pooled fit reproducible bit for bit.
    pool._results.put((2, "second"))
    pool._results.put((1, "first"))
    assert pool._collect() == ["first", "second"]

    # One alive, one exited, and the answer owed by the dead one never arrives.
    pool._processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=-9)]
    pool._results.put((1, "first"))
    with (
        pytest.raises(RuntimeError, match="stopped before answering"),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr("creditsurv.models.blocks._WORKER_POLL_SECONDS", 0.05)
        pool._collect()


def test_the_parent_evaluates_its_own_share_while_the_workers_evaluate_theirs() -> None:
    """The point goes out before the parent starts, and a failed share drains the queue.

    `_ask` put the commands and then blocked on the answers, so the workers computed while the
    parent waited and then the parent computed while the workers waited: `of` processes were
    worth `of/2`. Measured on 1.95 million rows of the production table, four workers went from
    1,747 ms an evaluation to 803, against 2,913 at one -- 3.6x of a possible 4x where it had
    been 1.75x.

    What the split costs is a hazard the single call did not have. One queue serves the whole
    pool, so an answer nobody collects is still there at the *next* evaluation, and a sum of two
    different points is worse than the failure that caused it. So a local share that raises
    drains the pool on its way out.
    """

    from creditsurv.models.blocks import _Pinned, _Pooled

    class Pool:
        def __init__(self) -> None:
            self.sent: list[tuple[str, float]] = []
            self.pending = 0
            self.discarded = 0

        def send(self, command: str, x: np.ndarray) -> None:
            self.sent.append((command, float(x[0])))
            self.pending += 1

        def value_and_gradient(self) -> list[tuple[float, np.ndarray]]:
            self.pending -= 1
            return [(0.25, np.array([1.0]))]

        def curvature(self) -> list[np.ndarray]:
            self.pending -= 1
            return [np.array([[2.0]])]

        def discard(self) -> None:
            self.pending -= 1
            self.discarded += 1

    pool = Pool()
    order: list[str] = []

    def local(x: np.ndarray) -> tuple[float, np.ndarray]:
        order.append("parent")
        if float(x[0]) < 0:
            message = "the parent's own share failed"
            raise ArithmeticError(message)
        return 0.75, np.array([3.0])

    objective = cast("Any", object.__new__(_Pooled))
    objective._local = local
    objective._workers = pool
    objective.floor = None
    objective.pinned = _Pinned()
    objective._started = 0.0
    objective._reported = np.inf
    objective.total_weight = 1.0
    objective.evaluations = 0
    objective._penalty = None

    value, gradient = objective(np.array([1.0]))

    assert pool.sent == [("value", 1.0)], "the point is sent before the parent's own share"
    assert order == ["parent"], "the parent evaluated rather than waited"
    assert value == pytest.approx(1.0), "the parent's share plus the workers'"
    np.testing.assert_allclose(gradient, [4.0])
    assert pool.pending == 0 and pool.discarded == 0

    with pytest.raises(ArithmeticError, match="own share failed"):
        objective(np.array([-1.0]))
    assert pool.pending == 0, "an uncollected answer would be read at the next point"
    assert pool.discarded == 1


def test_a_filter_that_selects_every_row_is_not_a_copy_of_the_design() -> None:
    """The common case on this panel, and it was the most expensive one.

    lifelines' interval-censored likelihood filters the design by the event flag and by its
    complement. The event flag here is `exact_observation`, which `panel.to_interval_censored`
    sets `False` with no condition -- the reporting interval tells us the month, never the day
    -- so one filter selects no rows and the other selects all of them. The second was boolean
    fancy-indexing the whole design into a new Fortran-ordered array, 50 MB on a 242,000-row
    block of the production table, at every evaluation and again at every Hessian.

    The filtered design with every row *is* the design, so it is the same slicer. Measured, a
    value-and-gradient on one such block went from 232.9 ms to 166.8.
    """
    from creditsurv.models.blocks import _Slicer

    design = np.asfortranarray(np.arange(24, dtype=float).reshape(8, 3))
    columns = pd.MultiIndex.from_tuples(
        [("lambda_", "Intercept"), ("lambda_", "credit_score"), ("rho_", "Intercept")]
    )
    slicer = _Slicer(design, columns)

    every = np.ones(8, dtype=bool)
    assert slicer.filter(every) is slicer, "no copy when nothing is excluded"
    np.testing.assert_array_equal(slicer.filter(every)["lambda_"], slicer["lambda_"])

    none = np.zeros(8, dtype=bool)
    assert slicer.filter(none).size == 0
    assert slicer.filter(none)["lambda_"].shape == (0, 2)

    # A real subset is still a copy, and still answered from the cache the second time.
    some = np.array([True, False] * 4)
    taken = slicer.filter(some)
    assert taken.size == 4
    assert slicer.filter(some) is taken
    np.testing.assert_array_equal(taken["lambda_"], design[some][:, :2])


CALENDAR = ["ltv_change", "unemployment_change"]


def test_a_fit_through_the_written_out_kernel_is_the_fit_autograd_makes(
    weighted: pd.DataFrame,
) -> None:
    """The same estimator, by two routes, on the same rows.

    One traces lifelines' likelihood with autograd over a stored design; the other reads
    `creditsurv.models.kernel`, which writes the likelihood out and holds the design as two
    tables -- one row per distinct loan combination and one per distinct calendar key -- so no
    design matrix is ever built. On the production table that is 5.1x on a value-and-gradient
    and 11.8x with the Hessian, at 0.81 GB for the whole training half.

    What is compared is what a report reads: the coefficients in units of their own standard
    errors, because that is what the engine promises and the two paths each stop within a
    thousandth of one; the standard errors themselves; and the log-likelihood, which the two
    reach by different summation orders over the same terms.
    """
    from creditsurv.models.aft import fit_aft

    traced = fit_aft(weighted, COVARIATES, FORMULA, weights_col="loan_months")
    written = fit_aft(weighted, COVARIATES, FORMULA, weights_col="loan_months", calendar=CALENDAR)

    assert traced.blocks is not None and written.blocks is not None
    assert written.n_episodes == traced.n_episodes
    assert written.n_events == traced.n_events
    assert written.blocks.loan_months == pytest.approx(traced.blocks.loan_months)

    moved = (written.fitter.params_ - traced.fitter.params_).abs() / traced.fitter.standard_errors_
    assert moved.max() < 2 * POLISH_TOLERANCE_SE, f"{moved.max()} standard errors apart"
    np.testing.assert_allclose(
        written.fitter.standard_errors_.to_numpy(),
        traced.fitter.standard_errors_.to_numpy(),
        rtol=1e-4,
    )
    assert written.log_likelihood == pytest.approx(traced.log_likelihood, rel=1e-9)

    # And the rows are fifteen bytes each: two indices, an age, an exit and a weight. The
    # compacted design is 23.5 a row on this fixture, where `StoredColumn` gets it down to
    # single-byte codes because the fixture has few distinct values; on the production table
    # it is 30, so the real ratio is two to one and here it is 1.57.
    assert written.blocks.stored_bytes == 15 * written.n_episodes
    assert written.blocks.stored_bytes < traced.blocks.stored_bytes


def test_the_kernel_refuses_a_shape_with_covariates_and_a_pool() -> None:
    """Two refusals rather than two workarounds.

    A shape with covariates is a different model -- the `occupancy` finding in
    `docs/decisions.md` is exactly that fit -- and the kernel's economy comes from a row
    depending on two scalars. And the partition of the design's columns is discovered from the
    data, so two processes scanning different shares could classify a column differently and
    index into the parameter vector in two different ways, silently; the kernel makes the
    second process unnecessary rather than making the agreement work.
    """
    from creditsurv.models.aft import fit_streamed

    with pytest.raises(ValueError, match="one process"):
        fit_streamed(
            lambda part, of: iter(()),
            COVARIATES,
            FORMULA,
            weights_col="loan_months",
            workers=2,
            calendar=CALENDAR,
        )


NARROWER = "credit_score + unemployment_change + C(purpose)"


def test_two_models_are_fitted_from_one_reading_of_the_rows(weighted: pd.DataFrame) -> None:
    """The change the branch is for: a selection's thirty fits, one scan.

    A fit through the written-out kernel on the production table is 53 seconds of arithmetic
    behind 12.1 minutes of reading, and every candidate used to pay that reading again -- the
    fifteen step-7 fits of one logged run each began by recomputing the identical base
    objective to twelve digits.

    Here the rows are read once, with no formula involved, and two different models are fitted
    from the same encoding. Each must land where a fit that read the rows for itself lands.
    """
    from creditsurv.data.panel import model_blocks
    from creditsurv.models.aft import FITTERS, fit_aft
    from creditsurv.models.blocks import encode_blocks, fit_encoded

    encoding = encode_blocks(
        model_blocks(weighted, COVARIATES, rows=5_000, weights_col="loan_months"),
        loan=[name for name in COVARIATES if name not in CALENDAR],
        calendar=CALENDAR,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
    )
    assert encoding.episodes == len(weighted)
    assert encoding.nbytes == 15 * len(weighted)

    for formula in (FORMULA, NARROWER):
        fresh = fit_aft(
            weighted,
            COVARIATES,
            formula,
            weights_col="loan_months",
            calendar=CALENDAR,
        )
        fitter = FITTERS["weibull"](penalizer=0.0)
        record = fit_encoded(fitter, encoding, formula=formula)

        assert record.rows == fresh.n_episodes, formula
        assert record.events == fresh.n_events, formula
        moved = (fitter.params_ - fresh.fitter.params_).abs() / fresh.fitter.standard_errors_
        assert moved.max() < 2 * POLISH_TOLERANCE_SE, f"{formula}: {moved.max()} se apart"
        np.testing.assert_allclose(
            fitter.standard_errors_.to_numpy(),
            fresh.fitter.standard_errors_.to_numpy(),
            rtol=1e-4,
            err_msg=formula,
        )
        assert fitter.log_likelihood_ == pytest.approx(fresh.log_likelihood, rel=1e-9), formula


def test_an_unclassified_covariate_has_no_index_to_be_looked_up_by(
    weighted: pd.DataFrame,
) -> None:
    """Between them the two keys must cover every covariate the blocks carry.

    A covariate in neither is not a slow fit but a wrong one: the encoded row would carry no
    index that distinguishes it, so two rows differing only in it would share a design row.
    """
    from creditsurv.data.panel import model_blocks
    from creditsurv.models.blocks import encode_blocks

    with pytest.raises(ValueError, match="neither in the loan key"):
        encode_blocks(
            model_blocks(weighted, COVARIATES, rows=5_000, weights_col="loan_months"),
            loan=["credit_score"],
            calendar=CALENDAR,
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            weights_col="loan_months",
        )


def test_newton_goes_first_from_a_cold_start_and_not_only_from_a_warm_one(
    weighted: pd.DataFrame,
) -> None:
    """The optimiser's long path was the price of an expensive Hessian, and it is not any more.

    A cold fit used to be fifty to a hundred and forty SLSQP evaluations, because a Hessian cost
    forty-nine times a value and a hundred of them was out of the question. The written-out
    kernel puts a Hessian at about twice a value-and-gradient, and on the production table the
    same specification from lifelines' own seed takes **10.18 minutes and 16 evaluations**
    through damped Newton against **43.79 and 142** through SLSQP, to the same log-likelihood
    of -10,691,177.6879.

    So Newton is tried first from wherever the fit starts. What the engine promises is unchanged
    and is what decides: the polish measures the distance to the optimum and refuses a fit it
    cannot drive under a thousandth of a standard error.
    """
    fitter = FITTERS["weibull"]()
    record = fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=True,
    )

    assert record.method == "newton", "no optimiser was needed"
    assert record.polish_steps > 0, "and it got there by taking steps"
    assert record.residual_error_se < POLISH_TOLERANCE_SE
    assert fitter.log_likelihood_ < 0

    # `polish=False` is the mode that reproduces lifelines exactly, so it keeps the optimiser.
    stock = FITTERS["weibull"]()
    without = fit_interval_censoring_in_blocks(
        stock,
        model_blocks(weighted, COVARIATES, rows=4_000, weights_col="loan_months"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="loan_months",
        polish=False,
    )
    assert without.method == "slsqp"
    # The two agree to a fraction of a standard error, and the polished one is the better of
    # them: lifelines' SLSQP stops on a change of 1e-10 in the *mean* log-likelihood, which
    # takes no account of how precisely the data pin a coefficient down, and on four quarters
    # of the book it stopped up to 5.9 standard errors out.
    apart = (fitter.params_ - stock.params_).abs() / fitter.standard_errors_
    assert apart.max() < 0.1, f"{apart.max()} standard errors apart"
    assert fitter.log_likelihood_ >= stock.log_likelihood_, "the polish finished the job"
    assert without.residual_error_se > record.residual_error_se
