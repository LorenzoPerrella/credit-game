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
    weighted: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
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
            failed = real(*args, **{**kwargs, "options": {"maxiter": 1}})
            failed.success = False
            failed.message = "Rank-deficient equality constraint subproblem HFTI"
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
    assert fitter.log_likelihood_ < 0, "a real fit came out of the fallback"
    assert record.residual_error_se < 1e-3, "and the polish certifies it is at the optimum"
