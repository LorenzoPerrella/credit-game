"""A fit in blocks is the fit lifelines makes.

The block engine mirrors private lifelines code, so the evidence that it is equivalent
has to be a fit of the same rows both ways, compared on every number a report reads.
These tests are that evidence, and the first thing to re-run against a new lifelines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

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
from creditsurv.models.blocks import StoredColumn, fit_interval_censoring_in_blocks
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    from lifelines.fitters import ParametericAFTRegressionFitter

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap", "purpose"]
FORMULA = "fico_s + cltv_drift + unemp_gap + C(purpose)"

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.7,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
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
    encoded["n"] = np.random.default_rng(5).integers(1, 6, len(encoded))
    encoded["purpose"] = encoded["purpose"].astype("category")
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
        model_frame(frame, COVARIATES).assign(n=frame["n"].to_numpy()),
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="n",
        formula=FORMULA,
    )
    return fitter


def block_fit(
    fitter: ParametericAFTRegressionFitter, frame: pd.DataFrame, rows: int
) -> ParametericAFTRegressionFitter:
    fit_interval_censoring_in_blocks(
        fitter,
        model_blocks(frame, COVARIATES, rows=rows, weights_col="n"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="n",
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
        model_blocks(weighted, COVARIATES, rows=single_level_rows(weighted), weights_col="n"),
        formula=FORMULA,
        lower_bound_col=LOWER_BOUND,
        upper_bound_col=UPPER_BOUND,
        event_col=EXACT_OBSERVATION,
        entry_col=AGE_START,
        weights_col="n",
    )

    assert record.rows == len(weighted)
    assert record.loan_months == weighted["n"].sum()
    assert record.events == weighted.loc[weighted["event"].astype(bool), "n"].sum()
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
    first = model_frame(weighted.iloc[:rows], COVARIATES).assign(n=1)
    second = model_frame(weighted.iloc[rows:], COVARIATES).assign(n=1)
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
            weights_col="n",
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
