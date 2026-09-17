"""The moratorium comparison, from two fits on the fixture book.

The fixture has no forbearance to exclude or censor, so the second outcome stands in with a
shorter window; what is tested is that the report sets the two policies side by side and
measures the coefficient moves in standard errors, not what the moves are.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.backtest.runner import backtest_split
from creditsurv.backtest.splits import cell_split
from creditsurv.data.panel import to_interval_censored
from creditsurv.models.aft import fit_aft
from creditsurv.reporting.moratorium import Outcome, coefficient_shift, generate, outcome
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = " + ".join(COVARIATES)
PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.9,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


def _outcome(policy: str, encoded: pd.DataFrame, as_of: pd.Period) -> Outcome:
    split = cell_split(encoded, as_of)
    fitted = fit_aft(split.train, COVARIATES, FORMULA, weights_col="n")
    _, result = backtest_split(split, COVARIATES, FORMULA, fitted=fitted)
    return outcome(policy, split, fitted, result)


@pytest.fixture(scope="module")
def outcomes(book_dir: Path, macro_module: pd.DataFrame) -> tuple[Outcome, Outcome]:
    panel, _ = build_panel(book_dir, macro_module, n_loans=900, seed=41, params=PARAMS)
    encoded = to_interval_censored(panel).assign(n=1)
    return (
        _outcome("exclude", encoded, pd.Period("2008-12", freq="M")),
        _outcome("censor", encoded, pd.Period("2008-06", freq="M")),
    )


def test_the_report_sets_both_policies_side_by_side(
    outcomes: tuple[Outcome, Outcome], tmp_path: Path
) -> None:
    exclude, censor = outcomes

    text = generate(exclude, censor, reports_dir=tmp_path).read_text()

    assert "exclude" in text
    assert "censor" in text
    assert "combined standard errors" in text


def test_a_coefficient_that_does_not_move_moves_by_zero_standard_errors(
    outcomes: tuple[Outcome, Outcome],
) -> None:
    exclude, censor = outcomes

    assert (coefficient_shift(exclude, replace(exclude, policy="censor"))["moved"] == 0).all()
    assert coefficient_shift(exclude, censor)["in standard errors"].abs().max() > 0
