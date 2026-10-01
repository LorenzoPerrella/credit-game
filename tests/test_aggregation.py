"""Grouped estimation must be lossless.

The point of these tests is not the speed-up -- which is modest, and absent
entirely at realistic covariate counts -- but the demonstration that collapsing
identical episodes into weighted cells leaves the fit unchanged. An aggregation
that quietly altered the likelihood would be far worse than no aggregation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.data.panel import (
    PanelValidationError,
    aggregate_episodes,
    to_interval_censored,
)
from creditsurv.features import BINNED_SUFFIX, bin_covariates
from creditsurv.models.aft import fit_aft
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    from creditsurv.models.aft import FitResult

RAW_COVARIATES = ("credit_score", "ltv_change", "unemployment_change")
COVARIATES = [name + BINNED_SUFFIX for name in RAW_COVARIATES]
FORMULA = " + ".join(COVARIATES)

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def encoded(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    panel, _ = build_panel(book_dir, macro_module, n_loans=1200, seed=41, params=PARAMS)
    return bin_covariates(to_interval_censored(panel))


@pytest.fixture(scope="module")
def grouped(encoded: pd.DataFrame) -> pd.DataFrame:
    return aggregate_episodes(encoded, COVARIATES)


@pytest.fixture(scope="module")
def fits(encoded: pd.DataFrame, grouped: pd.DataFrame) -> tuple[FitResult, FitResult]:
    return (
        fit_aft(encoded, COVARIATES, FORMULA),
        fit_aft(grouped, COVARIATES, FORMULA, weights_col="loan_months"),
    )


def test_weights_account_for_every_episode(encoded: pd.DataFrame, grouped: pd.DataFrame) -> None:
    """Nothing may be dropped or double-counted by the collapse."""
    assert int(grouped["loan_months"].sum()) == len(encoded)


def test_weights_are_integer_counts(grouped: pd.DataFrame) -> None:
    assert grouped["loan_months"].dtype == "int64"
    assert (grouped["loan_months"] > 0).all()


def test_binning_makes_the_collapse_worthwhile(
    encoded: pd.DataFrame, grouped: pd.DataFrame
) -> None:
    """Only true in low dimension; see docs/methodology.md for where it stops."""
    assert len(grouped) < len(encoded) / 5


def test_grouped_and_ungrouped_coefficients_agree(
    fits: tuple[FitResult, FitResult],
) -> None:
    """The claim the aggregation rests on."""
    ungrouped, weighted = fits

    difference = (ungrouped.fitter.params_ - weighted.fitter.params_).abs().max()
    assert difference < 1e-3


def test_grouped_and_ungrouped_log_likelihoods_agree(
    fits: tuple[FitResult, FitResult],
) -> None:
    """A frequency weight replicates a row, so the total must be identical."""
    ungrouped, weighted = fits

    assert weighted.log_likelihood == pytest.approx(ungrouped.log_likelihood, rel=1e-6)


def test_grouped_fit_counts_the_same_events(fits: tuple[FitResult, FitResult]) -> None:
    ungrouped, weighted = fits

    assert weighted.n_events == ungrouped.n_events


def test_exposure_weights_are_rejected(grouped: pd.DataFrame) -> None:
    """Weighting by amount answers a different question and breaks inference."""
    with_exposure = grouped.copy()
    with_exposure["exposure"] = grouped["loan_months"].astype(float) * 1234.56

    with pytest.raises(ValueError, match="frequency weights"):
        fit_aft(with_exposure, COVARIATES, FORMULA, weights_col="exposure")


def test_non_positive_weights_are_rejected(grouped: pd.DataFrame) -> None:
    broken = grouped.copy()
    broken.loc[broken.index[0], "loan_months"] = 0

    with pytest.raises(ValueError, match="non-positive"):
        fit_aft(broken, COVARIATES, FORMULA, weights_col="loan_months")


def test_aggregation_requires_the_encoded_columns() -> None:
    with pytest.raises(PanelValidationError, match="missing column"):
        aggregate_episodes(pd.DataFrame({"credit_score_binned": [1.0]}), COVARIATES)
