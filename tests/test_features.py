"""Tests for macro covariate construction and coarse classing.

The lag tests exist because look-ahead is invisible in results: a model that
quietly reads next quarter's unemployment looks excellent and is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.config import default_formula
from creditsurv.features import (
    BIN_EDGES,
    CONTEMPORANEOUS_SERIES,
    LAGGED_SERIES,
    add_macro_covariates,
    bin_covariates,
    binned_formula,
    lag_macro,
)


def value_at(frame: pd.DataFrame, period: str, column: str) -> float:
    """Read one cell by period label.

    Boolean selection rather than ``.loc[Period, column]``: pandas-stubs does not
    model a Period as a valid label key, and a helper reads better than an ignore
    on every call site.
    """
    return float(frame.loc[frame.index == pd.Period(period, freq="M"), column].iloc[0])


@pytest.fixture
def macro() -> pd.DataFrame:
    """Distinct, strictly increasing values so a shift is unambiguous.

    Every series the covariate builder reads, and starting a year before the panel
    under test: a year-on-year change on a lagged series reaches fifteen months back,
    and a fixture that stops short of that produces an empty result rather than a
    wrong one -- which is a confusing way to find out.
    """
    index = pd.period_range("1999-01", "2001-12", freq="M")
    steps = np.arange(len(index), dtype=float)
    columns = {
        "unemployment_rate": 4.0 + steps,
        "hpi": 100.0 + steps,
        "mortgage_rate_30y": 6.0 + steps,
        "nfci": steps,
    }
    # Distinct offsets so a covariate reading the wrong series is visible, and all
    # strictly increasing so a shift in either direction shows up as a wrong value.
    for offset, name in enumerate(set(LAGGED_SERIES + CONTEMPORANEOUS_SERIES) - set(columns)):
        columns[name] = 200.0 + 10.0 * offset + steps
    for offset, name in enumerate(("policy_rate", "cpi", "sentiment", "housing_starts")):
        columns.setdefault(name, 300.0 + 10.0 * offset + steps)
    return pd.DataFrame(columns, index=index)


def test_revised_series_are_shifted_by_the_lag(macro: pd.DataFrame) -> None:
    lagged = lag_macro(macro, lag_months=3)

    # The value usable in 2000-06 is the one observed in 2000-03.
    assert value_at(lagged, "2000-06", "unemployment_rate") == value_at(
        macro, "2000-03", "unemployment_rate"
    )


def test_the_mortgage_rate_is_not_lagged(macro: pd.DataFrame) -> None:
    """A weekly market quote is known in real time and never revised."""
    lagged = lag_macro(macro, lag_months=3)

    for column in CONTEMPORANEOUS_SERIES:
        pd.testing.assert_series_equal(lagged[column], macro[column])


def test_the_lag_leaves_the_opening_months_unusable(macro: pd.DataFrame) -> None:
    """Those observations did not exist yet, so they must be missing, not filled."""
    lagged = lag_macro(macro, lag_months=3)

    for column in LAGGED_SERIES:
        assert lagged[column].head(3).isna().all()


def _panel(macro: pd.DataFrame) -> pd.DataFrame:
    periods = pd.period_range("2000-06", "2000-09", freq="M")
    return pd.DataFrame(
        {
            "loan_id": 1,
            "orig_period": pd.Period("2000-06", freq="M"),
            "period": periods,
            "age": range(len(periods)),
            "orig_ltv": 80.0,
            "note_rate": 7.0,
        }
    )


def test_covariates_are_built_from_lagged_macro(macro: pd.DataFrame) -> None:
    """A covariate read at period p must trace to an observation at p - lag."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    at_origination = enriched.iloc[0]
    # unemp_gap compares period and origination, both lagged, so it starts at zero.
    assert at_origination["unemp_gap"] == pytest.approx(0.0)
    # refi_incentive uses the contemporaneous rate: 7.0 - mortgage_rate(2000-06).
    expected = 7.0 - value_at(macro, "2000-06", "mortgage_rate_30y")
    assert at_origination["refi_incentive"] == pytest.approx(expected)


def test_cltv_drift_starts_at_zero(macro: pd.DataFrame) -> None:
    """Zero at origination by construction: the level and the movement are separable."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    assert enriched.iloc[0]["cltv_drift"] == pytest.approx(0.0)


def test_rising_prices_reduce_indexed_leverage(macro: pd.DataFrame) -> None:
    """Mark-to-market leverage must fall as the index rises, not rise with it."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    assert enriched["cltv_drift"].is_monotonic_decreasing
    assert enriched.iloc[-1]["indexed_cltv"] < 80.0


def test_rows_without_lagged_history_are_dropped_not_imputed(macro: pd.DataFrame) -> None:
    """The earliest vintages have no history to reach back to."""
    panel = pd.DataFrame(
        {
            "loan_id": [1],
            "orig_period": [pd.Period("2000-01", freq="M")],
            "period": [pd.Period("2000-01", freq="M")],
            "age": [0],
            "orig_ltv": [80.0],
            "note_rate": [7.0],
        }
    )

    assert add_macro_covariates(panel, macro, lag_months=3).empty


def test_binning_uses_band_midpoints() -> None:
    frame = pd.DataFrame({"dti": [21.0, 27.9]})

    binned = bin_covariates(frame, edges={"dti": (10.0, 20.0, 28.0, 36.0)})

    # Both values fall in (20, 28], whose midpoint is 24.
    assert binned["dti_binned"].tolist() == [24.0, 24.0]


def test_binning_clips_rather_than_drops_outliers() -> None:
    frame = pd.DataFrame({"dti": [-50.0, 900.0]})

    binned = bin_covariates(frame, edges={"dti": (10.0, 20.0, 36.0)})

    assert binned["dti_binned"].tolist() == [15.0, 28.0]
    assert not binned["dti_binned"].isna().any()


def test_binning_covers_every_modelled_continuous_covariate() -> None:
    """A covariate without cut points would silently stay continuous.

    A subset rather than an equality: cut points also exist for covariates the
    aggregation does not carry into the cell key, which the loan-level path can still
    bin. An unused set of edges is harmless; a modelled covariate without any is not.
    """
    from creditsurv.config import STATIC_CONTINUOUS, TIME_VARYING_CONTINUOUS

    assert set(STATIC_CONTINUOUS) | set(TIME_VARYING_CONTINUOUS) <= set(BIN_EDGES)


def test_binned_formula_rewrites_only_continuous_terms() -> None:
    rewritten = binned_formula(default_formula())

    assert "fico_s_binned" in rewritten
    assert "cltv_drift_binned" in rewritten
    # Categoricals are already discrete and must be left alone.
    assert "C(purpose, Treatment('purchase'))" in rewritten
    assert "purpose_binned" not in rewritten
