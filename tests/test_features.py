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
        "house_price_index": 100.0 + steps,
        "mortgage_rate_30y": 6.0 + steps,
        "financial_conditions_index": steps,
    }
    # Distinct offsets so a covariate reading the wrong series is visible, and all
    # strictly increasing so a shift in either direction shows up as a wrong value.
    for offset, name in enumerate(set(LAGGED_SERIES + CONTEMPORANEOUS_SERIES) - set(columns)):
        columns[name] = 200.0 + 10.0 * offset + steps
    for offset, name in enumerate(
        ("fed_funds_rate", "consumer_price_index", "consumer_sentiment_index", "housing_starts")
    ):
        columns.setdefault(name, 300.0 + 10.0 * offset + steps)
    return pd.DataFrame(columns, index=index)


def test_revised_series_are_shifted_by_the_lag(macro: pd.DataFrame) -> None:
    lagged = lag_macro(macro, lag_months=3)

    # The value usable in 2000-06 is the one observed in 2000-03.
    assert value_at(lagged, "2000-06", "unemployment_rate") == value_at(
        macro, "2000-03", "unemployment_rate"
    )


def test_no_series_is_read_in_the_month_it_is_quoted(macro: pd.DataFrame) -> None:
    """A market quote is known in real time, and still cannot cause a default that month.

    A loan is 90+ days delinquent in month t because payments were missed in t-3 to t-1, so no
    reading from month t can be what caused it. ``equity_volatility`` was read contemporaneously,
    and the backtest's predicted default spiked on the two VIX peaks of the test window while
    realised default did not move. The lag reaches every series, not only the one that was noticed.
    """
    lagged = lag_macro(macro, lag_months=3)

    assert CONTEMPORANEOUS_SERIES == ()
    for column in ("vix_index", "baa_treasury_spread", "mortgage_rate_30y", "fed_funds_rate"):
        assert value_at(lagged, "2000-06", column) == value_at(macro, "2000-03", column)


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
            "origination_period": pd.Period("2000-06", freq="M"),
            "period": periods,
            "age": range(len(periods)),
            "original_ltv": 80.0,
            "note_rate": 7.0,
        }
    )


def test_covariates_are_built_from_lagged_macro(macro: pd.DataFrame) -> None:
    """A covariate read at period p must trace to an observation at p - lag."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    at_origination = enriched.iloc[0]
    # unemployment_change compares period and origination, both lagged, so it starts at zero.
    assert at_origination["unemployment_change"] == pytest.approx(0.0)
    # refinance_incentive uses the contemporaneous rate: 7.0 - mortgage_rate(2000-06).
    expected = 7.0 - value_at(macro, "2000-06", "mortgage_rate_30y")
    assert at_origination["refinance_incentive"] == pytest.approx(expected)


def test_cltv_drift_starts_at_zero(macro: pd.DataFrame) -> None:
    """Zero at origination by construction: the level and the movement are separable."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    assert enriched.iloc[0]["ltv_change"] == pytest.approx(0.0)


def test_rising_prices_reduce_indexed_leverage(macro: pd.DataFrame) -> None:
    """Mark-to-market leverage must fall as the index rises, not rise with it."""
    enriched = add_macro_covariates(_panel(macro), macro, lag_months=3)

    assert enriched["ltv_change"].is_monotonic_decreasing
    assert enriched.iloc[-1]["indexed_cltv"] < 80.0


def test_rows_without_lagged_history_are_dropped_not_imputed(macro: pd.DataFrame) -> None:
    """The earliest vintages have no history to reach back to."""
    panel = pd.DataFrame(
        {
            "loan_id": [1],
            "origination_period": [pd.Period("2000-01", freq="M")],
            "period": [pd.Period("2000-01", freq="M")],
            "age": [0],
            "original_ltv": [80.0],
            "note_rate": [7.0],
        }
    )

    assert add_macro_covariates(panel, macro, lag_months=3).empty


def test_binning_uses_band_midpoints() -> None:
    frame = pd.DataFrame({"debt_to_income": [21.0, 27.9]})

    binned = bin_covariates(frame, edges={"debt_to_income": (10.0, 20.0, 28.0, 36.0)})

    # Both values fall in (20, 28], whose midpoint is 24.
    assert binned["debt_to_income_binned"].tolist() == [24.0, 24.0]


def test_binning_clips_rather_than_drops_outliers() -> None:
    frame = pd.DataFrame({"debt_to_income": [-50.0, 900.0]})

    binned = bin_covariates(frame, edges={"debt_to_income": (10.0, 20.0, 36.0)})

    assert binned["debt_to_income_binned"].tolist() == [15.0, 28.0]
    assert not binned["debt_to_income_binned"].isna().any()


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

    assert "credit_score_binned" in rewritten
    assert "ltv_change_binned" in rewritten
    # Categoricals are already discrete and must be left alone.
    assert "C(purpose, Treatment('purchase'))" in rewritten
    assert "purpose_binned" not in rewritten


# --------------------------------------------------------------------------------------
# The loan's own rate, and the ratio a HARP loan does not report
# --------------------------------------------------------------------------------------


def test_the_spread_and_the_incentive_are_the_note_rate_against_the_market_rate(
    macro: pd.DataFrame,
) -> None:
    """Both were out of reach until the note rate entered the cell key. The spread is
    priced at origination and never moves; the incentive is the same comparison made now,
    which is what a prepayment model turns on.
    """
    from creditsurv.features import add_macro_family

    origination = pd.Series([2000 * 12 + 5] * 3)
    ages = pd.Series([0, 6, 12])
    episodes = pd.DataFrame({"note_rate": [6.5, 6.5, 6.5], "term_years": [30, 30, 30]})

    add_macro_family(episodes, macro, origination, origination + ages, lag_months=3)

    at_origination = value_at(macro.shift(3), "2000-06", "mortgage_rate_30y")
    assert episodes["origination_spread"].tolist() == pytest.approx([6.5 - at_origination] * 3)
    # The identity the three rate covariates obey: incentive = spread + the fall since.
    rebuilt = episodes["origination_spread"] + episodes["mortgage_rate_decline"]
    assert episodes["refinance_incentive"].tolist() == pytest.approx(rebuilt.tolist())


def test_the_benchmark_follows_the_term_as_the_refinancing_rate_does(macro: pd.DataFrame) -> None:
    """A fifteen-year loan is refinanced against the fifteen-year rate, which is only
    possible because the term is in the cell key.
    """
    from creditsurv.features import add_macro_family

    origination = pd.Series([2000 * 12 + 5] * 2)
    episodes = pd.DataFrame({"note_rate": [6.5, 6.5], "term_years": [15, 30]})

    add_macro_family(episodes, macro, origination, origination + 12, lag_months=3)

    short, long = episodes["refinance_incentive"].tolist()
    assert short != pytest.approx(long), "both terms read the same benchmark"
    assert short == pytest.approx(6.5 - value_at(macro.shift(3), "2001-06", "mortgage_rate_15y"))
    assert long == pytest.approx(6.5 - value_at(macro.shift(3), "2001-06", "mortgage_rate_30y"))


def test_a_ratio_a_harp_loan_does_not_report_is_absorbed_by_the_harp_level() -> None:
    """The dummy-variable adjustment, and the reason it is not an imputation: the fill is
    a constant on every row carrying the level, so the level's coefficient takes all of it
    and the slope is estimated on the loans that report the ratio.
    """
    from creditsurv.features import NOT_REPORTED, absorb_not_reported

    fill = NOT_REPORTED["debt_to_income"][2]
    frame = pd.DataFrame(
        {
            "debt_to_income": [32.0, np.nan, 41.0],
            "harp": pd.Categorical(["standard", "harp", "standard"]),
        }
    )

    absorb_not_reported(frame, ["debt_to_income", "harp"])

    assert frame["debt_to_income"].tolist() == [32.0, fill, 41.0]


def test_a_missing_ratio_nothing_explains_is_refused_rather_than_filled() -> None:
    from creditsurv.features import absorb_not_reported

    unexplained = pd.DataFrame(
        {"debt_to_income": [32.0, np.nan], "harp": pd.Categorical(["standard", "standard"])}
    )
    with pytest.raises(ValueError, match="do not report debt_to_income and are not harp"):
        absorb_not_reported(unexplained, ["debt_to_income", "harp"])

    no_level = pd.DataFrame({"debt_to_income": [32.0, np.nan]})
    with pytest.raises(ValueError, match="does not carry harp"):
        absorb_not_reported(no_level, ["debt_to_income", "harp"])


def test_the_ratio_cannot_be_fitted_without_the_level_that_absorbs_its_fill() -> None:
    """A model reading the filled ratio without the HARP level reads a constant of 32 as a
    real debt-to-income for 18% of a decade of vintages. That is the imputation the whole
    arrangement exists to avoid, so it raises.
    """
    from creditsurv.features import absorb_not_reported

    frame = pd.DataFrame(
        {
            "debt_to_income": [32.0, np.nan],
            "harp": pd.Categorical(["standard", "harp"]),
        }
    )

    with pytest.raises(ValueError, match="fitted without harp"):
        absorb_not_reported(frame, ["debt_to_income"])
