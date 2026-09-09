"""Tests for the canonical panel, episode splitting and interval encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import (
    PanelValidationError,
    model_frame,
    to_counting_process,
    to_interval_censored,
    to_loan_level,
    validate_episodes,
)


def make_panel(histories: dict[int, tuple[int, bool]]) -> pd.DataFrame:
    """Build a panel from ``{loan_id: (months_observed, defaulted)}``."""
    rows = []
    for loan_id, (months, defaulted) in histories.items():
        for age in range(months):
            terminal = age == months - 1
            rows.append((loan_id, age, defaulted and terminal, 0.5 * loan_id))
    return pd.DataFrame(rows, columns=["loan_id", "age", "event", "covariate"])


def test_accepts_a_well_formed_panel() -> None:
    validate_episodes(make_panel({1: (3, True), 2: (5, False)}))


def test_rejects_missing_columns() -> None:
    with pytest.raises(PanelValidationError, match="missing required column"):
        validate_episodes(pd.DataFrame({"loan_id": [1], "age": [0]}))


def test_rejects_duplicate_loan_months() -> None:
    panel = pd.concat([make_panel({1: (2, False)})] * 2, ignore_index=True)

    with pytest.raises(PanelValidationError, match="duplicated loan-month"):
        validate_episodes(panel)


def test_rejects_gaps_in_loan_age() -> None:
    """A gap silently drops exposure and biases the hazard downwards."""
    panel = make_panel({1: (4, False)})
    panel = panel[panel["age"] != 2].reset_index(drop=True)

    with pytest.raises(PanelValidationError, match="gaps or non-unit steps"):
        validate_episodes(panel)


def test_rejects_negative_age() -> None:
    panel = make_panel({1: (2, False)})
    panel.loc[0, "age"] = -1

    with pytest.raises(PanelValidationError, match="negative loan ages"):
        validate_episodes(panel)


def test_rejects_more_than_one_event_per_loan() -> None:
    panel = make_panel({1: (3, True)})
    panel.loc[0, "event"] = True

    with pytest.raises(PanelValidationError, match="more than one event"):
        validate_episodes(panel)


def test_rejects_an_event_before_the_final_month() -> None:
    """A loan cannot keep paying after it has defaulted."""
    panel = make_panel({1: (4, False)})
    panel.loc[1, "event"] = True

    with pytest.raises(PanelValidationError, match="before their final month"):
        validate_episodes(panel)


def test_counting_process_bounds_tile_the_history() -> None:
    episodes = to_counting_process(make_panel({1: (3, False)}))

    assert episodes["age_start"].tolist() == [0.0, 1.0, 2.0]
    assert episodes["age_stop"].tolist() == [1.0, 2.0, 3.0]


def test_surviving_months_are_encoded_as_right_censored() -> None:
    """Survival contributes log S(a+1)/S(a): lower at a+1, upper unbounded."""
    encoded = to_interval_censored(make_panel({1: (2, False)}))

    assert encoded["lower_bound"].tolist() == [1.0, 2.0]
    assert np.isinf(encoded["upper_bound"]).all()
    assert encoded["age_start"].tolist() == [0.0, 1.0]


def test_the_defaulting_month_is_encoded_as_interval_censored() -> None:
    """The default month is known, the day is not: bounds bracket that month."""
    encoded = to_interval_censored(make_panel({1: (3, True)}))
    terminal = encoded.iloc[-1]

    assert terminal["lower_bound"] == 2.0
    assert terminal["upper_bound"] == 3.0
    assert terminal["age_start"] == 2.0


def test_encoding_satisfies_the_lifelines_contract() -> None:
    """The three invariants lifelines enforces on interval-censored input.

    Violating any of them raises inside the fitter with a message far from the
    cause, so they are asserted here instead.
    """
    encoded = to_interval_censored(make_panel({1: (4, True), 2: (6, False), 3: (1, True)}))

    lower = encoded["lower_bound"].to_numpy()
    upper = encoded["upper_bound"].to_numpy()
    entry = encoded["age_start"].to_numpy()
    exact = encoded["exact_observation"].to_numpy()

    assert (lower <= upper).all()
    assert (entry < upper).all()
    # lower == upper if and only if the event time is exact, which it never is.
    assert ((lower == upper) == exact).all()
    assert not exact.any()


def test_model_frame_excludes_unmodelled_columns() -> None:
    """Identifiers must not reach the design matrix."""
    encoded = to_interval_censored(make_panel({1: (2, False)}))

    frame = model_frame(encoded, ["covariate"])

    assert "loan_id" not in frame.columns
    assert set(frame.columns) == {
        "covariate",
        "age_start",
        "lower_bound",
        "upper_bound",
        "exact_observation",
    }


def test_model_frame_rejects_an_unknown_covariate() -> None:
    encoded = to_interval_censored(make_panel({1: (2, False)}))

    with pytest.raises(PanelValidationError, match="missing column"):
        model_frame(encoded, ["not_a_column"])


def test_loan_level_collapses_to_duration_and_status() -> None:
    loans = to_loan_level(make_panel({1: (3, True), 2: (5, False)})).set_index("loan_id")

    assert loans.loc[1, "duration"] == 3.0
    assert bool(loans.loc[1, "event"]) is True
    assert loans.loc[2, "duration"] == 5.0
    assert bool(loans.loc[2, "event"]) is False
