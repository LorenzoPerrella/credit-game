"""Tests for the exploratory statistics.

Every one of these is exposure-weighted, and the tests are built so that an
unweighted implementation would fail them. That is the whole risk with an aggregated
panel: a cell holding six loan-months and one holding sixty thousand look identical
until the weight is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.explore import (
    collinear_pairs,
    concentration_report,
    curves_cross,
    default_rate_by_band,
    fill_rate,
    frequency_table,
    survival_by_stratum,
    weighted_correlation,
)


def cell(frame: pd.DataFrame, row: object, column: str) -> float:
    """Read one numeric cell by label.

    pandas-stubs cannot narrow a label lookup to a number, so the chained access it
    invites needs an ignore on every line. One helper reads better.
    """
    return float(frame.loc[frame.index == row, column].to_numpy(dtype=float)[0])


def flag(frame: pd.DataFrame, row: object, column: str) -> bool:
    return bool(frame.loc[frame.index == row, column].to_numpy(dtype=bool)[0])


def test_fill_rate_flags_a_mostly_empty_column() -> None:
    frame = pd.DataFrame(
        {"present": range(100), "absent": [None] * 100, "patchy": [1] * 50 + [None] * 50}
    )

    table = fill_rate(frame).set_index("column")

    assert str(table.loc[table.index == "absent", "verdict"].iloc[0]) == "drop: mostly missing"
    assert str(table.loc[table.index == "patchy", "verdict"].iloc[0]) == "usable"
    assert cell(table, "present", "filled") == 1.0


def test_frequency_table_is_weighted_by_exposure() -> None:
    """One cell of sixty thousand loan-months outweighs ten cells of six."""
    frame = pd.DataFrame({"grade": ["A", "B"], "n": [60000, 60]})

    table = frequency_table(frame, "grade").set_index("level")

    assert cell(table, "A", "share") == pytest.approx(60000 / 60060)
    assert cell(table, "A", "share") > 0.99


def test_frequency_table_always_shows_a_missing_row() -> None:
    """A table that omits it leaves the reader unable to tell "none missing" from
    "not checked"."""
    frame = pd.DataFrame({"grade": ["A", "B"], "n": [10, 10]})

    table = frequency_table(frame, "grade")

    assert "<missing>" in set(table["level"])
    assert float(table.loc[table["level"] == "<missing>", "loan_months"].iloc[0]) == 0.0


def test_concentration_uses_ninety_nine_percent_not_ninety() -> None:
    """A covariate that is 87% one level still has 13% of a very large book saying
    something. Degenerate means there is nothing left to estimate from."""
    frame = pd.DataFrame(
        {
            "lopsided": ["owner", "investor"],
            "degenerate": ["yes", "no"],
            "n": [870, 130],
        }
    )
    degenerate = pd.DataFrame({"degenerate": ["yes", "no"], "n": [9950, 50]})

    lopsided_report = concentration_report(frame, ["lopsided"]).set_index("column")
    degenerate_report = concentration_report(degenerate, ["degenerate"]).set_index("column")

    assert not flag(lopsided_report, "lopsided", "degenerate")
    assert flag(degenerate_report, "degenerate", "degenerate")


def test_default_rate_is_events_over_exposure() -> None:
    """Two bands can hold the same defaults and differ entirely in risk if one was
    watched ten times as long."""
    frame = pd.DataFrame(
        {
            "band": ["low", "low", "high", "high"],
            "event": [True, False, True, False],
            "n": [10, 9990, 10, 990],
        }
    )

    table = default_rate_by_band(frame, "band").set_index("level")

    assert cell(table, "low", "default_rate") == pytest.approx(10 / 10000)
    assert cell(table, "high", "default_rate") == pytest.approx(10 / 1000)
    assert cell(table, "high", "default_rate") > cell(table, "low", "default_rate")


def test_correlation_is_weighted() -> None:
    """An unweighted matrix describes the distribution of cells, which is an artefact
    of the binning, rather than of loan-months, which is the data."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "n": [1, 1, 1000]})

    correlation = weighted_correlation(frame, ["x", "y"])

    assert cell(correlation, "x", "y") == pytest.approx(1.0)
    assert correlation.shape == (2, 2)


def test_correlation_weighting_changes_the_answer() -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, -9.0], "n": [1000, 1000, 1000, 1]}
    )

    weighted = cell(weighted_correlation(frame, ["x", "y"]), "x", "y")
    unweighted = cell(frame[["x", "y"]].corr(), "x", "y")

    assert weighted > 0.9
    assert unweighted < weighted, "the outlier cell should dominate only when unweighted"


def test_collinear_pairs_reports_strongest_first() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.95, 0.1], [0.95, 1.0, 0.85], [0.1, 0.85, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    pairs = collinear_pairs(correlation, threshold=0.8)

    assert len(pairs) == 2
    assert (str(pairs.iloc[0]["first"]), str(pairs.iloc[0]["second"])) == ("a", "b")


def test_no_collinear_pairs_returns_an_empty_table() -> None:
    correlation = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["a", "b"])

    assert collinear_pairs(correlation).empty


def test_survival_by_stratum_separates_a_riskier_group() -> None:
    frame = pd.DataFrame(
        {
            "grade": ["good"] * 6 + ["bad"] * 6,
            "age": [0, 0, 6, 6, 12, 12] * 2,
            "event": [True, False] * 6,
            "n": [1, 999, 1, 999, 1, 999, 50, 950, 50, 950, 50, 950],
        }
    )

    curves = survival_by_stratum(frame, "grade")
    final = curves.groupby("stratum")["survival"].min()

    assert cell(final.to_frame(), "bad", "survival") < cell(final.to_frame(), "good", "survival")
    assert not curves_cross(curves)


def test_crossing_curves_are_detected() -> None:
    """Curves that cross cannot be reconciled by scaling one into the other, which is
    what a single model with covariates assumes."""
    curves = pd.DataFrame(
        {
            "stratum": ["a", "a", "b", "b"],
            "age": [0, 6, 0, 6],
            "survival": [0.9, 0.4, 0.8, 0.6],
        }
    )

    assert curves_cross(curves)


def test_a_single_stratum_cannot_cross() -> None:
    curves = pd.DataFrame({"stratum": ["a", "a"], "age": [0, 6], "survival": [0.9, 0.5]})

    assert not curves_cross(curves)


def test_survival_is_a_decreasing_probability() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "grade": ["x"] * 20,
            "age": np.repeat(np.arange(10), 2),
            "event": [True, False] * 10,
            "n": rng.integers(50, 500, 20),
        }
    )

    curves = survival_by_stratum(frame, "grade")

    assert curves["survival"].is_monotonic_decreasing
    assert ((curves["survival"] >= 0) & (curves["survival"] <= 1)).all()
