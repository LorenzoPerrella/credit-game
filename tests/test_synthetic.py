"""Tests for the synthetic generator.

A stub macro frame is used rather than live FRED data so these run offline. The
generator's contract is structural: whatever it emits must satisfy every panel
invariant, because the whole pipeline downstream assumes it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditsurv.data.panel import to_interval_censored, validate_episodes
from creditsurv.data.synthetic import build_synthetic_panel


def test_generated_panel_satisfies_every_invariant(macro: pd.DataFrame) -> None:
    panel, _ = build_synthetic_panel(macro, n_loans=300, seed=3)

    validate_episodes(panel)


def test_every_loan_is_observed_from_origination(macro: pd.DataFrame) -> None:
    """Ages must start at zero: a loan silently starting later would be a
    selection effect the likelihood is not told about."""
    panel, _ = build_synthetic_panel(macro, n_loans=300, seed=3)

    assert panel.groupby("loan_id")["age"].min().eq(0).all()


def test_cltv_drift_is_zero_at_origination(macro: pd.DataFrame) -> None:
    """By construction: at origination the indexed ratio is the original one."""
    panel, _ = build_synthetic_panel(macro, n_loans=200, seed=4)

    at_origination = panel.loc[panel["age"] == 0, "cltv_drift"]
    assert np.allclose(at_origination.to_numpy(), 0.0)


def test_covariates_are_finite(macro: pd.DataFrame) -> None:
    panel, _ = build_synthetic_panel(macro, n_loans=200, seed=4)

    covariates = ["cltv_drift", "unemp_gap", "refi_incentive", "nfci_lagged", "fico_s"]
    assert np.isfinite(panel[covariates].to_numpy()).all()


def test_default_and_prepayment_are_mutually_exclusive(macro: pd.DataFrame) -> None:
    panel, _ = build_synthetic_panel(macro, n_loans=400, seed=5)

    assert not (panel["event"] & panel["prepaid"]).any()


def test_generation_is_reproducible(macro: pd.DataFrame) -> None:
    first, _ = build_synthetic_panel(macro, n_loans=150, seed=11)
    second, _ = build_synthetic_panel(macro, n_loans=150, seed=11)

    pd.testing.assert_frame_equal(first, second)


def test_generated_panel_encodes_cleanly(macro: pd.DataFrame) -> None:
    """The generator's output must satisfy the lifelines interval contract."""
    panel, _ = build_synthetic_panel(macro, n_loans=250, seed=6)

    encoded = to_interval_censored(panel)

    assert (encoded["age_start"] < encoded["upper_bound"]).all()
    assert (encoded["lower_bound"] <= encoded["upper_bound"]).all()
    assert not encoded["exact_observation"].any()
