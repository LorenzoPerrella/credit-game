"""Shared fixtures.

The macro fixture is a stub rather than live FRED data so the suite runs offline
and deterministically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _stub_macro() -> pd.DataFrame:
    """Every series the macro family reads, on a shared set of cycles.

    The full set rather than the four the model ends up using: the covariate builder
    is meant to be exercised whole, and a fixture carrying only what today's formula
    needs stops testing the moment the formula changes.

    The panel opens two years before the loans, for the reason ``MACRO_START`` gives:
    a year-on-year change on a lagged series reaches fifteen months back, and rows it
    cannot be built for are dropped rather than imputed.
    """
    index = pd.period_range("1997-01", "2014-12", freq="M")
    months = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "unemployment_rate": 5.5 + 2.5 * np.sin(months / 29.0 + 1.1),
            # The cycle matters: under a monotonically rising index cltv_drift becomes
            # a near-deterministic function of loan age, which is the model's time
            # scale, so the covariate is absorbed by the baseline hazard.
            "hpi": 100.0 * (1.0 + 0.32 * np.sin(months / 41.0)),
            "mortgage_rate_30y": 6.0 + 1.0 * np.cos(months / 33.0),
            "mortgage_rate_15y": 5.4 + 0.9 * np.cos(months / 33.0),
            "nfci": 0.35 * np.sin(months / 23.0 + 0.4),
            "policy_rate": 2.5 + 2.0 * np.cos(months / 37.0),
            "treasury_10y": 4.2 + 1.1 * np.cos(months / 44.0),
            "term_spread": 1.0 + 0.9 * np.sin(months / 31.0),
            "credit_spread": 2.4 + 0.8 * np.sin(months / 27.0 + 0.9),
            "cpi": 160.0 * np.exp(months / 520.0),
            "equity_index": 2000.0 * (1.0 + 0.45 * np.sin(months / 47.0 + 0.3)),
            "vix": 20.0 + 7.0 * np.sin(months / 19.0 + 2.0),
            "sentiment": 85.0 + 12.0 * np.sin(months / 35.0),
            "housing_starts": 1400.0 * (1.0 + 0.30 * np.sin(months / 43.0 + 0.6)),
        },
        index=index,
    )


@pytest.fixture
def macro() -> pd.DataFrame:
    """A stub macro panel with a genuine house-price cycle.

    The cycle matters. Under a monotonically rising index, ``cltv_drift`` becomes
    a near-deterministic function of loan age, and loan age is the model's time
    scale -- so the covariate is absorbed by the baseline hazard and cannot be
    identified. A real index rises and falls, which means two loans at the same
    age but different vintages face different leverage. That contrast is what
    identifies the coefficient, so the stub has to reproduce it.
    """
    return _stub_macro()


@pytest.fixture(scope="module")
def macro_module() -> pd.DataFrame:
    """Module-scoped copy of :func:`macro`, for tests that fit a model.

    A fit costs a few seconds, so the tests asserting against one share it rather
    than refitting per assertion.
    """
    return _stub_macro()


@pytest.fixture(scope="module")
def book_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory for one module's simulated Freddie Mac files."""
    return tmp_path_factory.mktemp("book")
