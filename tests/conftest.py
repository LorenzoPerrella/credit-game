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
    index = pd.period_range("1999-01", "2014-12", freq="M")
    months = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "unemployment_rate": 5.5 + 2.5 * np.sin(months / 29.0 + 1.1),
            "hpi": 100.0 * (1.0 + 0.32 * np.sin(months / 41.0)),
            "mortgage_rate_30y": 6.0 + 1.0 * np.cos(months / 33.0),
            "nfci": 0.35 * np.sin(months / 23.0 + 0.4),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def macro_module() -> pd.DataFrame:
    """Module-scoped copy of :func:`macro`, for tests that fit a model.

    A fit costs a few seconds, so the tests asserting against one share it rather
    than refitting per assertion.
    """
    index = pd.period_range("1999-01", "2014-12", freq="M")
    months = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "unemployment_rate": 5.5 + 2.5 * np.sin(months / 29.0 + 1.1),
            "hpi": 100.0 * (1.0 + 0.32 * np.sin(months / 41.0)),
            "mortgage_rate_30y": 6.0 + 1.0 * np.cos(months / 33.0),
            "nfci": 0.35 * np.sin(months / 23.0 + 0.4),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def book_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory for one module's simulated Freddie Mac files."""
    return tmp_path_factory.mktemp("book")
