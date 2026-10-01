"""Anchoring the level of the model, and nothing else about it.

A survival model can rank loans well and sit at the wrong height. The two failings are
independent, they have different causes, and only one of them can be corrected after the
fit without estimating a different model -- so the correction has to be the weakest one
that fixes a level and has to be visible as a separate object rather than folded into
coefficients nobody can then read.

That is one multiplier on the default hazard:

    k = actual defaults / expected defaults, over the anchoring window

applied to every loan-month. Not the coefficients, not the shape of the hazard, not the
ranking: multiplying every hazard by the same number leaves the order of loans exactly as
it was, which is why discrimination is unchanged by construction and why the segment views
still mean something. A per-segment adjustment would fix the level of every segment and in
doing so would absorb the model's errors into the very cuts the model is examined through.

**The window is the anchoring window and nothing else.** 2022-01 to 2024-12, fixed in
`docs/rules.md` before any of these fits: the development window ends in 2021-12 and the
test window begins in 2025-01, so the level is set on months the coefficients never saw and
scored on months the level never saw. Anchoring on the development window would re-fit the
intercept badly; anchoring on the test window would be marking one's own homework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

#: The months the level is anchored on, inclusive. From `docs/rules.md`.
ANCHOR_WINDOW: Final[tuple[str, str]] = ("2022-01", "2024-12")

#: The largest multiplier that is an adjustment rather than a different model. A level this
#: far out is not a level to be scaled: it says the specification is wrong, and quietly
#: multiplying by four would hide that behind a number that then looks calibrated.
MAX_MULTIPLIER: Final = 4.0


@dataclass(frozen=True)
class Anchor:
    """The one number the level is corrected by, with what produced it."""

    multiplier: float
    window: tuple[str, str]
    actual_defaults: float
    expected_defaults: float
    loan_months: float

    def describe(self) -> dict[str, object]:
        return {
            "window": f"{self.window[0]} to {self.window[1]}",
            "loan_months": int(self.loan_months),
            "actual_defaults": round(self.actual_defaults, 1),
            "expected_defaults": round(self.expected_defaults, 1),
            "multiplier": round(self.multiplier, 4),
        }

    def apply(self, hazard: np.ndarray | pd.Series) -> np.ndarray:
        """This anchor's hazard: every rate scaled by the same number.

        Clipped below one, because a hazard is a probability and a multiplier is not
        required to respect that on its own. At the rates this book runs -- a few basis
        points a month -- the clip never binds, and it is here so that it cannot bind
        silently somewhere else.
        """
        scaled: np.ndarray = np.clip(np.asarray(hazard, dtype=float) * self.multiplier, 0.0, 1.0)
        return scaled


def anchor_on_window(
    hazard: np.ndarray | pd.Series,
    events: np.ndarray | pd.Series,
    weight: np.ndarray | pd.Series,
    periods: pd.PeriodIndex | pd.Series,
    *,
    window: tuple[str, str] = ANCHOR_WINDOW,
) -> Anchor:
    """The multiplier that puts actual over expected at one on the anchoring window.

    Everything is exposure-weighted, as every statistic on cells is: a cell stands for a
    number of loan-months, and an unweighted ratio would describe the binning rather than
    the book. That mistake has already been made once here, in the decile table of the
    backtest, where a plain mean of cell hazards put the riskiest decile at 0.929 where it
    is 1.064.
    """
    import pandas as pd

    months = pd.PeriodIndex(periods)
    first, last = pd.Period(window[0], freq="M"), pd.Period(window[1], freq="M")
    inside = (months >= first) & (months <= last)
    if not inside.any():
        message = f"No exposure in the anchoring window {window[0]} to {window[1]}."
        raise ValueError(message)

    weights = np.asarray(weight, dtype=float)[inside]
    expected = float((np.asarray(hazard, dtype=float)[inside] * weights).sum())
    actual = float((np.asarray(events, dtype=float)[inside] * weights).sum())
    if expected <= 0:
        message = "The model expects no defaults in the anchoring window; nothing to anchor."
        raise ValueError(message)

    multiplier = actual / expected
    if multiplier > MAX_MULTIPLIER or multiplier < 1.0 / MAX_MULTIPLIER:
        message = (
            f"The level is out by {multiplier:.2f}x on {window[0]} to {window[1]}, past the "
            f"{MAX_MULTIPLIER:g}x a multiplier may carry. A gap that size is a specification "
            "to be looked at, not a level to be scaled."
        )
        raise ValueError(message)
    return Anchor(
        multiplier=multiplier,
        window=window,
        actual_defaults=actual,
        expected_defaults=expected,
        loan_months=float(weights.sum()),
    )
