"""The stress scenario and the fitted specification have to describe the same model.

The validation found the adverse scenario shocking four series of which two fed no covariate in the
final formula, while two of the model's four macro covariates -- ``equity_volatility``, the largest
standardised effect, among them -- had no shocked path. The published multiplier understated the
model's own sensitivity as a result, and nothing failed: the scenario predated the specification by
three days and simply was not updated.

These tests make the next drift a failure instead of a finding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.config import TIME_VARYING_CONTINUOUS
from creditsurv.features import MACRO_DERIVED, MACRO_SOURCES, add_macro_family
from creditsurv.models.lifetime_pd import ADVERSE

if TYPE_CHECKING:
    from collections.abc import Iterable


def test_every_shocked_series_feeds_a_covariate_the_model_reads() -> None:
    """A shock with no path to the model moves nothing and reads as a stress."""
    read = {source for name in TIME_VARYING_CONTINUOUS for source in MACRO_SOURCES[name]}
    inert = set(ADVERSE.shocks) - read

    assert not inert, f"the scenario shocks {sorted(inert)}, which no fitted covariate reads"


def test_every_macro_covariate_the_model_reads_is_moved_by_the_scenario() -> None:
    """A fitted covariate with no shocked source holds still under stress, which
    understates the model's sensitivity without any sign that it has."""
    unmoved = [
        name
        for name in TIME_VARYING_CONTINUOUS
        if not set(MACRO_SOURCES[name]) & set(ADVERSE.shocks)
    ]

    assert not unmoved, f"{unmoved} have no shocked source, so the scenario cannot move them"


def test_every_derived_covariate_has_its_sources_stated() -> None:
    assert set(MACRO_SOURCES) == set(MACRO_DERIVED)


def _build(macro: pd.DataFrame) -> set[str]:
    """Which derived covariates the builder produces from this macro panel."""
    ages = pd.Series(range(12, 24))
    origination = pd.Series([2005 * 12] * len(ages))
    # The note rate too: the origination spread and the refinancing incentive are the loan's
    # own rate against a market one, so a panel without it is not a full panel for them.
    episodes = pd.DataFrame(
        {
            "original_ltv": [80.0] * len(ages),
            "term_years": [30] * len(ages),
            "note_rate": [6.0] * len(ages),
        }
    )
    add_macro_family(episodes, macro, origination, origination + ages, lag_months=3)
    return set(episodes.columns) & set(MACRO_DERIVED)


def _without(macro: pd.DataFrame, series: Iterable[str]) -> pd.DataFrame:
    return macro.drop(columns=list(series))


@pytest.mark.parametrize("covariate", sorted(MACRO_SOURCES))
def test_the_source_map_is_what_the_builder_actually_reads(
    macro: pd.DataFrame, covariate: str
) -> None:
    """The map is only worth having if it is true.

    With every series present the covariate is built; with its stated sources removed it
    is not. A covariate that still appears after its sources are gone reads something the
    map does not list, and a scenario checked against the map would miss it.
    """
    assert covariate in _build(macro), f"{covariate} is not built from the full panel"
    assert covariate not in _build(_without(macro, MACRO_SOURCES[covariate])), (
        f"{covariate} is still built without {MACRO_SOURCES[covariate]}, "
        "so it reads a series the map does not state"
    )
