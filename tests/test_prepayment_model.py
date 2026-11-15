"""The prepayment model as rebuilt on 14 November 2026, on a book written in Freddie Mac's format.

Rule 9 as first written: the refinancing incentive, in bands, read off the **calendar** key, where
the note rate's band and the observation month put it, so the column no longer couples the two
sides of the kernel. Beside it the two candidates declared with it, burnout and the home-sale
season, and the conditional formulation of rule 18, which the factorised kernel evaluates.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from creditsurv.config import PREPAYMENT_CAUSE, calendar_covariates

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LOAN = ["credit_score"]
PROTECTED = ["refinance_incentive"]
CANDIDATES = ["burnout", "home_sale_season", "ltv_change"]
COVERING = (*LOAN, *PROTECTED, *CANDIDATES)


@pytest.fixture(scope="module")
def crr_table(
    tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame
) -> Iterator[Path]:
    """A book filed, ingested and aggregated with the note rate in the key, as rule 19's is."""
    from creditsurv.data.aggregate import CRR_SPEC, build_cells
    from creditsurv.data.book import DefaultDefinition
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    root = tmp_path_factory.mktemp("prepayment")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=700, seed=23)
        ingest()
        cells = build_cells(spec=CRR_SPEC, definition=DefaultDefinition.CRR)
        yield save_cells(cells, "exclude.crr")


@pytest.fixture
def fits(crr_table: Path, macro_module: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> object:
    from creditsurv.data.panel import CellBlocks
    from creditsurv.data.store import cells_identity
    from creditsurv.models.fits import Fits

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(crr_table.parent.parent))
    source = CellBlocks(
        str(crr_table), macro_module, COVERING, rows=20_000, cause=PREPAYMENT_CAUSE
    ).prepared()
    return Fits(
        None,
        identity=cells_identity("exclude.crr"),
        as_of="2014-12",
        moratorium="exclude.crr",
        distribution="weibull",
        cause=PREPAYMENT_CAUSE,
        blocks=source,
        calendar=calendar_covariates(),
        formulation="conditional",
    )


def test_the_incentive_and_its_history_are_read_off_the_calendar_key(fits: object) -> None:
    """The note rate is not on the loan side of the prepayment model, so the incentive's band,
    a function of the note rate's band and the month, is a function of the calendar key."""
    from creditsurv.models.fits import Fits

    assert isinstance(fits, Fits)
    reading = replace(fits, covering=COVERING).reading()
    loan_keys, calendar_keys = reading.factorisation.keys()
    assert set(loan_keys.columns) == set(LOAN)
    assert {*PROTECTED, *CANDIDATES} <= set(calendar_keys.columns)
    # Bands, not a line: every value is one of rule 9's midpoints.
    assert set(calendar_keys["refinance_incentive"].unique()) <= {-3.5, -0.5, 0.25, 0.75, 1.5, 4.0}
    assert set(calendar_keys["burnout"].unique()) <= {0.0, 6.5, 24.5}


def test_the_exposure_by_band_off_the_keys_is_the_exposure_off_the_episodes(
    fits: object, crr_table: Path, macro_module: pd.DataFrame
) -> None:
    """Rule 12's reference is the band carrying the most loan-months: the same sum two ways."""
    from creditsurv.data.panel import WEIGHT, cells_to_episodes
    from creditsurv.models.fits import Fits
    from creditsurv.models.selection import exposure_by_value

    assert isinstance(fits, Fits)
    keyed = exposure_by_value(replace(fits, covering=COVERING).reading(), "refinance_incentive")
    episodes = cells_to_episodes(
        pd.read_parquet(crr_table), macro_module, covariates=list(COVERING), cause=PREPAYMENT_CAUSE
    )
    direct = episodes.groupby("refinance_incentive")[WEIGHT].sum().astype(float)
    pd.testing.assert_series_equal(
        keyed.sort_index(), direct.sort_index(), check_names=False, check_index_type=False
    )
    assert keyed.index[0] == direct.idxmax()


def test_the_prepayment_selection_runs_with_the_incentive_protected_in_bands(fits: object) -> None:
    """End to end: the incentive enters the base as a factor and stays, the candidates are
    screened under rule 6's signs, and every fit is the conditional formulation's."""
    from creditsurv.models.fits import Fits
    from creditsurv.models.procedure import run_selection
    from creditsurv.models.selection import PREPAYMENT_SIGNS, exposure_by_value

    assert isinstance(fits, Fits)
    reference = float(
        exposure_by_value(replace(fits, covering=COVERING).reading(), "refinance_incentive").index[
            0
        ]
    )
    record = run_selection(
        None,
        fits,
        static=LOAN,
        ordinal=[],
        protected_macro=tuple(PROTECTED),
        banded={"refinance_incentive": reference},
        macro=CANDIDATES,
        base_categorical={},
        candidate_categorical={},
        signs=PREPAYMENT_SIGNS,
    )

    assert f"C(refinance_incentive, Treatment({reference}))" in record.selected.formula
    assert "refinance_incentive" not in record.eliminated
    summary = record.summary()
    assert "refinance_incentive" in cast("list[str]", summary["time_varying_continuous"])
    assert summary["cause"] == PREPAYMENT_CAUSE
    # Every candidate was screened, under its declared sign.
    screened = set(record.screening["covariate"])
    assert set(CANDIDATES) <= screened
    expected = record.screening.drop_duplicates("covariate").set_index("covariate")["expected_sign"]
    assert {name: int(expected[name]) for name in CANDIDATES} == {
        name: PREPAYMENT_SIGNS[name] for name in CANDIDATES
    }
    # The model it ended on is a likelihood, fitted in the conditional formulation (rule 18).
    final = fits.fit(record.selected)
    assert final.formulation == "conditional"
    assert np.isfinite(final.log_likelihood)


def test_the_books_own_hazard_is_its_exits_over_the_months_at_risk() -> None:
    from creditsurv.models.nonparametric import observed_hazard_by_age

    outcomes = pd.DataFrame(
        {
            "age": [0, 0, 0, 1, 1],
            "outcome": ["none", "default", "prepayment", "none", "prepayment"],
            "loan_months": [970.0, 10.0, 20.0, 900.0, 50.0],
        }
    )
    hazard = observed_hazard_by_age(outcomes, "default")
    assert hazard["age"].tolist() == [0, 1]
    assert hazard["at_risk"].tolist() == [1000.0, 950.0]
    assert hazard["hazard"].tolist() == pytest.approx([0.01, 0.0])


def test_rule_2_chooses_the_prepayment_family_against_the_books_own_incidence(
    crr_table: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both families through a prepayment selection, then the rule: the gap to the book's
    Aalen-Johansen incidence of prepayment, with the default hazard at the book's own."""
    from typer.testing import CliRunner

    from creditsurv.cli import app
    from creditsurv.data.panel import month_ordinal
    from creditsurv.data.store import cells_identity
    from creditsurv.models.fits import Fits, cell_source
    from creditsurv.models.procedure import run_selection
    from creditsurv.models.selection import PREPAYMENT_SIGNS, exposure_by_value
    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(crr_table.parent.parent))
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "creditsurv.data.fred.load_macro_panel", lambda *_args, **_kwargs: macro_module
    )
    covering = (*LOAN, *PROTECTED, "home_sale_season")
    for distribution in ("weibull", "loglogistic"):
        source = cell_source(
            "exclude.crr",
            macro_module,
            list(covering),
            block_rows=20_000,
            until=month_ordinal(pd.Period("2014-12", freq="M")),
            cause=PREPAYMENT_CAUSE,
        )
        fits = Fits(
            None,
            identity=cells_identity("exclude.crr"),
            as_of="2014-12",
            moratorium="exclude.crr",
            distribution=distribution,
            cause=PREPAYMENT_CAUSE,
            blocks=source,
            calendar=calendar_covariates(),
            formulation="conditional",
        )
        exposure = exposure_by_value(replace(fits, covering=covering).reading(), PROTECTED[0])
        record = run_selection(
            None,
            fits,
            static=LOAN,
            ordinal=[],
            protected_macro=tuple(PROTECTED),
            banded={PROTECTED[0]: float(exposure.index[0])},
            macro=["home_sale_season"],
            base_categorical={},
            candidate_categorical={},
            signs=PREPAYMENT_SIGNS,
        )
        selection.generate(record, reports_dir=tmp_path / "reports")

    result = CliRunner().invoke(
        app,
        ["family", "--cause", "prepayment", "--moratorium", "exclude.crr", "--as-of", "2014-12"],
    )

    assert result.exit_code == 0, result.output
    body = (tmp_path / "reports" / "family_prepayment.md").read_text()
    assert "cumulative incidence of prepayment" in body
    assert "weibull" in body and "loglogistic" in body
    # The published family's own report is not this one, and is not written by it.
    assert not (tmp_path / "reports" / "family.md").exists()
