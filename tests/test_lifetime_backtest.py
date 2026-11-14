"""A lifetime PD judged as it is used: a vintage projected from origination against its own
Aalen-Johansen incidence, on a book written in Freddie Mac's format and aggregated for real."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.backtest import lifetime
from creditsurv.backtest.lifetime import projected_incidence, vintage_curve
from creditsurv.data.panel import AGE, OUTCOME, WEIGHT, cells_to_episodes
from creditsurv.models.aft import fit_aft
from creditsurv.models.lifetime_pd import competing_paths, hazard_paths, project_panel
from creditsurv.models.nonparametric import cumulative_incidence

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from creditsurv.models.aft import FitResult

COVARIATES = ["credit_score", "original_ltv", "unemployment_change"]


@pytest.fixture(scope="module")
def shelf(tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame) -> Iterator[Path]:
    from creditsurv.data.aggregate import build_cells
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    root = tmp_path_factory.mktemp("lifetime")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=600, seed=97)
        ingest()
        save_cells(build_cells())
        yield root


@pytest.fixture
def on_shelf(shelf: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(shelf))
    return shelf


@pytest.fixture(scope="module")
def models(shelf: Path, macro_module: pd.DataFrame) -> dict[str, tuple[FitResult, list[str]]]:
    from creditsurv.data.store import load_cells

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(shelf))
        cells = load_cells("exclude")
    fitted = {}
    for cause in ("default", "prepayment"):
        episodes = cells_to_episodes(cells, macro_module, covariates=COVARIATES, cause=cause)
        fitted[cause] = (
            fit_aft(episodes, COVARIATES, " + ".join(COVARIATES), weights_col=WEIGHT),
            COVARIATES,
        )
    return fitted


def _busiest_vintage(macro: pd.DataFrame, runway: int = 24) -> int:
    """The origination year with the most loans that the macro panel follows for ``runway``."""
    from creditsurv.data.store import load_cells

    cells = load_cells("exclude", columns=["origination_month", AGE, WEIGHT])
    at_zero = cells[cells[AGE] == 0]
    years = at_zero["origination_month"].astype(int) // 12
    loans = at_zero[WEIGHT].groupby(years).sum()
    end = macro.index.max()
    followed = [
        year for year in loans.index if (end - pd.Period(f"{year}-12", freq="M")).n >= runway
    ]
    return int(loans.loc[followed].idxmax())


def _book(vintage: int, macro: pd.DataFrame) -> pd.DataFrame:
    from creditsurv.data.store import load_originations

    cells = load_originations("exclude", first=vintage * 12, last=vintage * 12 + 11)
    return cells_to_episodes(cells, macro, covariates=COVARIATES)


def test_the_projection_is_the_competing_paths_of_the_book_weighted_by_its_loans(
    on_shelf: Path, models: dict[str, tuple[FitResult, list[str]]], macro_module: pd.DataFrame
) -> None:
    """Cut into blocks of profiles or not, the curve is each profile's incidence weighted."""
    book = _book(_busiest_vintage(macro_module), macro_module).reset_index(drop=True)
    horizon = 24
    table = projected_incidence(models, book, macro_module, horizon=horizon)

    named = book.assign(loan_id=[f"row_{index:09d}" for index in range(len(book))])
    panel = project_panel(named, macro_module, horizon_months=horizon)
    paths = competing_paths(
        {cause: hazard_paths(fit, panel, covariates) for cause, (fit, covariates) in models.items()}
    )
    weight = named.set_index("loan_id")[WEIGHT].astype(float).reindex(paths.survival.index)
    share = weight / weight.sum()
    np.testing.assert_allclose(
        table["default"].to_numpy(),
        paths.incidence["default"].mul(share, axis=0).sum(axis=0).to_numpy(),
        rtol=1e-12,
    )
    # Every loan is still there, defaulted or repaid, in every month.
    np.testing.assert_allclose(
        (table["survival"] + table["default"] + table["prepayment"]).to_numpy(), 1.0, rtol=1e-12
    )


def test_blocks_of_profiles_change_no_month(
    on_shelf: Path,
    models: dict[str, tuple[FitResult, list[str]]],
    macro_module: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    book = _book(_busiest_vintage(macro_module), macro_module)
    whole = projected_incidence(models, book, macro_module, horizon=18)
    monkeypatch.setattr(lifetime, "PROFILES_PER_BLOCK", 7)
    blocked = projected_incidence(models, book, macro_module, horizon=18)
    pd.testing.assert_frame_equal(blocked, whole, rtol=1e-12)


def test_a_multiplier_on_the_default_hazard_moves_both_incidences(
    on_shelf: Path, models: dict[str, tuple[FitResult, list[str]]], macro_module: pd.DataFrame
) -> None:
    """Rule 4 anchors the default hazard only; more default leaves fewer loans to repay."""
    book = _book(_busiest_vintage(macro_module), macro_module)
    plain = projected_incidence(models, book, macro_module, horizon=12)
    doubled = projected_incidence(
        models, book, macro_module, horizon=12, multipliers={"default": 2.0}
    )
    assert (doubled["default"] > plain["default"]).all()
    assert (doubled["prepayment"] <= plain["prepayment"]).all()


def test_a_projection_needs_both_causes(
    on_shelf: Path, models: dict[str, tuple[FitResult, list[str]]], macro_module: pd.DataFrame
) -> None:
    book = _book(_busiest_vintage(macro_module), macro_module)
    with pytest.raises(ValueError, match="both causes"):
        projected_incidence({"default": models["default"]}, book, macro_module, horizon=6)


def test_a_vintage_is_compared_with_its_own_aalen_johansen_at_the_right_age(
    on_shelf: Path, models: dict[str, tuple[FitResult, list[str]]], macro_module: pd.DataFrame
) -> None:
    """Month ``m`` of the projection is the incidence after ``m`` months at risk, which the
    Aalen-Johansen table holds at age ``m - 1``."""
    from creditsurv.data.store import load_cells

    vintage = _busiest_vintage(macro_module)
    curve = vintage_curve(models, vintage, moratorium="exclude", macro=macro_module, horizon=24)
    cells = load_cells("exclude", columns=["origination_month", AGE, WEIGHT, OUTCOME])
    mine = cells[cells["origination_month"].astype(int) // 12 == vintage]
    observed = cumulative_incidence(mine).set_index("age")

    read = curve.at((12,))
    assert len(read) == 1
    assert read["default_observed"].iloc[0] == pytest.approx(observed.loc[11, "default"])
    assert read["default_predicted"].iloc[0] == pytest.approx(
        curve.predicted.set_index("month").loc[12, "default"]
    )
    assert curve.loans == pytest.approx(float(mine.loc[mine[AGE] == 0, WEIGHT].sum()))
