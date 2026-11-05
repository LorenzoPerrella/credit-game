"""The book on the shelf, scored: the quantity the backtest validated, computed the way it was.

On a book written in Freddie Mac's own format, ingested and aggregated for real, so the cell
reader, the window filter and the macro family are exercised on every test here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.backtest.metrics import annualised, grade_of
from creditsurv.backtest.runner import predicted_hazard
from creditsurv.backtest.scoring import (
    REGULATORY_PD_FLOOR,
    anchor_streamed,
    score_month,
)
from creditsurv.data.panel import AGE, WEIGHT, cells_to_episodes, ended_in, month_ordinal
from creditsurv.models.aft import fit_aft
from creditsurv.models.anchoring import anchor_on_window

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from creditsurv.models.aft import FitResult

COVARIATES = ["credit_score", "original_ltv", "unemployment_change"]
#: Inside the fixture book, which ends in the 2010s: rule 1's own window is 2022-01 to 2024-12.
WINDOW = ("2013-01", "2013-12")


@pytest.fixture(scope="module")
def shelf(tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame) -> Iterator[Path]:
    from creditsurv.data.aggregate import build_cells
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    root = tmp_path_factory.mktemp("scoring")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=600, seed=83)
        ingest()
        save_cells(build_cells())
        yield root


@pytest.fixture(scope="module")
def fitted(shelf: Path, macro_module: pd.DataFrame) -> FitResult:
    from creditsurv.data.store import load_cells_window

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(shelf))
        cells = load_cells_window("exclude", last=month_ordinal(pd.Period("2012-12", freq="M")))
    episodes = cells_to_episodes(cells, macro_module, covariates=COVARIATES)
    return fit_aft(episodes, COVARIATES, " + ".join(COVARIATES), weights_col=WEIGHT)


@pytest.fixture
def on_shelf(shelf: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(shelf))
    return shelf


def _window(macro: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    from creditsurv.data.store import load_cells_window

    cells = load_cells_window(
        "exclude",
        first=month_ordinal(pd.Period(first, freq="M")),
        last=month_ordinal(pd.Period(last, freq="M")),
    )
    return cells_to_episodes(cells, macro, covariates=COVARIATES)


def test_the_multiplier_streamed_a_month_at_a_time_is_the_one_read_whole(
    on_shelf: Path, fitted: FitResult, macro_module: pd.DataFrame
) -> None:
    """`views` reads the anchoring window in one frame; the scorer sums it month by month."""
    whole = _window(macro_module, *WINDOW)
    expected = anchor_on_window(
        predicted_hazard(fitted, whole, COVARIATES),
        ended_in(whole),
        whole[WEIGHT],
        whole["period"],
        window=WINDOW,
    )
    streamed = anchor_streamed(
        fitted, COVARIATES, moratorium="exclude", macro=macro_module, window=WINDOW
    )
    assert streamed.multiplier == pytest.approx(expected.multiplier, rel=1e-12)
    assert streamed.actual_defaults == pytest.approx(expected.actual_defaults, rel=1e-12)
    assert streamed.expected_defaults == pytest.approx(expected.expected_defaults, rel=1e-12)
    assert streamed.loan_months == expected.loan_months


def test_a_scored_cell_carries_the_pd_the_master_scale_graded(
    on_shelf: Path, fitted: FitResult, macro_module: pd.DataFrame
) -> None:
    """``1 - (1 - k h)^12`` at the cell's own covariates, the floor beside it, and its grade."""
    anchor = anchor_streamed(
        fitted, COVARIATES, moratorium="exclude", macro=macro_module, window=WINDOW
    )
    month = pd.Period("2014-03", freq="M")
    book = score_month(
        fitted,
        COVARIATES,
        fit="fixture",
        month=month,
        anchor=anchor,
        moratorium="exclude",
        macro=macro_module,
    )
    episodes = _window(macro_module, str(month), str(month))
    hazard = anchor.apply(predicted_hazard(fitted, episodes, COVARIATES))
    cells = book.cells

    assert len(cells) == len(episodes)
    assert cells[WEIGHT].sum() == episodes[WEIGHT].sum()
    np.testing.assert_allclose(cells["hazard"], hazard, rtol=1e-15)
    np.testing.assert_allclose(cells["pd_12m"], annualised(hazard), rtol=1e-15)
    np.testing.assert_array_equal(
        cells["pd_12m_floored"], np.maximum(cells["pd_12m"], REGULATORY_PD_FLOOR)
    )
    np.testing.assert_array_equal(cells["grade"], grade_of(cells["pd_12m"].to_numpy()))
    # What the cell is, so a reader can segment the book without the cell file.
    assert {AGE, "credit_score", "delinquency_state", "origination_month"} <= set(cells.columns)


def test_the_floor_is_a_second_object_and_never_lowers_a_pd(
    on_shelf: Path, fitted: FitResult, macro_module: pd.DataFrame
) -> None:
    anchor = anchor_streamed(
        fitted, COVARIATES, moratorium="exclude", macro=macro_module, window=WINDOW
    )
    book = score_month(
        fitted,
        COVARIATES,
        fit="fixture",
        month=pd.Period("2014-03", freq="M"),
        anchor=anchor,
        moratorium="exclude",
        macro=macro_module,
    )
    cells = book.cells
    assert (cells["pd_12m_floored"] >= cells["pd_12m"]).all()
    assert (cells["pd_12m_floored"] >= REGULATORY_PD_FLOOR).all()
    unbound = cells["pd_12m"] >= REGULATORY_PD_FLOOR
    assert (cells.loc[unbound, "pd_12m_floored"] == cells.loc[unbound, "pd_12m"]).all()

    summary = book.summary()
    grades = book.by_grade()
    assert grades["loans"].sum() == pytest.approx(summary["loans"])
    assert grades["share"].sum() == pytest.approx(1.0)
    weight = cells[WEIGHT].astype(float)
    assert summary["pd_12m"] == pytest.approx(
        float((cells["pd_12m"] * weight).sum() / weight.sum())
    )
    assert summary["pd_12m_floored"] >= summary["pd_12m"]
    assert summary["fit"] == "fixture"


def test_a_month_the_table_does_not_observe_is_refused(
    on_shelf: Path, fitted: FitResult, macro_module: pd.DataFrame
) -> None:
    from creditsurv.data.store import last_observed_month

    last = last_observed_month("exclude")
    after = pd.Period(year=last // 12, month=last % 12 + 1, freq="M") + 1
    anchor = anchor_streamed(
        fitted, COVARIATES, moratorium="exclude", macro=macro_module, window=WINDOW
    )
    with pytest.raises(ValueError, match="No cells are observed"):
        score_month(
            fitted,
            COVARIATES,
            fit="fixture",
            month=after,
            anchor=anchor,
            moratorium="exclude",
            macro=macro_module,
        )


def test_the_last_observed_month_is_the_tables_own(on_shelf: Path) -> None:
    from creditsurv.data.store import last_observed_month, load_cells

    cells = load_cells("exclude", columns=["origination_month", AGE])
    observed = cells["origination_month"].astype(int) + cells[AGE].astype(int)
    assert last_observed_month("exclude") == int(observed.max())


def test_the_score_command_scores_the_published_fit_and_writes_the_cells(
    on_shelf: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: a cached fit of the configured specification, found the way `views` finds
    it, scored on one month, and the cells written where only this machine can read them."""
    from typer.testing import CliRunner

    from creditsurv.cli import app
    from creditsurv.config import default_covariates, default_formula, default_shape
    from creditsurv.data.panel import CellBlocks
    from creditsurv.data.store import cells_path, fit_fingerprint, save_fit
    from creditsurv.models.aft import fit_streamed
    from creditsurv.models.fits import fit_description

    monkeypatch.setattr(
        "creditsurv.data.fred.load_macro_panel", lambda *_args, **_kwargs: macro_module
    )
    as_of = "2012-12"
    covariates, formula, shape = default_covariates(), default_formula(), default_shape()
    source = CellBlocks(
        str(cells_path("exclude")),
        macro_module,
        tuple(covariates),
        rows=20_000,
        months=(None, month_ordinal(pd.Period(as_of, freq="M"))),
    ).prepared()
    model = fit_streamed(source, covariates, formula, weights_col=WEIGHT, ancillary=shape)
    record = model.blocks
    assert record is not None
    described = fit_description(
        (model.n_episodes, int(record.loan_months)),
        formula,
        as_of=as_of,
        moratorium="exclude",
        ancillary=shape,
    )
    fingerprint = fit_fingerprint(**described)
    save_fit(model, fingerprint, described)

    written = tmp_path / "book.parquet"
    result = CliRunner().invoke(
        app,
        [
            "score",
            "--as-of",
            as_of,
            "--month",
            "2014-03",
            "--anchor-window",
            ",".join(WINDOW),
            "--output",
            str(written),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fingerprint in result.output
    assert "twelve-month PD" in result.output
    cells = pd.read_parquet(written)
    assert {"pd_12m", "pd_12m_floored", "grade", WEIGHT} <= set(cells.columns)
    assert (cells["pd_12m_floored"] >= REGULATORY_PD_FLOOR).all()
