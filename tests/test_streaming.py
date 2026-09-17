"""Fitting from the cell file, without the episode frame ever existing.

The episode frame is the largest object the pipeline holds, and holding it beside the fit is
what put a production fit at a 15 GB footprint. These tests hold the streamed path to the
in-memory one: the same coefficients, the same counts, on a book written in Freddie Mac's own
format and aggregated for real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.aggregate import build_cells
from creditsurv.data.ingest import ingest
from creditsurv.data.panel import (
    WEIGHT,
    CellBlocks,
    cell_blocks,
    cell_shape,
    cells_to_episodes,
)
from creditsurv.data.store import save_cells
from creditsurv.models.aft import fit_aft, fit_streamed
from fixtures import write_book_archives

if TYPE_CHECKING:
    from pathlib import Path

COVARIATES = ["credit_score", "original_ltv", "purpose"]
FORMULA = "credit_score + original_ltv + C(purpose, Treatment('purchase'))"


@pytest.fixture(scope="module")
def cell_file(tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame) -> Path:
    """A book simulated, filed as archives, ingested and aggregated, as the pipeline does."""
    root = tmp_path_factory.mktemp("streamed")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=900, seed=9)
        ingest()
        return save_cells(build_cells())


@pytest.fixture(scope="module")
def cells(cell_file: Path) -> pd.DataFrame:
    return pd.read_parquet(cell_file)


def test_the_width_and_the_levels_are_read_from_the_whole_file(
    cell_file: Path, cells: pd.DataFrame
) -> None:
    step, levels = cell_shape(cell_file)

    assert step == 1
    assert set(levels["purpose"]) == set(cells["purpose"].astype(str))
    assert "occupancy" in levels


def test_a_fit_streamed_from_the_file_is_the_fit_made_in_memory(
    cell_file: Path, cells: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    episodes = cells_to_episodes(cells, macro_module, covariates=COVARIATES)
    in_memory = fit_aft(episodes, COVARIATES, FORMULA, weights_col=WEIGHT)

    streamed = fit_streamed(
        cell_blocks(cell_file, macro_module, COVARIATES, rows=len(cells) // 4 + 1),
        COVARIATES,
        FORMULA,
        weights_col=WEIGHT,
    )

    assert streamed.blocks is not None
    assert streamed.blocks.blocks > 2, "the file must arrive in several blocks"
    assert streamed.n_episodes == in_memory.n_episodes
    assert streamed.n_events == in_memory.n_events
    np.testing.assert_allclose(
        streamed.fitter.params_.to_numpy(), in_memory.fitter.params_.to_numpy(), rtol=1e-6
    )
    assert streamed.log_likelihood == pytest.approx(in_memory.log_likelihood, rel=1e-10)


def test_a_window_of_observation_months_selects_what_a_split_would(
    cell_file: Path, cells: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    """The training half, taken while reading rather than after expanding."""
    from creditsurv.backtest.splits import split_cells

    as_of = (
        pd.PeriodIndex(
            cells_to_episodes(cells, macro_module, covariates=COVARIATES)["period"]
        ).max()
        - 6
    )
    cut = as_of.year * 12 + as_of.month - 1

    split = split_cells(cells, macro_module, as_of, covariates=COVARIATES)
    streamed = sum(
        len(block) for block in cell_blocks(cell_file, macro_module, COVARIATES, months=(None, cut))
    )
    after = sum(
        len(block)
        for block in cell_blocks(cell_file, macro_module, COVARIATES, months=(cut + 1, None))
    )

    assert streamed == len(split.train)
    assert after == len(split.test)


def test_the_stability_halves_are_taken_while_reading(
    cell_file: Path, macro_module: pd.DataFrame
) -> None:
    """Loans originated in even and in odd years, as the selection's step 9 takes them."""
    even = sum(
        len(block) for block in cell_blocks(cell_file, macro_module, COVARIATES, vintage_parity=0)
    )
    odd = sum(
        len(block) for block in cell_blocks(cell_file, macro_module, COVARIATES, vintage_parity=1)
    )
    whole = sum(len(block) for block in cell_blocks(cell_file, macro_module, COVARIATES))

    assert even > 0
    assert odd > 0
    assert even + odd == whole


def test_the_same_fit_comes_out_of_three_processes_as_out_of_one(
    cell_file: Path, macro_module: pd.DataFrame
) -> None:
    """Each worker reads its own share, so the blocks are never sent or held twice."""
    source = CellBlocks(str(cell_file), macro_module, tuple(COVARIATES), rows=400)

    alone = fit_streamed(source, COVARIATES, FORMULA, weights_col=WEIGHT)
    shared = fit_streamed(source, COVARIATES, FORMULA, weights_col=WEIGHT, workers=3)

    assert shared.blocks is not None and alone.blocks is not None
    assert shared.blocks.blocks == alone.blocks.blocks
    assert shared.n_episodes == alone.n_episodes
    assert shared.n_events == alone.n_events
    assert shared.blocks.loan_months == pytest.approx(alone.blocks.loan_months)
    np.testing.assert_allclose(
        shared.fitter.params_.to_numpy(), alone.fitter.params_.to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        shared.fitter.standard_errors_.to_numpy(),
        alone.fitter.standard_errors_.to_numpy(),
        rtol=1e-8,
    )
    assert shared.log_likelihood == pytest.approx(alone.log_likelihood, rel=1e-12)


def test_fitting_in_processes_needs_a_description_of_the_rows(
    cell_file: Path, macro_module: pd.DataFrame
) -> None:
    blocks = cell_blocks(cell_file, macro_module, COVARIATES)

    with pytest.raises(TypeError, match="blocks\\(part, of\\)"):
        fit_streamed(blocks, COVARIATES, FORMULA, weights_col=WEIGHT, workers=2)


def test_a_file_written_under_the_former_names_streams_under_the_current_ones(
    tmp_path: Path, macro_module: pd.DataFrame
) -> None:
    """The cells on disk predate the rename, and their levels do too.

    Read without translating them, every renamed level became a missing category -- a design
    of NaNs, which lifelines refuses only once the whole file has been read.
    """
    months = pd.PeriodIndex(macro_module.index[-40:-1])
    former = pd.DataFrame(
        {
            "fico_s": [-0.4, 1.0] * len(months),
            "orig_ltv": [75.0, 85.0] * len(months),
            "dti": [32.0, 28.0] * len(months),
            "purpose": pd.Categorical(["refinance_cashout", "purchase"] * len(months)),
            "has_mi": pd.Categorical(["N", "Y"] * len(months)),
            "first_time_buyer": pd.Categorical(["Y", "N"] * len(months)),
            "orig_month": [month.year * 12 + month.month - 1 for month in months for _ in (0, 1)],
            "age": [0, 1] * len(months),
            "event": [False, True] * len(months),
            "n": [10, 3] * len(months),
        }
    )
    path = tmp_path / "cells_exclude.parquet"
    former.to_parquet(path)

    step, levels = cell_shape(path)
    blocks = list(cell_blocks(path, macro_module, ["credit_score", "purpose"], rows=8))

    assert step == 1
    assert list(levels["purpose"]) == ["cash_out_refinance", "purchase"]
    assert list(levels["mortgage_insurance"]) == ["insured", "uninsured"]
    assert blocks
    for frame in blocks:
        assert not frame.isna().to_numpy().any()
        assert set(frame["purpose"].cat.categories) == {"cash_out_refinance", "purchase"}
        assert frame["credit_score"].between(600.0, 800.0).all()
