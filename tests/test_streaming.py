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
from creditsurv.data.panel import WEIGHT, cell_blocks, cell_shape, cells_to_episodes
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


def test_a_further_selection_reads_only_the_cells_it_wants(
    cell_file: Path, macro_module: pd.DataFrame
) -> None:
    """Loans originated in even years, which is how the selection's stability halves are taken."""
    from creditsurv.data.panel import origination_months

    def even(cells: pd.DataFrame) -> np.ndarray:
        years: np.ndarray = (origination_months(cells).to_numpy() // 12) % 2 == 0
        return years

    blocks = list(cell_blocks(cell_file, macro_module, COVARIATES, select=even))
    whole = list(cell_blocks(cell_file, macro_module, COVARIATES))

    assert 0 < sum(len(block) for block in blocks) < sum(len(block) for block in whole)
