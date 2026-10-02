"""The selection procedure, run end to end on the fixture book.

Small enough for the suite: two static covariates, three macro candidates, the categoricals
the fixture carries. What is checked is the sequence -- every step runs, a covariate one step
removes is never seen by the next, every elimination says which step removed it and why, and
a second run is read from the cache -- because the numbers belong to the population, not to
twelve hundred loans.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import to_interval_censored
from creditsurv.models.blocks import Pinned
from creditsurv.models.procedure import (
    Fits,
    SelectionRecord,
    Specification,
    _not_identified,
    _worst,
    run_selection,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    from creditsurv.models.aft import FitResult

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def train(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    panel, _ = build_panel(book_dir, macro_module, n_loans=1200, seed=31, params=PARAMS)
    encoded = to_interval_censored(panel).assign(loan_months=1)
    for name in ("purpose", "buyer_type"):
        if name in encoded.columns:
            encoded[name] = encoded[name].astype("category")
    return encoded


def _run(train: pd.DataFrame, *, identity: str = "fixture") -> tuple[SelectionRecord, Fits]:
    reference = str(train["purpose"].cat.categories[0])
    candidates = {"buyer_type": "repeat"} if "buyer_type" in train.columns else {}
    halves = pd.PeriodIndex(train["origination_period"]).year.to_numpy() % 2 == 0
    fits = Fits(train, identity=identity, as_of="2008-12", moratorium="exclude")
    record = run_selection(
        train,
        fits,
        static=["credit_score", "debt_to_income"],
        ordinal=[],
        macro=["ltv_change", "unemployment_change", "equity_volatility"],
        base_categorical={"purpose": reference},
        candidate_categorical=candidates,
        stability=halves,
    )
    return record, fits


def test_the_procedure_runs_every_step_and_says_why_it_removed_each_covariate(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    record, _ = _run(train)

    assert record.selected.covariates, "something must survive a selection on its own DGP"
    assert set(record.eliminated).isdisjoint(record.selected.covariates)
    assert all(reason.startswith("step ") for reason in record.eliminated.values())
    assert not record.screening.empty
    assert "round" in record.stability.columns

    written = selection.generate(record, reports_dir=tmp_path / "reports")
    summary = json.loads((tmp_path / "reports" / selection.SUMMARY_FILE).read_text())
    assert written.exists()
    assert summary["formula"] == record.selected.formula
    for filename in [*selection.TABLE_FILES.values(), selection.FITS_FILE]:
        assert (tmp_path / "reports" / filename).exists(), filename


def test_a_second_run_reads_every_fit_from_the_cache(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection on the whole population is days, so a stopped run must resume."""
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    first, _ = _run(train)
    second, fits = _run(train)

    assert fits.record, "the second run must have asked for fits"
    assert all(entry["cached"] for entry in fits.record)
    assert second.selected == first.selected


def test_report_finds_the_fit_the_selection_ended_on_and_starts_from_it(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``creditsurv report`` fits the specification the selection ended on, on the same rows.

    Started from the selection's fit, Newton goes to the optimum a cold fit reaches, which
    is what makes borrowing it a saving and not a different model. Under another cell table
    there is nothing to borrow.
    """
    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.aft import fit_aft
    from creditsurv.models.procedure import selected_fit

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    record, fits = _run(train)
    ended = fits.fit(record.selected)
    formula = record.selected.formula

    found = selected_fit(identity="fixture", as_of="2008-12", moratorium="exclude", formula=formula)
    assert found is not None
    assert found.fitter.params_.equals(ended.fitter.params_)
    rebuilt = selected_fit(
        identity="rebuilt", as_of="2008-12", moratorium="exclude", formula=formula
    )
    assert rebuilt is None

    covariates = record.selected.covariates
    cold = fit_aft(train, covariates, formula, weights_col=WEIGHT)
    warm = fit_aft(
        train, covariates, formula, weights_col=WEIGHT, initial_point=found.fitter.params_
    )

    assert warm.blocks is not None
    assert warm.blocks.method == "newton"
    moved = (warm.fitter.params_ - cold.fitter.params_).abs() / cold.fitter.standard_errors_
    assert moved.max() < 2e-3


def test_report_starts_from_the_selection_only_on_the_table_it_selected_on(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The glue ``creditsurv report`` uses: the record's formula, date and policy, and the
    cell table as it stands now. A table rebuilt since the selection offers nothing to
    start from, and neither does a record written for another reporting date."""
    from creditsurv.cli import _selection_start
    from creditsurv.config import reports_dir
    from creditsurv.data.store import cells_identity, cells_path
    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(tmp_path / "reports"))
    table = cells_path("exclude")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(b"cells")

    record, fits = _run(train, identity=cells_identity("exclude"))
    selection.generate(record, reports_dir=reports_dir())
    ended = fits.fit(record.selected)

    start = _selection_start("2008-12", "exclude")
    assert start is not None
    assert start.equals(ended.fitter.params_)
    assert _selection_start("2009-12", "exclude") is None

    table.write_bytes(b"cells, rebuilt since")
    assert _selection_start("2008-12", "exclude") is None


def test_the_formula_states_every_reference_level() -> None:
    """Coefficients have to be comparable across refits, so no level is left implicit."""
    spec = Specification(continuous=("credit_score",), categorical=(("purpose", "purchase"),))

    assert spec.plus("mortgage_insurance", reference="uninsured").formula == (
        "credit_score + C(purpose, Treatment('purchase')) "
        "+ C(mortgage_insurance, Treatment('uninsured'))"
    )
    assert spec.minus("purpose").formula == "credit_score"


def _fitted(rows: dict[str, tuple[float, float, float]]) -> SimpleNamespace:
    """A stand-in with the one table the rules read: coefficient, error and p-value."""
    index = pd.MultiIndex.from_tuples([("lambda_", name) for name in rows])
    summary = pd.DataFrame(
        [list(values) for values in rows.values()], index=index, columns=["coef", "se(coef)", "p"]
    )
    return SimpleNamespace(
        fitter=SimpleNamespace(summary=summary, _primary_parameter_name="lambda_")
    )


def test_a_backwards_sign_goes_before_an_insignificant_covariate() -> None:
    """A wrong sign says the specification is wrong, not that the evidence is thin; of two
    wrong signs the weaker goes first, and only one per step."""
    spec = Specification(
        continuous=("credit_score", "ltv_change", "unemployment_change", "equity_volatility")
    )
    result = _fitted(
        {
            "credit_score": (0.3, 0.01, 0.0),
            "ltv_change": (+0.02, 0.001, 0.0),  # expected negative, z = 20
            "unemployment_change": (+0.05, 0.01, 0.0),  # expected negative, z = 5
            "equity_volatility": (-0.001, 0.01, 0.9),  # right sign, insignificant
        }
    )

    worst = _worst(spec, cast("FitResult", result))

    assert worst is not None
    assert worst[0] == "unemployment_change"
    assert worst[1].startswith("wrong sign")


def test_a_reversed_sign_goes_after_a_backwards_one_and_before_a_thin_one() -> None:
    """The first run's marginal/conditional reversal rule, now run rather than argued: a
    covariate with no declared prior whose sign in the full model contradicts its sign
    beside the loan block alone goes, the weaker of two first. A covariate with a declared
    prior answers to the prior instead, whatever it did alone."""
    spec = Specification(
        continuous=(
            "credit_score",
            "mortgage_rate_decline",
            "inflation_rate",
            "unemployment_change",
            "equity_volatility",
        )
    )
    alone = {
        "mortgage_rate_decline": -0.19,
        "inflation_rate": +10.0,
        "unemployment_change": +0.02,
        "equity_volatility": -0.02,
    }
    fitted = {
        "credit_score": (0.3, 0.01, 0.0),
        "mortgage_rate_decline": (+0.018, 0.0008, 0.0),  # no prior, reversed, z = 22.5
        "inflation_rate": (-8.0, 0.063, 0.0),  # no prior, reversed, z = 127
        "unemployment_change": (
            -0.04,
            0.0004,
            0.0,
        ),  # prior negative, agrees, though alone it did not
        "equity_volatility": (-0.001, 0.01, 0.9),  # prior negative, agrees, insignificant
    }

    worst = _worst(spec, cast("FitResult", _fitted(fitted)), alone=alone)
    assert worst is not None
    assert worst[0] == "mortgage_rate_decline"
    assert worst[1].startswith("reversed sign")

    backwards = {**fitted, "unemployment_change": (+0.04, 0.0004, 0.0)}
    worst = _worst(spec, cast("FitResult", _fitted(backwards)), alone=alone)
    assert worst is not None
    assert worst[0] == "unemployment_change"

    agreeing = {
        **fitted,
        "mortgage_rate_decline": (-0.018, 0.0008, 0.0),
        "inflation_rate": (+8.0, 0.063, 0.0),
    }
    worst = _worst(spec, cast("FitResult", _fitted(agreeing)), alone=alone)
    assert worst is not None
    assert worst[0] == "equity_volatility"
    assert worst[1].startswith("p =")


def test_an_unstable_covariate_goes_only_beside_a_larger_one_of_its_dimension() -> None:
    table = pd.DataFrame(
        {
            "covariate": ["equity_volatility", "financial_conditions", "unemployment_change"],
            "dimension": ["financial stress", "financial stress", "labour"],
            "effect_all": [-0.32, +0.05, -0.09],
            "effect_even": [-0.30, -0.02, +0.01],
            "effect_odd": [-0.33, +0.06, -0.10],
            "stable": [True, False, False],
        }
    )

    assert _not_identified(table) == ("financial_conditions", "equity_volatility")
    # Unstable, but alone in its dimension: kept and left visible.
    assert _not_identified(table[table["covariate"] != "financial_conditions"]) is None


def test_the_selection_starts_from_candidates_the_configuration_cannot_change() -> None:
    """The configured specification is the selection's output. Read back as its input, a
    covariate removed once could never be considered again, and one admitted from the
    candidates would enter the next run twice."""
    from creditsurv.config import (
        CATEGORICAL_REFERENCE,
        MACRO_CANDIDATES,
        ORDINAL,
        STATIC_CONTINUOUS,
        TIME_VARYING_CONTINUOUS,
    )
    from creditsurv.models.procedure import (
        BASE_CATEGORICAL,
        CANDIDATE_CATEGORICAL,
        LOAN_CONTINUOUS,
        LOAN_ORDINAL,
    )

    assert set(STATIC_CONTINUOUS) <= set(LOAN_CONTINUOUS)
    assert set(ORDINAL) <= set(LOAN_ORDINAL)
    assert set(TIME_VARYING_CONTINUOUS) <= set(MACRO_CANDIDATES)
    assert set(CATEGORICAL_REFERENCE) <= set(BASE_CATEGORICAL) | set(CANDIDATE_CATEGORICAL)
    assert not set(BASE_CATEGORICAL) & set(CANDIDATE_CATEGORICAL)


def test_the_configuration_is_what_the_last_selection_chose() -> None:
    """The validation's F1: the specification was copied by hand from a procedure nothing
    re-ran. Once ``creditsurv select`` has written its record, the configuration has to
    agree with it, or this fails.

    Every part of the specification, not only the macro block: the loan block can lose a
    covariate to a backwards sign, and ``mortgage_insurance`` and ``buyer_type`` are in the model
    exactly when the record says so (M3). The configuration's reasons for an elimination
    are argued prose and the record's are the rule that fired, so what has to agree there
    is who went."""
    from creditsurv.config import (
        CATEGORICAL_REFERENCE,
        DISTRIBUTION,
        ELIMINATED,
        ORDINAL,
        STATIC_CONTINUOUS,
        TIME_VARYING_CONTINUOUS,
        reports_dir,
    )
    from creditsurv.reporting.selection import SUMMARY_FILE

    path = reports_dir() / SUMMARY_FILE
    if not path.exists():
        pytest.skip("no selection has been run on the whole population yet")
    summary = json.loads(path.read_text())

    assert list(STATIC_CONTINUOUS) == summary["static_continuous"]
    assert list(ORDINAL) == summary["ordinal"]
    assert list(TIME_VARYING_CONTINUOUS) == summary["time_varying_continuous"]
    assert dict(CATEGORICAL_REFERENCE) == summary["categorical"]
    assert set(ELIMINATED) == set(summary["eliminated"])
    # The family is an output too, chosen by rule 2 of docs/rules.md between two selections
    # rather than between two fits of one specification. Records written before the family
    # was an input do not carry it, and a missing key is not a disagreement.
    assert summary.get("distribution", DISTRIBUTION) == DISTRIBUTION


def test_the_prepayment_macro_block_is_what_its_own_selection_chose() -> None:
    """The same tie as above, for the covariates a stress scenario has to move.

    Only the macro block of the prepayment model is copied into the configuration, because the
    scenario is its only consumer -- everything else reads the fit's own formula. A copy with
    nothing holding it to the record is how the adverse path came to promise a volatility spike
    to a model that no longer read volatility, so this is that holding.
    """
    from creditsurv.config import (
        PREPAYMENT_TIME_VARYING_CONTINUOUS,
        STRESSED_COVARIATES,
        TIME_VARYING_CONTINUOUS,
        reports_dir,
    )
    from creditsurv.reporting.selection import record_name

    path = reports_dir() / f"{record_name(distribution='weibull', cause='prepayment')}.json"
    if not path.exists():
        pytest.skip("no prepayment selection has been run on the whole population yet")
    summary = json.loads(path.read_text())

    assert list(PREPAYMENT_TIME_VARYING_CONTINUOUS) == summary["time_varying_continuous"]
    assert summary["cause"] == "prepayment"

    # And the stressed set is the union of the two, in order, with no covariate counted twice.
    assert set(STRESSED_COVARIATES) == set(TIME_VARYING_CONTINUOUS) | set(
        PREPAYMENT_TIME_VARYING_CONTINUOUS
    )
    assert len(STRESSED_COVARIATES) == len(set(STRESSED_COVARIATES))


def test_an_extra_fit_is_estimated_once_and_read_back_after(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report's extra fits ran 78 and 25 minutes on the training half and were thrown
    away, so regenerating a report's prose cost them again."""
    from creditsurv.cli import _cached_fit
    from creditsurv.data.panel import WEIGHT
    from creditsurv.models import aft

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    calls: list[str] = []
    real = aft.fit_aft

    def counted(*args: object, **kwargs: object) -> aft.FitResult:
        calls.append(str(kwargs.get("distribution")))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(aft, "fit_aft", counted)
    fit = _cached_fit(as_of="2008-12", moratorium="exclude")
    formula = "credit_score + ltv_change"

    first = fit(
        train,
        ["credit_score", "ltv_change"],
        formula,
        distribution="loglogistic",
        weights_col=WEIGHT,
    )
    again = fit(
        train,
        ["credit_score", "ltv_change"],
        formula,
        distribution="loglogistic",
        weights_col=WEIGHT,
    )
    other = fit(
        train, ["credit_score", "ltv_change"], formula, distribution="weibull", weights_col=WEIGHT
    )

    assert calls == ["loglogistic", "weibull"]
    assert again.fitter.params_.equals(first.fitter.params_)
    assert other.distribution == "weibull"


# --------------------------------------------------------------------------------------
# The family as an input, and step 10
# --------------------------------------------------------------------------------------


def test_the_family_is_an_input_and_reaches_the_fit_and_its_name(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2 of docs/rules.md compares the two *selected* models, so the whole procedure
    has to run under a family it is told rather than one written into it -- and the two
    runs must not share a cache, since they fit the same formulas on the same rows.
    """
    from creditsurv.data.store import fit_fingerprint
    from creditsurv.models.procedure import selection_description

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    fits = Fits(
        train, identity="fixture", as_of="2008-12", moratorium="exclude", distribution="loglogistic"
    )

    result = fits.fit(Specification(continuous=("credit_score",)))

    assert result.distribution == "loglogistic"
    assert fits.record[0]["distribution"] == "loglogistic"
    common = {
        "identity": "fixture",
        "as_of": "2008-12",
        "moratorium": "exclude",
        "formula": "credit_score",
    }
    weibull = fit_fingerprint(**selection_description(**common))
    loglogistic = fit_fingerprint(**selection_description(**common, distribution="loglogistic"))
    assert weibull != loglogistic


def test_step_ten_removes_the_macro_covariate_whose_effect_is_immaterial() -> None:
    """The threshold is on the effect of one standard deviation on log survival time, and
    it applies to the macro block only: a small coefficient on a macro series is usually a
    statement about which of five correlated series happened to be left.
    """
    from creditsurv.models.procedure import MATERIALITY_THRESHOLD, _immaterial

    spec = Specification(continuous=("credit_score", "ltv_change", "unemployment_change"))
    summary = pd.DataFrame(
        {"coef": [0.5, 0.004, -0.06]},
        index=["credit_score", "ltv_change", "unemployment_change"],
    )
    result = cast(
        "FitResult",
        SimpleNamespace(
            fitter=SimpleNamespace(
                summary=pd.concat({"lambda_": summary}), _primary_parameter_name="lambda_"
            )
        ),
    )
    deviations = pd.Series({"credit_score": 0.001, "ltv_change": 1.0, "unemployment_change": 1.0})

    verdict = _immaterial(spec, result, deviations, macro=["ltv_change", "unemployment_change"])

    assert verdict is not None
    name, effect = verdict
    assert name == "ltv_change"
    assert effect == pytest.approx(0.004)
    assert abs(effect) < MATERIALITY_THRESHOLD
    # The credit score's effect is 0.0005, far under the threshold, and it is not eligible.
    assert _immaterial(spec, result, deviations, macro=["unemployment_change"]) is None, (
        "the loan block is not screened for materiality"
    )


def test_the_materiality_step_is_reported_and_recorded(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    record, _ = _run(train)

    assert list(record.materiality.columns) == [
        "step",
        "removed",
        "effect_1sd",
        "threshold",
        "remaining",
    ]
    assert set(record.materiality["removed"]).isdisjoint(record.selected.covariates)
    for name in record.materiality["removed"].astype(str):
        assert record.eliminated[name].startswith("step 10:")

    selection.generate(record, reports_dir=tmp_path / "reports")
    body = (tmp_path / "reports" / "selection.md").read_text()
    assert "10. Materiality" in body
    assert (tmp_path / "reports" / "selection_materiality.csv").exists()


def test_the_prepayment_model_is_selected_under_its_own_priors() -> None:
    """A model of a different exit has different economics, and the same map read for both
    would eliminate every covariate of the second for disagreeing with the first.

    The credit score is the clearest case: better credit lengthens survival and *shortens*
    the time to repayment, because the borrowers who can refinance are the ones who qualify.
    """
    from creditsurv.models.selection import EXPECTED_SIGNS, PREPAYMENT_SIGNS

    assert EXPECTED_SIGNS["credit_score"] == +1
    assert PREPAYMENT_SIGNS["credit_score"] == -1
    assert PREPAYMENT_SIGNS["ltv_change"] == +1, "leverage that has risen blocks a refinance"
    assert PREPAYMENT_SIGNS["unemployment_change"] == +1, "a weaker labour market prepays less"
    assert PREPAYMENT_SIGNS["house_price_growth"] == -1, "rising prices free equity"
    # The refinancing incentive rule 6 names is not here: the key cannot carry the note
    # rate at this cell count, and the fall in the market rate stands in for it.
    assert "refinance_incentive" not in PREPAYMENT_SIGNS
    assert PREPAYMENT_SIGNS["mortgage_rate_decline"] == -1


def test_a_backwards_sign_is_read_against_the_map_the_run_was_given() -> None:
    from creditsurv.models.procedure import _worst
    from creditsurv.models.selection import PREPAYMENT_SIGNS

    spec = Specification(continuous=("credit_score",))
    summary = pd.DataFrame({"coef": [0.5], "se(coef)": [0.01], "p": [0.0]}, index=["credit_score"])
    result = cast(
        "FitResult",
        SimpleNamespace(
            fitter=SimpleNamespace(
                summary=pd.concat({"lambda_": summary}), _primary_parameter_name="lambda_"
            )
        ),
    )

    assert _worst(spec, result) is None, "a positive score is what the default model expects"
    verdict = _worst(spec, result, signs=PREPAYMENT_SIGNS)
    assert verdict is not None
    assert verdict[0] == "credit_score"
    assert "wrong sign" in verdict[1]


def test_a_prepayment_selection_is_cached_under_a_name_of_its_own() -> None:
    from creditsurv.data.store import fit_fingerprint
    from creditsurv.models.procedure import selection_description

    common = {
        "identity": "cells",
        "as_of": "2021-12",
        "moratorium": "exclude",
        "formula": "credit_score",
    }
    default = selection_description(**common)
    prepayment = selection_description(**common, cause="prepayment")

    assert "cause" not in default
    assert prepayment["cause"] == "prepayment"
    assert fit_fingerprint(**default) != fit_fingerprint(**prepayment)


# --------------------------------------------------------------------------------------
# The selection without the rows
# --------------------------------------------------------------------------------------

STREAMED_CANDIDATES = ["credit_score", "original_ltv", "ltv_change", "unemployment_change"]


@pytest.fixture(scope="module")
def selection_cells(tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame) -> Path:
    """A book filed, ingested and aggregated, as the pipeline does it."""
    from creditsurv.data.aggregate import build_cells
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    root = tmp_path_factory.mktemp("selection")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=700, seed=17)
        ingest()
        return save_cells(build_cells())


def test_the_moments_are_the_same_read_whole_or_read_in_batches(
    selection_cells: Path, macro_module: pd.DataFrame
) -> None:
    """Steps 5 and 6 need a weighted covariance and nothing else, and a covariance is a sum:
    the same sum whether the rows arrive as one frame or as the batches of the cell file.
    """
    from creditsurv.data.panel import WEIGHT, CellBlocks, cells_to_episodes
    from creditsurv.models.selection import weighted_moments

    cells = pd.read_parquet(selection_cells)
    whole = cells_to_episodes(cells, macro_module, covariates=STREAMED_CANDIDATES)
    source = CellBlocks(
        str(selection_cells), macro_module, tuple(STREAMED_CANDIDATES), rows=len(cells) // 5 + 1
    ).prepared()

    held = weighted_moments(whole, STREAMED_CANDIDATES, weight=WEIGHT)
    streamed = weighted_moments(source(), STREAMED_CANDIDATES, weight=WEIGHT)

    assert streamed.rows == held.rows
    assert streamed.loan_months == pytest.approx(held.loan_months)
    np.testing.assert_allclose(
        streamed.covariance.to_numpy(), held.covariance.to_numpy(), rtol=1e-10
    )


def test_the_variance_inflation_step_can_be_given_the_covariance_it_needs() -> None:
    """The selection was taking the same pass over the training half twice, once for its
    correlation table and once inside this step.
    """
    from creditsurv.models.selection import stepwise_vif, weighted_covariance

    rng = np.random.default_rng(6)
    first = rng.normal(size=4_000)
    frame = pd.DataFrame(
        {
            "a": first,
            "b": first + rng.normal(scale=0.01, size=4_000),
            "keep": rng.normal(size=4_000),
        }
    )
    columns = ["keep", "a", "b"]

    from_rows = stepwise_vif(frame, columns, threshold=10.0)
    from_matrix = stepwise_vif(
        None, columns, threshold=10.0, covariance=weighted_covariance(frame, columns)
    )

    assert from_rows[1] == from_matrix[1]
    pd.testing.assert_frame_equal(from_rows[0], from_matrix[0])
    with pytest.raises(ValueError, match="either the rows or a covariance"):
        stepwise_vif(None, columns)


def test_a_selection_streamed_from_the_cells_chooses_what_the_held_panel_chose(
    selection_cells: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refactor's whole claim. The training half used to be expanded into a frame the
    twenty-odd fits then sat beside; here the cell file is described instead and each fit
    reads its own share. The two must end on the same specification, with the same
    covariates eliminated for the same reasons.

    The two runs are given different identities so neither can be served the other's cached
    fits, which is exactly what they would be in production.
    """
    from creditsurv.data.panel import WEIGHT, CellBlocks, cells_to_episodes
    from creditsurv.models.selection import weighted_moments

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    cells = pd.read_parquet(selection_cells)
    held = cells_to_episodes(cells, macro_module, covariates=STREAMED_CANDIDATES)
    halves = (held["origination_month"].to_numpy() // 12) % 2 == 0
    in_memory = run_selection(
        held,
        Fits(held, identity="held", as_of="2014-12", moratorium="exclude"),
        static=["credit_score", "original_ltv"],
        ordinal=[],
        macro=["ltv_change", "unemployment_change"],
        base_categorical={"purpose": "purchase"},
        candidate_categorical={},
        stability=halves,
    )

    source = CellBlocks(
        str(selection_cells), macro_module, tuple(STREAMED_CANDIDATES), rows=4_000
    ).prepared()
    moments = weighted_moments(source(), STREAMED_CANDIDATES, weight=WEIGHT)
    streamed = run_selection(
        None,
        Fits(None, identity="streamed", as_of="2014-12", moratorium="exclude", blocks=source),
        static=["credit_score", "original_ltv"],
        ordinal=[],
        macro=["ltv_change", "unemployment_change"],
        base_categorical={"purpose": "purchase"},
        candidate_categorical={},
        stability=True,
        moments=moments,
    )

    assert streamed.selected.formula == in_memory.selected.formula
    assert streamed.eliminated == in_memory.eliminated
    assert streamed.rows == in_memory.rows
    assert streamed.loan_months == pytest.approx(in_memory.loan_months)
    # The halves are the same halves: taken while reading, or masked over rows held.
    held_halves = in_memory.stability.set_index(["covariate", "round"])
    streamed_halves = streamed.stability.set_index(["covariate", "round"])
    np.testing.assert_allclose(
        streamed_halves["effect_even"].to_numpy(),
        held_halves["effect_even"].to_numpy(),
        rtol=1e-4,
    )


def test_moments_that_do_not_cover_the_candidates_are_refused_before_the_first_fit(
    train: pd.DataFrame,
) -> None:
    """On the production table the pass that produces them is twenty minutes of parquet, so
    a mismatch has to be an error here rather than an index error an hour in.
    """
    from creditsurv.models.selection import weighted_moments

    partial = weighted_moments(train, ["credit_score"], weight="loan_months")

    with pytest.raises(ValueError, match="missing \\['debt_to_income'\\]"):
        run_selection(
            None,
            Fits(None, identity="fixture", as_of="2008-12", moratorium="exclude"),
            static=["credit_score", "debt_to_income"],
            ordinal=[],
            macro=[],
            moments=partial,
        )


def test_two_selections_write_two_records(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`selection.json` belongs to the published model, which the configuration is tested
    against. A second family, or the prepayment model, is the record of a run rather than of
    the model in the configuration, and would otherwise overwrite it.
    """
    from dataclasses import replace as replace_field

    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    record, _ = _run(train)
    reports = tmp_path / "reports"

    assert selection.record_name(distribution="weibull", cause="default") == "selection"
    assert (
        selection.record_name(distribution="loglogistic", cause="default")
        == "selection_loglogistic"
    )
    assert (
        selection.record_name(distribution="weibull", cause="prepayment")
        == "selection_weibull_prepayment"
    )

    selection.generate(record, reports_dir=reports)
    other = selection.generate(
        replace_field(record, distribution="loglogistic"), reports_dir=reports
    )

    assert (reports / "selection.json").exists()
    assert (reports / "selection_loglogistic.json").exists()
    assert (reports / "selection_loglogistic_screening.csv").exists()
    assert other.name == "selection_loglogistic.md"
    assert "--dist loglogistic" in other.read_text()


def test_the_family_command_reads_both_records_and_applies_the_rule(
    selection_cells: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on a fixture book: three selections cached, then the rule applied to what
    they chose. The numbers belong to the population; what this holds is that the command
    finds each family's record and its fit, and that it fits nothing itself.
    """
    from typer.testing import CliRunner

    from creditsurv.cli import app
    from creditsurv.data.panel import WEIGHT, CellBlocks
    from creditsurv.data.store import cells_identity, save_cells
    from creditsurv.models.selection import weighted_moments
    from creditsurv.reporting import selection

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "creditsurv.data.fred.load_macro_panel", lambda *_args, **_kwargs: macro_module
    )
    save_cells(pd.read_parquet(selection_cells))
    # The command looks a fit up by the cell file's identity, as `select` saves it under: the
    # two have to agree, or a cached fit is invisible to the command that needs it.
    identity = cells_identity("exclude")

    candidates = ["credit_score", "ltv_change"]
    for distribution, cause in (
        ("weibull", "default"),
        ("loglogistic", "default"),
        ("weibull", "prepayment"),
    ):
        source = CellBlocks(
            str(tmp_path / "processed" / "cells_exclude.parquet"),
            macro_module,
            tuple(candidates),
            rows=20_000,
            cause=cause,
        ).prepared()
        record = run_selection(
            None,
            Fits(
                None,
                identity=identity,
                as_of="2014-12",
                moratorium="exclude",
                distribution=distribution,
                cause=cause,
                blocks=source,
            ),
            static=["credit_score"],
            ordinal=[],
            macro=["ltv_change"],
            base_categorical={},
            candidate_categorical={},
            moments=weighted_moments(source(), candidates, weight=WEIGHT),
        )
        selection.generate(record, reports_dir=tmp_path / "reports")

    result = CliRunner().invoke(app, ["family", "--as-of", "2014-12"])

    assert result.exit_code == 0, result.output
    body = (tmp_path / "reports" / "family.md").read_text()
    assert "The verdict" in body
    assert "weibull" in body and "loglogistic" in body


def test_a_candidate_that_cannot_be_fitted_is_a_finding_not_a_crash() -> None:
    """It cost five hours to learn this once. The payment state diverged to coefficients of
    1e+80 and took the whole selection with it, on the seventeenth fit of eighteen candidates.

    A screen exists to judge candidates, and "this one cannot be fitted beside the loan block"
    is a judgement about the candidate. The run continues and the record says what happened.
    """
    from lifelines import exceptions

    from creditsurv.models.selection import Moments

    class Refuses(Fits):
        """Fits everything but one candidate, which will not converge."""

        def fit(self, spec: Specification, **kwargs: object) -> FitResult:
            if "equity_volatility" in spec.formula:
                message = "Fitting did not converge after 11 evaluations"
                raise exceptions.ConvergenceError(message)
            return cast("FitResult", _stub_fit(spec))

    def _stub_fit(spec: Specification) -> object:
        index = pd.Index([*spec.continuous, "Intercept"])
        summary = pd.DataFrame(
            {
                "coef": [0.1] * len(index),
                "se(coef)": [0.01] * len(index),
                "p": [0.0] * len(index),
            },
            index=index,
        )
        return SimpleNamespace(
            log_likelihood=-100.0,
            fitter=SimpleNamespace(
                summary=pd.concat({"lambda_": summary}), _primary_parameter_name="lambda_"
            ),
            blocks=None,
        )

    names = ["credit_score", "ltv_change", "equity_volatility"]
    moments = Moments(
        covariance=pd.DataFrame(np.eye(3), index=names, columns=names),
        rows=10,
        loan_months=100.0,
    )

    record = run_selection(
        None,
        Refuses(None, identity="fixture", as_of="2021-12", moratorium="exclude"),
        static=["credit_score"],
        ordinal=[],
        macro=["ltv_change", "equity_volatility"],
        base_categorical={},
        candidate_categorical={},
        moments=moments,
    )

    assert "equity_volatility" in record.eliminated
    assert "did not converge" in record.eliminated["equity_volatility"]
    assert "equity_volatility" not in record.selected.covariates
    # The run went on: the next candidate was screened, and then removed by step 8 on its
    # sign -- which is the stub's doing and, more to the point, proof that step 8 ran at all.
    assert record.eliminated["ltv_change"].startswith("step 8:")
    assert not record.screening.empty


def test_a_warm_start_that_will_not_converge_is_retried_cold(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm start is an optimisation, not part of any rule: it turns a 45-minute cold fit
    into 13 minutes. When it fails it fails as a starting point, and the answer is to start
    where lifelines would have and pay for it.

    The prepayment selection lost six hours to this. Its backward elimination walked six
    damped Newton steps from 144 standard errors out to 3.86e+03, with damping at 1e+12, and
    then SLSQP diverged -- on a model that fits perfectly well from a cold start.
    """
    from lifelines import exceptions

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    attempts: list[bool] = []
    real = Fits._once

    def refuses_warm(self: Fits, spec: Specification, **kwargs: object) -> FitResult:
        warm = kwargs.get("start") is not None
        attempts.append(warm)
        if warm:
            message = "Fitting did not converge after 43 evaluations"
            raise exceptions.ConvergenceError(message)
        return real(self, spec, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Fits, "_once", refuses_warm)
    fits = Fits(train, identity="fixture", as_of="2008-12", moratorium="exclude")
    base = fits.fit(Specification(continuous=("credit_score",)))

    result = fits.fit(Specification(continuous=("credit_score", "debt_to_income")), start=base)

    assert result.log_likelihood < 0, "the cold fit is a real fit"
    assert attempts == [False, True, False], "cold, then a warm start that failed, then cold"


def test_a_nested_model_that_fits_better_than_its_parent_is_refused() -> None:
    """Mathematics, not a threshold: dropping a covariate cannot raise the maximised
    log-likelihood, because the parent could always have set that coefficient to zero.

    A fit that reports an improvement has found the region where lifelines clips the interval
    probability and adds the truncation term unclipped. The prepayment model's step 8 spent two
    hours and forty minutes reaching one such "solution" -- a log-likelihood of -3.06e+06 against
    its parent's -- and this comparison costs nothing.
    """
    from lifelines import exceptions

    from creditsurv.models.procedure import _check_nested

    parent = Specification(continuous=("credit_score", "ltv_change"))
    child = Specification(continuous=("credit_score",))
    unrelated = Specification(continuous=("debt_to_income",))

    def fitted(value: float) -> FitResult:
        return cast("FitResult", SimpleNamespace(log_likelihood=value))

    # Worse, as a nested model must be: nothing happens.
    _check_nested(child, fitted(-1_000.0), parent=parent, parent_fit=fitted(-900.0))
    # Better by less than the allowance: also fine.
    _check_nested(child, fitted(-899.9999), parent=parent, parent_fit=fitted(-900.0))
    # No parent to compare with, or not nested: nothing to say.
    _check_nested(child, fitted(0.0), parent=None, parent_fit=fitted(-900.0))
    _check_nested(unrelated, fitted(0.0), parent=parent, parent_fit=fitted(-900.0))

    with pytest.raises(exceptions.ConvergenceError, match="cannot fit better"):
        _check_nested(child, fitted(-500.0), parent=parent, parent_fit=fitted(-900.0))


def test_the_floor_handed_to_a_nested_fit_is_its_parents_optimum() -> None:
    """A mean, so the parent's total log-likelihood is divided by the exposure it was measured
    over, with one log-likelihood unit allowed back.
    """
    from creditsurv.models.procedure import _NESTED_TOLERANCE, _floor

    parent = Specification(continuous=("credit_score", "ltv_change"))
    child = Specification(continuous=("credit_score",))
    fitted = cast(
        "FitResult",
        SimpleNamespace(log_likelihood=-37_800.0, blocks=SimpleNamespace(loan_months=2_000_000.0)),
    )

    floor = _floor(child, parent, fitted)

    assert floor == pytest.approx((37_800.0 - _NESTED_TOLERANCE) / 2_000_000.0)
    assert _floor(child, None, fitted) is None, "nothing to bound against"
    assert _floor(parent, child, fitted) is None, "the parent is not nested in the child"
    assert _floor(child, parent, cast("FitResult", SimpleNamespace(blocks=None))) is None


def test_the_allowance_stays_inside_the_edge_of_the_clipped_region() -> None:
    """The allowance is one unit because lifelines' clipped region starts directly below the
    maximum, so there is no slack to be had -- and both ends of that are measured.

    Below: two evaluations of the same point over different block partitions agree to 1.97e-16
    relative, three hundredths of a millionth of a unit on this parent, so the arithmetic needs
    no allowance. Above: on the prepayment model's step 8 the optimiser probed points reading
    11.8, 158, 183 and 213 units better than the parent's optimum, and the parent is not the one
    in the wrong -- re-polished with the tolerance driven from 1e-03 to 1e-09 standard errors it
    gains -0.000 units. Those probes are the shallow edge of the clipped region, and an allowance
    wide enough to admit them lets a fit converge onto clipped ground and be cached as an optimum.
    """
    from creditsurv.models.procedure import _NESTED_TOLERANCE, _floor

    parent = Specification(continuous=("credit_score", "ltv_change"))
    child = Specification(continuous=("credit_score",))
    loglik, weight = -151_504_880.223044, 2_112_532_468.0
    fitted = cast(
        "FitResult",
        SimpleNamespace(log_likelihood=loglik, blocks=SimpleNamespace(loan_months=weight)),
    )

    floor = _floor(child, parent, fitted)
    assert floor is not None
    allowance = (-loglik / weight - floor) * weight
    assert allowance == pytest.approx(_NESTED_TOLERANCE)

    # Thirty million times the reproducibility measured over a different block partition.
    assert allowance > 1e6 * 1.97e-16 * abs(loglik)

    # And inside the edge: every probe from that run's clipped ground is still refused.
    for probe in (0.071717178645, 0.071717109308, 0.071717083355, 0.069535115476):
        assert probe < floor, f"{(-loglik / weight - probe) * weight:,.1f} units better, refused"


def test_a_pinned_optimiser_is_not_refitted_cold() -> None:
    """A warm start that diverges is refitted cold; an optimiser pinned against the floor is not.

    The two failures ask for different remedies. A bad starting point is answered by starting
    where lifelines would have -- that is what the retry is for, and it turned a diverging warm
    start into a fit more than once. Being pinned is a statement about the surface: the optimiser
    has found where lifelines' clipped region begins, directly below the parent's optimum, and a
    cold fit walks back to the same maximum and meets the same edge. The prepayment model paid
    that second hour twice before the two were told apart.
    """
    from lifelines import exceptions

    from creditsurv.models.blocks import Pinned

    spec = Specification(continuous=("credit_score",))
    parent = cast("FitResult", SimpleNamespace(fitter=SimpleNamespace(params_=None)))
    attempts: list[object] = []

    def once(fits: Fits, failure: Exception) -> None:
        def attempt(_spec: object, **kwargs: object) -> FitResult:
            attempts.append(kwargs["start"])
            raise failure

        fits._once = attempt  # type: ignore[assignment]

    fits = Fits(None, identity="x", as_of="2021-12", moratorium="exclude")

    once(fits, exceptions.ConvergenceError("diverged"))
    with pytest.raises(exceptions.ConvergenceError):
        fits._estimate(spec, where=None, parity=None, start=parent)
    assert attempts == [parent, None], "the warm start is retried cold"

    attempts.clear()
    once(fits, Pinned("circling a boundary"))
    with pytest.raises(Pinned):
        fits._estimate(spec, where=None, parity=None, start=parent)
    assert attempts == [parent], "being pinned is not a starting point's fault"


def test_the_reference_levels_are_one_set_written_twice_and_held_together() -> None:
    """`config.CATEGORICAL_REFERENCE` is `procedure`'s two blocks merged, and must stay so.

    It cannot be derived: `procedure` imports `config`, so the arrow only goes one way, and the
    other direction is forbidden on purpose -- the comment above `CANDIDATE_CATEGORICAL` argues
    that which covariates are *candidates* must not be read back from the configuration, because
    the configuration is the selection's output and that would be a circle in the argument, not
    just in the imports.

    So the levels are written twice and this holds them equal. A reference level that drifted
    would not fail anything else: both spellings produce a valid formula, the coefficients would
    simply be measured against different baselines in the selection and in the report.
    """
    from creditsurv.config import CATEGORICAL_REFERENCE
    from creditsurv.models.procedure import BASE_CATEGORICAL, CANDIDATE_CATEGORICAL

    assert dict(CATEGORICAL_REFERENCE) == {**BASE_CATEGORICAL, **CANDIDATE_CATEGORICAL}
    assert not set(BASE_CATEGORICAL) & set(CANDIDATE_CATEGORICAL), "a covariate in both blocks"


@pytest.mark.parametrize(
    ("step", "verdict"),
    [("step 9", "_not_identified"), ("step 10", "_immaterial")],
)
def test_every_nested_fit_of_the_selection_is_bounded_by_its_parent(
    train: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    verdict: str,
) -> None:
    """The floor reaches a fit only when its call site says which model this one is nested in.

    Step 8 said it and the other three did not. Steps 9 and 10 both refit a strictly nested
    model warm-started from its parent -- ``current.minus(name)`` from the fit of ``current``
    -- so either could walk into the region where lifelines' clipped likelihood is unbounded
    below and be cached as an optimum, which is what ``_floor`` exists to prevent.

    Neither step eliminates anything on twelve hundred loans, so the verdict each one reads is
    replaced by one that removes a covariate once. What is then checked is the invariant that
    cannot be read off a call site: whatever a fit was started from, if this specification is
    strictly inside that one, a parent was named. The two stability halves are the mirror image
    -- different rows, so no parent can bound them -- and that is checked too.
    """
    from creditsurv.models import procedure

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))

    removals = {"left": 1}

    def unstable(table: pd.DataFrame) -> tuple[str, str] | None:
        if removals["left"] and len(table) > 1:
            removals["left"] -= 1
            names = table["covariate"].astype(str).tolist()
            return names[0], names[1]
        return None

    def immaterial(
        spec: Specification, result: FitResult, deviations: pd.Series, *, macro: list[str]
    ) -> tuple[str, float] | None:
        eligible = [name for name in spec.covariates if name in macro]
        if removals["left"] and len(eligible) > 1:
            removals["left"] -= 1
            return eligible[0], 0.001
        return None

    monkeypatch.setattr(
        procedure, verdict, unstable if verdict == "_not_identified" else immaterial
    )

    asked: list[tuple[Specification, Specification | None, str, Specification | None]] = []
    fitted_on: dict[int, Specification] = {}
    original = Fits.fit

    def spy(
        self: Fits,
        spec: Specification,
        *,
        sample: str = "training half",
        start: FitResult | None = None,
        parent: Specification | None = None,
        **rest: object,
    ) -> FitResult:
        came_from = None if start is None else fitted_on.get(id(start))
        result = original(self, spec, sample=sample, start=start, parent=parent, **rest)  # type: ignore[arg-type]
        fitted_on[id(result)] = spec
        asked.append((spec, parent, sample, came_from))
        return result

    monkeypatch.setattr(Fits, "fit", spy)
    _run(train, identity=f"floor-{verdict}")

    assert not removals["left"], f"{step}: the forced removal never happened"
    nested = [
        (spec, parent, sample)
        for spec, parent, sample, came_from in asked
        if came_from is not None and set(spec.covariates) < set(came_from.covariates)
    ]
    assert nested, f"{step}: no strictly nested model was refitted"
    for spec, parent, sample in nested:
        assert parent is not None, f"{sample}: {spec.formula} refitted without its parent"
        assert set(spec.covariates) <= set(parent.covariates)

    halves = [(parent, sample) for _, parent, sample, _ in asked if "origination years" in sample]
    assert halves, "the fixture runs the stability step"
    assert all(parent is None for parent, _ in halves), "a half cannot be bounded by the whole"


def test_a_parent_cannot_bound_a_fit_that_sees_other_rows(train: pd.DataFrame) -> None:
    """The floor is the parent's optimum divided by the parent's exposure, so it bounds only
    the rows the parent itself saw. Step 9 fits one specification three times -- the whole and
    two halves of the book -- and a parent passed to a half would hand a fit on half the rows
    a bound computed on all of them. Refused before anything is fitted.
    """
    fits = Fits(train, identity="fixture", as_of="2008-12", moratorium="exclude")
    parent = Specification(continuous=("credit_score", "ltv_change"))
    child = Specification(continuous=("credit_score",))

    with pytest.raises(ValueError, match="different sample"):
        fits.fit(child, sample="even origination years", parity=0, parent=parent)
    with pytest.raises(ValueError, match="different sample"):
        fits.fit(child, sample="odd", where=np.zeros(len(train), dtype=bool), parent=parent)
    assert not fits.record, "nothing may be fitted before the bound is checked"


def test_a_rebuilt_table_starts_from_the_same_model_fitted_on_the_old_one(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fit is cached under the cell table's name, size and time of writing, so rebuilding the
    table invalidates every fit made before it: 36 of the 175 on disk are a selection run on a
    table that has since been replaced, 17.5 hours of them.

    They are not useless. The specification is the same and the book is mostly the same book,
    so the old optimum is a far better guess at the new one than lifelines' seed -- a cold fit
    of the training half is 45 to 91 minutes and a warm one 4.7 to 13.

    The rows here are deliberately **identical** and only the declared identity differs,
    because what has to be shown is not that the lookup finds something but that what it finds
    cannot move the answer: the borrowed start must be used, the fit must still be made, and it
    must land where the cold fit landed, inside the thousandth of a standard error the polish
    promises.
    """
    from creditsurv.models.blocks import POLISH_TOLERANCE_SE

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    spec = Specification(continuous=("credit_score", "unemployment_change"))

    before = Fits(
        train, identity="cells_exclude.parquet:1:1", as_of="2008-12", moratorium="exclude"
    )
    cold = before.fit(spec)
    assert cold.blocks is not None

    after = Fits(train, identity="cells_exclude.parquet:2:2", as_of="2008-12", moratorium="exclude")
    # What has to be shown is that the lookup was consulted and answered, and the recorded
    # method no longer distinguishes that: damped Newton goes first from any starting point, so
    # a cold fit says `newton` too.
    borrowed: list[FitResult | None] = []
    original = Fits._elsewhere

    def spy(self: Fits, described: dict[str, object]) -> FitResult | None:
        found = original(self, described)
        borrowed.append(found)
        return found

    monkeypatch.setattr(Fits, "_elsewhere", spy)
    warm = after.fit(spec)

    assert borrowed and borrowed[0] is not None, "the old table's fit was not found"
    assert warm.blocks is not None
    assert warm.blocks.method == "newton"
    assert after.record[-1]["cached"] is False, "a start is not a result"
    assert warm is not cold

    moved = (warm.fitter.params_ - cold.fitter.params_).abs() / cold.fitter.standard_errors_
    assert moved.max() < 2 * POLISH_TOLERANCE_SE, f"the start moved the optimum by {moved.max()}"


def test_a_start_borrowed_from_another_table_can_never_fail_a_fit(
    train: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Pinned` says the surface, not the start: an optimiser held against the parent's optimum
    has found a boundary that sits in the same place for every method and every starting point,
    so a cold retry would spend another hour reaching the same refusal. That reasoning does not
    hold for a start borrowed from a *different* cell table, which is a guess about these rows
    -- so there, and only there, `Pinned` is retried cold.
    """
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    spec = Specification(continuous=("credit_score",))

    before = Fits(
        train, identity="cells_exclude.parquet:1:1", as_of="2008-12", moratorium="exclude"
    )
    before.fit(spec)

    after = Fits(train, identity="cells_exclude.parquet:2:2", as_of="2008-12", moratorium="exclude")
    attempts: list[bool] = []
    original = Fits._once

    def refuse_the_warm_one(self: Fits, spec: Specification, **rest: object) -> FitResult:
        warm = rest.get("start") is not None
        attempts.append(warm)
        if warm:
            message = "the optimiser is circling a boundary it cannot cross"
            raise Pinned(message)
        return original(self, spec, **rest)  # type: ignore[arg-type]

    monkeypatch.setattr(Fits, "_once", refuse_the_warm_one)
    result = after.fit(spec)

    assert attempts == [True, False], "the borrowed start was tried, then given up on"
    assert result.log_likelihood < 0


def test_a_selection_from_one_encoding_chooses_what_a_re_reading_chooses(
    selection_cells: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim at the level of the procedure rather than of one fit.

    A selection makes about thirty fits and every one of them used to re-read the cell file,
    re-derive the macro family, rebuild the design and throw all of it away. On the production
    table that reading is 12.1 minutes against 53 seconds of arithmetic, and the fifteen
    step-7 fits of one logged run each began by recomputing the identical base objective to
    twelve digits.

    Naming the calendar covariates reads the rows **once** -- once per sample, so three times
    in a run with a stability step -- and builds each candidate's design from the keys. What
    has to come out of it is the same specification, with the same covariates eliminated for
    the same reasons.
    """
    from creditsurv.data.panel import WEIGHT, CellBlocks
    from creditsurv.models.selection import weighted_moments

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    source = CellBlocks(
        str(selection_cells), macro_module, tuple(STREAMED_CANDIDATES), rows=4_000
    ).prepared()
    moments = weighted_moments(source(), STREAMED_CANDIDATES, weight=WEIGHT)

    def run(fits: Fits) -> SelectionRecord:
        return run_selection(
            None,
            fits,
            static=["credit_score", "original_ltv"],
            ordinal=[],
            macro=["ltv_change", "unemployment_change"],
            base_categorical={"purpose": "purchase"},
            candidate_categorical={},
            stability=True,
            moments=moments,
        )

    expected = run(
        Fits(None, identity="re-read", as_of="2014-12", moratorium="exclude", blocks=source)
    )
    once = Fits(
        None,
        identity="encoded",
        as_of="2014-12",
        moratorium="exclude",
        blocks=source,
        calendar=["ltv_change", "unemployment_change"],
    )
    got = run(once)

    assert got.selected.formula == expected.selected.formula
    assert got.eliminated == expected.eliminated
    assert got.rows == expected.rows
    assert int(got.loan_months) == int(expected.loan_months)

    # One reading a sample, not one a candidate: the whole half and the two halves of step 9.
    assert len(once.held) <= 3
    assert len(once.record) > len(once.held), "the run must have made more fits than readings"


def test_a_row_mask_is_refused_where_the_rows_are_read_rather_than_held(
    selection_cells: Path, macro_module: pd.DataFrame
) -> None:
    """A mask selects rows of a frame the reading path never builds.

    The sample is a property of the reading -- `vintage_parity`, applied while the cells are
    read -- so a mask silently ignored here would fit the whole half and record it as a half,
    which is the stability step's two rounds comparing the same numbers with themselves.
    """
    from creditsurv.data.panel import CellBlocks

    source = CellBlocks(
        str(selection_cells), macro_module, tuple(STREAMED_CANDIDATES), rows=4_000
    ).prepared()
    fits = Fits(
        None,
        identity="encoded",
        as_of="2014-12",
        moratorium="exclude",
        blocks=source,
        calendar=["ltv_change", "unemployment_change"],
    )

    with pytest.raises(ValueError, match="has no meaning when the rows are read"):
        fits.fit(
            Specification(continuous=("credit_score",)),
            sample="even origination years",
            where=np.ones(3, dtype=bool),
        )
