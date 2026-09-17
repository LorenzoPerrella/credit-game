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

import pandas as pd
import pytest

from creditsurv.data.panel import to_interval_censored
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
    intercept=4.9,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def train(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    panel, _ = build_panel(book_dir, macro_module, n_loans=1200, seed=31, params=PARAMS)
    encoded = to_interval_censored(panel).assign(n=1)
    for name in ("purpose", "first_time_buyer"):
        if name in encoded.columns:
            encoded[name] = encoded[name].astype("category")
    return encoded


def _run(train: pd.DataFrame, *, identity: str = "fixture") -> tuple[SelectionRecord, Fits]:
    reference = str(train["purpose"].cat.categories[0])
    candidates = {"first_time_buyer": "N"} if "first_time_buyer" in train.columns else {}
    halves = pd.PeriodIndex(train["orig_period"]).year.to_numpy() % 2 == 0
    fits = Fits(train, identity=identity, as_of="2008-12", moratorium="exclude")
    record = run_selection(
        train,
        fits,
        static=["fico_s", "dti"],
        ordinal=[],
        macro=["cltv_drift", "unemp_gap", "vix"],
        base_categorical={"purpose": reference},
        candidate_categorical=candidates,
        halves=halves,
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
    spec = Specification(continuous=("fico_s",), categorical=(("purpose", "purchase"),))

    assert spec.plus("has_mi", reference="N").formula == (
        "fico_s + C(purpose, Treatment('purchase')) + C(has_mi, Treatment('N'))"
    )
    assert spec.minus("purpose").formula == "fico_s"


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
    spec = Specification(continuous=("fico_s", "cltv_drift", "unemp_gap", "vix"))
    result = _fitted(
        {
            "fico_s": (0.3, 0.01, 0.0),
            "cltv_drift": (+0.02, 0.001, 0.0),  # expected negative, z = 20
            "unemp_gap": (+0.05, 0.01, 0.0),  # expected negative, z = 5
            "vix": (-0.001, 0.01, 0.9),  # right sign, insignificant
        }
    )

    worst = _worst(spec, cast("FitResult", result))

    assert worst is not None
    assert worst[0] == "unemp_gap"
    assert worst[1].startswith("wrong sign")


def test_a_reversed_sign_goes_after_a_backwards_one_and_before_a_thin_one() -> None:
    """The first run's marginal/conditional reversal rule, now run rather than argued: a
    covariate with no declared prior whose sign in the full model contradicts its sign
    beside the loan block alone goes, the weaker of two first. A covariate with a declared
    prior answers to the prior instead, whatever it did alone."""
    spec = Specification(continuous=("fico_s", "rate_gap", "inflation", "unemp_gap", "vix"))
    alone = {"rate_gap": -0.19, "inflation": +10.0, "unemp_gap": +0.02, "vix": -0.02}
    fitted = {
        "fico_s": (0.3, 0.01, 0.0),
        "rate_gap": (+0.018, 0.0008, 0.0),  # no prior, reversed, z = 22.5
        "inflation": (-8.0, 0.063, 0.0),  # no prior, reversed, z = 127
        "unemp_gap": (-0.04, 0.0004, 0.0),  # prior negative, agrees, though alone it did not
        "vix": (-0.001, 0.01, 0.9),  # prior negative, agrees, insignificant
    }

    worst = _worst(spec, cast("FitResult", _fitted(fitted)), alone=alone)
    assert worst is not None
    assert worst[0] == "rate_gap"
    assert worst[1].startswith("reversed sign")

    backwards = {**fitted, "unemp_gap": (+0.04, 0.0004, 0.0)}
    worst = _worst(spec, cast("FitResult", _fitted(backwards)), alone=alone)
    assert worst is not None
    assert worst[0] == "unemp_gap"

    agreeing = {**fitted, "rate_gap": (-0.018, 0.0008, 0.0), "inflation": (+8.0, 0.063, 0.0)}
    worst = _worst(spec, cast("FitResult", _fitted(agreeing)), alone=alone)
    assert worst is not None
    assert worst[0] == "vix"
    assert worst[1].startswith("p =")


def test_an_unstable_covariate_goes_only_beside_a_larger_one_of_its_dimension() -> None:
    table = pd.DataFrame(
        {
            "covariate": ["vix", "nfci_lagged", "unemp_gap"],
            "dimension": ["financial stress", "financial stress", "labour"],
            "effect_all": [-0.32, +0.05, -0.09],
            "effect_even": [-0.30, -0.02, +0.01],
            "effect_odd": [-0.33, +0.06, -0.10],
            "stable": [True, False, False],
        }
    )

    assert _not_identified(table) == ("nfci_lagged", "vix")
    # Unstable, but alone in its dimension: kept and left visible.
    assert _not_identified(table[table["covariate"] != "nfci_lagged"]) is None


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
    covariate to a backwards sign, and ``has_mi`` and ``first_time_buyer`` are in the model
    exactly when the record says so (M3). The configuration's reasons for an elimination
    are argued prose and the record's are the rule that fired, so what has to agree there
    is who went."""
    from creditsurv.config import (
        CATEGORICAL_REFERENCE,
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
