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
