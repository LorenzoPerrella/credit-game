"""The registry of variable names: every name the code uses is in it, and reads as a label."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from creditsurv import names
from creditsurv.config import (
    CATEGORICAL_REFERENCE,
    ECONOMIC_DIMENSION,
    MACRO_CANDIDATES,
    MACRO_SERIES,
    ORDINAL,
    REFERENCE_SERIES,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
)
from creditsurv.data.aggregate import _CATEGORICAL, _SOURCE, PRODUCTION_EDGES
from creditsurv.features import BIN_EDGES, MACRO_DERIVED, MACRO_SOURCES
from creditsurv.models.lifetime_pd import ADVERSE
from creditsurv.models.procedure import BASE_CATEGORICAL, CANDIDATE_CATEGORICAL, LOAN_CONTINUOUS
from creditsurv.names import Kind


def _covariates() -> set[str]:
    return {
        *STATIC_CONTINUOUS,
        *ORDINAL,
        *TIME_VARYING_CONTINUOUS,
        *MACRO_CANDIDATES,
        *CATEGORICAL_REFERENCE,
        *ECONOMIC_DIMENSION,
        *MACRO_SOURCES,
        *MACRO_DERIVED,
        *BIN_EDGES,
        *PRODUCTION_EDGES,
        *_SOURCE,
        *_CATEGORICAL,
        *LOAN_CONTINUOUS,
        *BASE_CATEGORICAL,
        *CANDIDATE_CATEGORICAL,
    }


def _series() -> set[str]:
    return {
        *(spec.column for spec in (*MACRO_SERIES, *REFERENCE_SERIES)),
        *(source for sources in MACRO_SOURCES.values() for source in sources),
        *ADVERSE.shocks,
    }


@pytest.mark.parametrize("name", sorted(_covariates()))
def test_every_covariate_the_code_names_is_registered_under_its_current_name(name: str) -> None:
    assert name in names.VARIABLES, f"{name} is a former name or not registered"
    assert names.VARIABLES[name].kind in (Kind.LOAN, Kind.MACRO)


@pytest.mark.parametrize("name", sorted(_series()))
def test_every_macro_series_the_code_names_is_registered_under_its_current_name(name: str) -> None:
    assert name in names.VARIABLES, f"{name} is a former name or not registered"
    assert names.VARIABLES[name].kind is Kind.SERIES


_THEN: re.Pattern[str] = re.compile(r"THEN '([^']*)'")


@pytest.mark.parametrize("name", sorted(_CATEGORICAL))
def test_every_level_the_aggregation_produces_is_a_current_level(name: str) -> None:
    entry = names.VARIABLES[name]
    produced = set(_THEN.findall(_CATEGORICAL[name]))
    if name == "region":
        produced = {"Northeast", "Midwest", "South", "West", "Other"}

    for code in produced:
        assert code in {level.code for level in entry.levels}, f"{name}: {code} is not current"


def test_every_reference_level_is_a_level_of_its_variable() -> None:
    for name, reference in {**CATEGORICAL_REFERENCE, **BASE_CATEGORICAL}.items():
        assert reference in {level.code for level in names.VARIABLES[name].levels}


def test_labels_are_unique_among_what_a_reader_compares() -> None:
    for kind in (Kind.LOAN, Kind.MACRO, Kind.SERIES):
        counts = Counter(entry.label for entry in names.VARIABLES.values() if entry.kind is kind)
        assert [label for label, count in counts.items() if count > 1] == [], kind


def test_a_former_name_points_to_one_variable_of_each_kind() -> None:
    for kind in Kind:
        formers = [
            entry.former
            for entry in names.VARIABLES.values()
            if entry.kind is kind and entry.former is not None
        ]
        assert len(formers) == len(set(formers)), kind


def test_names_are_snake_case_and_no_name_is_another_variables_former_name() -> None:
    current = set(names.VARIABLES)
    for entry in names.VARIABLES.values():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", entry.name), entry.name
        assert entry.former not in current - {entry.name} or entry.kind is Kind.SERIES


def test_a_name_two_kinds_shared_resolves_to_the_covariate_unless_asked() -> None:
    assert names.variable("vix").name == "equity_volatility"
    assert names.variable("vix", kind=Kind.SERIES).name == "vix_index"
    assert names.label("term_spread") == "Yield curve slope"


def test_a_formula_term_reads_as_a_sentence() -> None:
    term = "C(has_mi, Treatment('N'))[T.Y]"

    assert names.term_label(term) == "Mortgage insurance: Insured (against Not insured)"
    assert names.term_label("cltv_drift") == "Loan-to-value change since origination"
    assert names.term_label("rho_") == "Shape"


def test_levels_read_under_their_current_and_their_former_codes() -> None:
    assert names.level_label("purpose", "refinance_cashout") == "Cash-out refinance"
    assert names.level_label("purpose", "cash_out_refinance") == "Cash-out refinance"
    assert names.level_label("first_time_buyer", "Y") == "First-time buyer"


def test_an_unknown_name_is_shown_rather_than_raised() -> None:
    assert names.label("some_new_column") == "some new column"
    assert names.level_label("purpose", "not mapped") == "not mapped"


def test_the_glossary_has_a_row_per_variable() -> None:
    rows = names.glossary()

    assert len(rows) == len(names.VARIABLES)
    assert all(row["Label"] and row["What it is"] for row in rows)
