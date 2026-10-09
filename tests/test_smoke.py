"""Smoke tests guarding the package skeleton and its quality gate."""

from __future__ import annotations

import creditsurv


def test_version_is_exposed() -> None:
    assert creditsurv.__version__ == "0.1.0"


def test_the_outcome_column_and_its_levels_are_declared_apart() -> None:
    """The CLI used to keep its own copy of `DEFAULT_CAUSE`, because importing the panel module
    put 0.85 s on every `creditsurv --help`, and a test held the two to each other.

    The copy is gone. The three states a cell's loan-months can end in are declared in
    `config`, which is below everything that needs them -- the command line, and
    `config.record_name`, which spells a record's filename by distribution and cause. The
    *column* stays in `data.panel`, where the columns are. What this checks is that the
    separation holds: the column's name is the panel's and its levels are the configuration's,
    and neither module declares the other's.
    """
    from creditsurv import config
    from creditsurv.data import panel

    assert config.CAUSES == (config.DEFAULT_CAUSE, config.PREPAYMENT_CAUSE)
    assert config.CENSORED not in config.CAUSES
    assert panel.OUTCOME == "outcome"
    assert not hasattr(config, "OUTCOME"), "the configuration declares the levels, not the column"
    # That nobody reads a level *through* the panel is enforced where it is stronger than here:
    # mypy's strict mode refuses an implicit re-export, so `panel.DEFAULT_CAUSE` fails the type
    # check even though the name is bound there -- the panel uses it as a default in four
    # signatures. A runtime assertion could not say that, and this one tried.
