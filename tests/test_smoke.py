"""Smoke tests guarding the package skeleton and its quality gate."""

from __future__ import annotations

import creditsurv


def test_version_is_exposed() -> None:
    assert creditsurv.__version__ == "0.1.0"


def test_the_cli_spells_the_default_cause_the_way_the_panel_does() -> None:
    """The CLI keeps its own copy because importing the panel module put 0.85 s on every
    `creditsurv --help`, and a copy is only safe while something checks it.
    """
    from creditsurv import cli
    from creditsurv.data import panel

    assert cli.DEFAULT_CAUSE == panel.DEFAULT_CAUSE
