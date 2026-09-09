"""Smoke tests guarding the package skeleton and its quality gate."""

from __future__ import annotations

import creditsurv


def test_version_is_exposed() -> None:
    assert creditsurv.__version__ == "0.1.0"
