"""The private lifelines calls this engine stands in for.

Four of them, all private, which is why ``tests/test_blocks.py`` fits the same rows both ways
and compares every number a report reads. Re-run it before trusting a different lifelines.

* ``fit_interval_censoring``'s regressor mapping and censoring attributes;
* ``_create_initial_point``'s univariate fit, which reads the rows only through their bounds and
  weights -- so it runs on the **distinct** bounds with the weights summed, a few hundred rows
  in place of millions, and the same likelihood;
* the warm start, which rescales a fitted ``params_`` into the space lifelines optimises in,
  where each coefficient is multiplied by its column's standard deviation;
* and the attributes a fitted model has to carry for ``summary``, ``AIC_`` and the predictions
  to work -- without the full-length copies of the training data lifelines attaches, which
  nothing in this project reads.
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import lifelines
import numpy as np
import pandas as pd
from lifelines import utils

from creditsurv.models.engine.storage import (
    _CONSTANT,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lifelines.fitters import ParametericAFTRegressionFitter

    from creditsurv.models.engine.contract import _Evaluator

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


def _seed_regressors(
    fitter: ParametericAFTRegressionFitter, formula: str, ancillary: str | bool | None
) -> dict[str, str]:
    """The parameter-to-formula mapping ``fit_interval_censoring`` builds for a formula."""
    regressors = {fitter._primary_parameter_name: formula}
    if isinstance(ancillary, str):
        fitter.model_ancillary = True
        regressors[fitter._ancillary_parameter_name] = ancillary
    elif ancillary is True or fitter.model_ancillary:
        fitter.model_ancillary = True
        regressors[fitter._ancillary_parameter_name] = formula
    else:
        regressors[fitter._ancillary_parameter_name] = "1"
    return regressors


def _initial_point(
    fitter: ParametericAFTRegressionFitter,
    columns: pd.MultiIndex,
    raw_std: np.ndarray,
    bounds: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """``ParametericAFTRegressionFitter._create_initial_point``, on the distinct bounds.

    lifelines fits the matching univariate model to every row's bounds and weight, and
    seeds the first constant column -- the scale intercept -- with the log of its
    parameter. Rows sharing their bounds contribute identically, so fitting the distinct
    bounds with their weights summed is the same likelihood on a few hundred rows.
    """
    constant_col = pd.Series(raw_std < _CONSTANT, index=columns).idxmax()
    univariate_class = getattr(lifelines, fitter._class_name.replace("AFT", ""), None)
    if univariate_class is None:
        return {
            name: np.zeros(int((columns.get_level_values(0) == name).sum()))
            for name in fitter._fitted_parameter_names
        }

    univariate = univariate_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        univariate.fit_interval_censoring(
            bounds["lower"].to_numpy(dtype=float),
            bounds["upper"].to_numpy(dtype=float),
            entry=bounds["entry"].to_numpy(dtype=float),
            weights=bounds["weight"].to_numpy(dtype=float),
        )
    fitter._ll_null_ = univariate.log_likelihood_

    # Parameter blocks in sorted order, which is how ``Index.groupby`` hands them over.
    parameters = columns.get_level_values(0)
    seeded: dict[str, np.ndarray] = {}
    for name in sorted(set(parameters)):
        covariates = columns[parameters == name].tolist()
        seeded[name] = np.zeros(len(covariates))
        if constant_col in covariates:
            value = getattr(univariate, name)
            seeded[name][covariates.index(constant_col)] = value if value <= 0 else np.log(value)
    return seeded


def _warm_start(
    seeded: dict[str, np.ndarray],
    columns: pd.MultiIndex,
    norm_std: pd.Series,
    params: pd.Series,
) -> dict[str, np.ndarray]:
    """A starting point from coefficients on their natural scale, a nested model's say.

    lifelines optimises each coefficient multiplied by its column's standard deviation,
    so a fitted ``params_`` is scaled back into that space before it can seed another
    fit. Columns the other model did not have keep lifelines' own seed.

    Backward elimination refits a model one covariate smaller at every step. On the whole
    population each fit is well over an hour, and the model it removes a covariate from is
    the best available guess at where the smaller one ends.
    """
    started = {name: values.copy() for name, values in seeded.items()}
    parameters = columns.get_level_values(0)
    for name, values in started.items():
        for position, key in enumerate(columns[parameters == name].tolist()):
            if key in params.index:
                values[position] = float(params[key]) * float(norm_std[key])
    return started


def _set_censoring(
    fitter: ParametericAFTRegressionFitter,
    names: tuple[str, str, str, str | None, str | None],
) -> None:
    """Tell the fitter what it is fitting, as ``fit_interval_censoring`` does."""
    lower_bound_col, upper_bound_col, event_col, entry_col, weights_col = names
    utils.CensoringType.set_censoring_type(fitter, utils.CensoringType.INTERVAL)
    fitter._time_fit_was_called = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    fitter.lower_bound_col = lower_bound_col
    fitter.upper_bound_col = upper_bound_col
    fitter.event_col = event_col
    fitter.entry_col = entry_col
    fitter.weights_col = weights_col
    fitter.robust = False


def _store(
    fitter: ParametericAFTRegressionFitter,
    columns: pd.MultiIndex,
    x: np.ndarray,
    value: float,
    curvature: np.ndarray,
    objective: _Evaluator,
    unflatten: Callable[[np.ndarray], dict[str, np.ndarray]],
) -> None:
    """What ``_fit_model`` and ``_fit`` set once the optimum is found."""
    params = unflatten(x)
    fitter.log_likelihood_ = -objective.total_weight * value
    fitter._hessian_ = objective.total_weight * curvature

    keys = list(fitter.regressors.keys())
    if keys != list(fitter._norm_std.index.get_level_values(0).unique()):
        message = "Parameter blocks and design columns are out of order."
        raise AssertionError(message)
    fitter.params_ = np.concatenate([params[key] for key in keys]) / fitter._norm_std
    fitter._compare_to_values = np.zeros_like(fitter.params_)
    fitter.variance_matrix_ = pd.DataFrame(
        fitter._compute_variance_matrix(), index=columns, columns=columns
    )
    # Without robust errors lifelines reads only the variance matrix here.
    fitter.standard_errors_ = fitter._compute_standard_errors(None, None, None, None, None)
    fitter.confidence_intervals_ = fitter._compute_confidence_intervals()


def _family(fitter: ParametericAFTRegressionFitter) -> str:
    """``WeibullAFTFitter`` -> ``weibull``, which is also the key in ``aft.FITTERS``."""
    name = type(fitter).__name__
    return name.removesuffix(_FAMILY_SUFFIX).lower()


#: The written-out kernel's name for each fitter, from the fitter's own class name.
_FAMILY_SUFFIX: Final = "AFTFitter"
