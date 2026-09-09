"""Choosing the distributional form, and judging how well it fits.

The plan for this module was to lean on the generalised gamma, which nests the
exponential, Weibull, gamma, log-normal and inverse-Weibull families, so that the
single parameter ``lambda`` *tests* which family the data supports instead of
ranking candidates by AIC. That is not available here:

* ``GeneralizedGammaRegressionFitter`` fails to converge on the episode panel
  under every remedy tried -- penalties from 0.001 to 0.1, durations rescaled,
  and both L-BFGS-B and SLSQP.
* ``GeneralizedGammaFitter`` on loan-level data nominally converges but returns a
  singular Hessian: standard errors are NaN, lifelines warns against trusting the
  parameters, and it estimates ``lambda = 4.04`` where the generating process is
  Weibull (``lambda = 1``).

An unusable test is worse than no test, so selection rests on four weaker but
sound layers instead:

1. AIC across univariate marginal fits, which do converge with usable variances.
2. AIC across the regression fits that converge on the episode panel.
3. A likelihood ratio test for a covariate-dependent shape, which is a genuine
   nested comparison within the Weibull family.
4. The Kaplan-Meier overlay in :mod:`creditsurv.models.nonparametric`, which is
   distribution-free and therefore the strongest evidence available that the
   imposed shape is not contradicted by the data.

Out-of-time performance breaks any remaining tie, because every criterion above is
in-sample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from lifelines import (
    ExponentialFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)
from scipy import stats

from creditsurv.data.panel import EVENT, to_loan_level
from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS, FitResult, Likelihood, fit_aft

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lifelines.fitters import ParametricUnivariateFitter

#: Univariate families used for the marginal shape check.
#:
#: The generalised gamma is excluded deliberately: see the module docstring.
UNIVARIATE_FITTERS: Final[dict[str, type[ParametricUnivariateFitter]]] = {
    "exponential": ExponentialFitter,
    "weibull": WeibullFitter,
    "lognormal": LogNormalFitter,
    "loglogistic": LogLogisticFitter,
}


def marginal_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank univariate families on the loan-level marginal distribution.

    A covariate-free check on the shape of the baseline hazard. It cannot decide
    the final model -- covariates change which family fits best -- but it is cheap
    and it catches a badly wrong choice before any regression is attempted.
    """
    loans = to_loan_level(panel)
    duration = loans["duration"]
    observed = loans[EVENT].astype(bool)

    rows = []
    for name, factory in UNIVARIATE_FITTERS.items():
        fitter = factory()
        fitter.fit(duration, event_observed=observed)
        rows.append(
            {
                "distribution": name,
                "log_likelihood": float(fitter.log_likelihood_),
                "aic": float(fitter.AIC_),
                "n_parameters": len(fitter._fitted_parameter_names),
            }
        )

    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table


def distribution_comparison(
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    *,
    distributions: Sequence[str] = CONVERGENT_DISTRIBUTIONS,
    likelihood: Likelihood = Likelihood.INTERVAL_CENSORED,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Compare regression fits on identical episodes.

    AIC here is computed on *episodes*, not loans, because that is the unit the
    likelihood sums over. It therefore ranks distributions on one panel and means
    nothing across different panel constructions -- comparing an interval-censored
    episode panel against a loan-level right-censored one by AIC is not a
    comparison at all.
    """
    rows = []
    for distribution in distributions:
        result = fit_aft(
            encoded,
            covariates,
            formula,
            distribution=distribution,
            likelihood=likelihood,
            weights_col=weights_col,
        )
        rows.append(
            {
                "distribution": distribution,
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "n_episodes": result.n_episodes,
                "seconds": round(result.elapsed_seconds, 2),
            }
        )

    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table


def likelihood_ratio_test(
    restricted_log_likelihood: float,
    full_log_likelihood: float,
    degrees_of_freedom: int,
) -> tuple[float, float]:
    """Return the test statistic and p-value for a nested comparison."""
    if degrees_of_freedom <= 0:
        message = "The full model must have more parameters than the restricted one."
        raise ValueError(message)
    statistic = 2.0 * (full_log_likelihood - restricted_log_likelihood)
    p_value = float(stats.chi2.sf(max(statistic, 0.0), degrees_of_freedom))
    return float(statistic), p_value


def shape_depends_on_covariates(
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    ancillary_formula: str,
    *,
    distribution: str = "weibull",
    likelihood: Likelihood = Likelihood.INTERVAL_CENSORED,
) -> pd.DataFrame:
    """Test whether the hazard's shape varies with covariates.

    The default model lets a covariate move *when* default happens while leaving
    the shape of the hazard over the life of the loan alone. That is an
    assumption, and a testable one: letting the shape parameter depend on
    covariates gives a strictly larger model, and a likelihood ratio test says
    whether the extra parameters earn their place.

    It matters for lifetime PD specifically. If the shape genuinely varies, the
    term structure of default differs by loan rather than merely shifting, and a
    single shape misstates the timing of losses even when it gets the total right.
    """
    restricted = fit_aft(
        encoded, covariates, formula, distribution=distribution, likelihood=likelihood
    )
    full = fit_aft(
        encoded,
        covariates,
        formula,
        distribution=distribution,
        likelihood=likelihood,
        ancillary=ancillary_formula,
    )

    added = int(full.fitter.params_.shape[0] - restricted.fitter.params_.shape[0])
    statistic, p_value = likelihood_ratio_test(
        restricted.log_likelihood, full.log_likelihood, added
    )
    return pd.DataFrame(
        [
            {
                "restricted_log_likelihood": restricted.log_likelihood,
                "full_log_likelihood": full.log_likelihood,
                "added_parameters": added,
                "statistic": statistic,
                "p_value": p_value,
                "restricted_aic": restricted.aic,
                "full_aic": full.aic,
            }
        ]
    )


def cox_snell_residuals(
    result: FitResult,
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    *,
    sample_size: int = 2000,
    seed: int = 0,
) -> pd.Series:
    """Cox-Snell residuals for a sample of the fitted episodes.

    If the model is correct these behave like a unit-exponential sample, so their
    own cumulative hazard traces the 45-degree line. Unlike an AIC ranking, that
    is a statement about the fit which does not depend on choosing a comparison
    family.

    Sampled rather than exhaustive: the cumulative hazard has to be evaluated at a
    different time for every row, and lifelines predicts a full row-by-time grid,
    so the full panel would mean a matrix with hundreds of thousands of columns. A
    few thousand rows is ample for the diagnostic, which is read as a plot.
    """
    rows = encoded
    if len(rows) > sample_size:
        rows = rows.sample(sample_size, random_state=seed)

    frame = rows.loc[:, list(covariates)]
    times = rows["lower_bound"].to_numpy(dtype=float)
    hazard = np.array(
        [
            float(
                result.fitter.predict_cumulative_hazard(
                    frame.iloc[[position]], times=[times[position]]
                ).to_numpy()[0, 0]
            )
            for position in range(len(frame))
        ]
    )
    return pd.Series(hazard, index=rows.index, name="cox_snell")
