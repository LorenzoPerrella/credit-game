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


# --------------------------------------------------------------------------------------
# Variable selection
# --------------------------------------------------------------------------------------

#: A covariate whose variance inflation exceeds this is carrying the same information
#: as the others already in the model. The conventional threshold, and the one `nmds`
#: uses.
VIF_THRESHOLD: Final = 10.0

#: A coefficient no more significant than this has not earned its place.
PVALUE_THRESHOLD: Final = 0.05

#: The sign each covariate's coefficient must take, on the accelerated-failure-time
#: scale where **positive lengthens survival and therefore lowers risk**.
#:
#: This is the piece of the `nmds` procedure most worth having. A covariate whose
#: coefficient comes out economically backwards is not salvaged by being significant
#: -- it is evidence that something else is wrong, usually collinearity -- so it is
#: eliminated on the sign alone. A model that says higher credit scores default sooner
#: fits its sample and will not survive the next one.
EXPECTED_SIGNS: Final[dict[str, int]] = {
    "fico_s": +1,  # better credit survives longer
    "orig_ltv": -1,  # more leverage fails sooner
    "orig_cltv": -1,
    "dti": -1,  # more debt burden fails sooner
    "cltv_drift": -1,  # leverage rising after origination fails sooner
    "unemp_gap": -1,  # unemployment above origination fails sooner
    "nfci_lagged": -1,  # tighter financial conditions fail sooner
    "mi_percent": +1,  # insured loans are underwritten against a stricter standard
}


def variance_inflation(
    frame: pd.DataFrame, columns: Sequence[str], *, weight: str | None = None
) -> pd.DataFrame:
    """Variance inflation factor for each covariate, largest first.

    ``1 / (1 - R²)`` from regressing each covariate on the others, weighted by
    exposure where a weight is given — unweighted on an aggregated panel would
    measure collinearity among *cells*, which is a property of the binning.

    Computed with least squares directly rather than through scikit-learn. `nmds`
    uses ``LinearRegression(normalize=True)``, which was removed in scikit-learn 1.2,
    so its implementation no longer runs; the method is worth taking, the code is not.
    """
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    weights = frame[weight].to_numpy(dtype=float) if weight else np.ones(len(frame), dtype=float)
    root = np.sqrt(weights)

    rows = []
    for index, name in enumerate(columns):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])

        coefficients, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
        residual = target - design @ coefficients
        weighted_mean = float((target * weights).sum() / weights.sum())

        residual_ss = float((weights * residual**2).sum())
        total_ss = float((weights * (target - weighted_mean) ** 2).sum())
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
        inflation = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else np.inf

        rows.append({"covariate": name, "vif": inflation, "tolerance": 1.0 - r_squared})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def stepwise_vif(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    weight: str | None = None,
    threshold: float = VIF_THRESHOLD,
    priority: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Drop the most inflated covariate, recompute, repeat.

    Returns the elimination log and the surviving covariates.

    ``priority`` names covariates to protect, most protected last — the convention
    `nmds` uses, and the reason it matters: when two covariates are collinear the
    procedure has no view on which one the model is *for*. Left alone it drops
    whichever happens to have the larger factor, which is arbitrary and unstable
    across samples. Stating the order makes the choice explicit and repeatable.
    """
    protected = {name: rank for rank, name in enumerate(priority)}
    surviving = list(columns)
    log: list[dict[str, object]] = []

    while len(surviving) > 1:
        inflation = variance_inflation(frame, surviving, weight=weight)
        worst = inflation.iloc[0]
        if float(worst["vif"]) <= threshold:
            break

        offenders = inflation[inflation["vif"] > threshold]["covariate"].tolist()
        # Among the offenders, drop the least protected. `offenders` is already
        # ordered by descending inflation, so an unprotected tie falls to the largest.
        victim = min(offenders, key=lambda name: protected.get(name, -1))

        log.append(
            {
                "step": len(log) + 1,
                "removed": victim,
                "vif": float(inflation.loc[inflation["covariate"] == victim, "vif"].iloc[0]),
                "remaining": len(surviving) - 1,
            }
        )
        surviving.remove(victim)

    return pd.DataFrame(log, columns=["step", "removed", "vif", "remaining"]), surviving


def univariate_screening(
    encoded: pd.DataFrame,
    candidates: Sequence[str],
    *,
    always_include: Sequence[str] = (),
    distribution: str = "weibull",
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Fit one model per candidate covariate and report its significance.

    A cheap first pass that removes covariates carrying nothing at all, before the
    multivariate fit has to carry them. Following `nmds`, the covariates in
    ``always_include`` are forced into every fit, so each candidate is judged on what
    it adds rather than on what it happens to proxy.

    It is a screen and not a decision: a covariate can be insignificant alone and
    matter in combination, which is why the surviving set still goes through backward
    elimination.
    """
    rows = []
    for name in candidates:
        terms = [*always_include, name]
        formula = " + ".join(terms)
        try:
            result = fit_aft(
                encoded, terms, formula, distribution=distribution, weights_col=weights_col
            )
        except Exception as error:  # a candidate that will not converge is a result
            rows.append(
                {
                    "covariate": name,
                    "coef": np.nan,
                    "p": np.nan,
                    "aic": np.nan,
                    "note": type(error).__name__,
                }
            )
            continue

        summary = result.fitter.summary
        key = ("lambda_", name)
        if key not in summary.index:
            continue
        rows.append(
            {
                "covariate": name,
                "coef": float(summary.loc[key, "coef"]),
                "p": float(summary.loc[key, "p"]),
                "aic": result.aic,
                "note": "",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["keep"] = table["p"] <= PVALUE_THRESHOLD
    return table.sort_values("p").reset_index(drop=True)


def _wrong_sign(covariate: str, coefficient: float) -> bool:
    expected = EXPECTED_SIGNS.get(covariate)
    return expected is not None and coefficient * expected < 0


def backward_elimination(
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    *,
    categorical: Sequence[str] = (),
    distribution: str = "weibull",
    weights_col: str | None = None,
    threshold: float = PVALUE_THRESHOLD,
) -> tuple[pd.DataFrame, list[str], FitResult]:
    """Remove one covariate at a time until every survivor earns its place.

    Two criteria, applied together, as `nmds` does. A coefficient no more significant
    than ``threshold`` goes; so does one whose sign is economically backwards, even
    when it is significant. The second matters more than it looks: a wrong sign is not
    a weak result but a symptom, usually of collinearity, and a model asserting that
    higher credit scores default sooner will fit this sample and no other.

    Returns the elimination log, the surviving covariates, and the final fit.
    """
    surviving = list(covariates)
    log: list[dict[str, object]] = []

    while True:
        terms = [*surviving, *categorical]
        formula = " + ".join([*surviving, *(f"C({name})" for name in categorical)])
        result = fit_aft(
            encoded, terms, formula, distribution=distribution, weights_col=weights_col
        )
        summary = result.fitter.summary.loc["lambda_"]

        worst: tuple[str, float, str] | None = None
        for name in surviving:
            if name not in summary.index:
                continue
            coefficient = float(summary.loc[name, "coef"])
            p_value = float(summary.loc[name, "p"])
            if _wrong_sign(name, coefficient):
                # A backwards sign outranks any p-value: it says the specification is
                # wrong, not that the evidence is thin.
                worst = (name, p_value, "wrong sign")
                break
            if p_value > threshold and (worst is None or p_value > worst[1]):
                worst = (name, p_value, "insignificant")

        if worst is None:
            return (
                pd.DataFrame(log, columns=["step", "removed", "p", "reason", "remaining"]),
                surviving,
                result,
            )

        name, p_value, reason = worst
        log.append(
            {
                "step": len(log) + 1,
                "removed": name,
                "p": p_value,
                "reason": reason,
                "remaining": len(surviving) - 1,
            }
        )
        surviving.remove(name)
        if not surviving:
            return (
                pd.DataFrame(log, columns=["step", "removed", "p", "reason", "remaining"]),
                surviving,
                result,
            )
