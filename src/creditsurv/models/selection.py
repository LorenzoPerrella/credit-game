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

from creditsurv.data.panel import EVENT, duration_view
from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS, FitResult, Likelihood, fit_aft

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def marginal_comparison(panel: pd.DataFrame, *, weights_col: str | None = None) -> pd.DataFrame:
    """Rank univariate families on the loan-level marginal distribution.

    Cheap -- seconds, on a table of a few hundred rows -- and **it does not answer the
    question it appears to answer.** That is worth stating at length, because the
    obvious reading of this table is the wrong one.

    The marginal hazard is not the baseline hazard. On this panel it peaks at about
    forty-eight months of loan age, which looks like a seasoning curve and is not: it
    is 2008 to 2010. Every vintage meets the crisis at a different loan age, so pooling
    them smears a calendar event across the age axis and produces a hump belonging to
    the economy rather than to the loan.

    Slicing the data differently does not fix it, because the problem is structural.
    Calendar period, origination cohort and loan age satisfy ``period = cohort + age``
    **identically**, so no two of them can be held fixed while the third varies. Hold
    the calendar fixed and the ages are different cohorts: at June 2009 the hazard rises
    to 61 bp at twenty-three months and falls to 8 bp at eighty, and the twenty-three
    month old loans are the 2007 vintage while the eighty month old ones are 2002. The
    profile tracks vintage quality exactly as well as it tracks age.

    This is the age-period-cohort identification problem, and the consequence here is
    that **the shape of the baseline hazard is not identified non-parametrically at
    all**. It becomes identified only under a restriction, and the restriction this
    model makes is that calendar time enters through a handful of macroeconomic
    covariates rather than as a free period effect. Which family fits therefore cannot
    be separated from which covariates are in the model, and this table -- computed
    without any -- is answering a different question.

    What it is still good for: catching a family that is wrong by an order of
    magnitude, and giving the regression comparison something to disagree with.
    """
    loans = duration_view(panel, weights_col=weights_col)
    duration = loans["duration"]
    observed = loans[EVENT].astype(bool)
    weights = None if weights_col is None else loans[weights_col]

    rows = []
    for name, factory in UNIVARIATE_FITTERS.items():
        fitter = factory()
        fitter.fit(duration, event_observed=observed, weights=weights)
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
    fitted: FitResult | None = None,
    fit: Callable[..., FitResult] = fit_aft,
    fits: dict[str, FitResult] | None = None,
) -> pd.DataFrame:
    """Compare regression fits on identical episodes.

    AIC here is computed on *episodes*, not loans, because that is the unit the
    likelihood sums over. It therefore ranks distributions on one panel and means
    nothing across different panel constructions -- comparing an interval-censored
    episode panel against a loan-level right-censored one by AIC is not a
    comparison at all.

    ``fitted`` supplies a model already estimated on this panel, and is reused instead
    of refitting its distribution. On a table of this size a fit is hours, so silently
    recomputing a model the caller already holds is not a small waste. ``fit`` is how the
    others are estimated -- :func:`fit_aft`, or a cached version of it -- and ``fits``, when
    given, receives every family's fit, so a caller can hold them against Kaplan-Meier
    without estimating any of them twice.

    Each family is also held to the expected signs. A family that fits worse *and* points a
    declared prior the wrong way is rejected twice, for independent reasons: the
    validation's log-logistic fit on the specification before it was 623,126 AIC points
    behind the Weibull and turned ``original_ltv`` around.
    """
    rows = []
    for distribution in distributions:
        if fitted is not None and fitted.distribution == distribution:
            result = fitted
        else:
            result = fit(
                encoded,
                covariates,
                formula,
                distribution=distribution,
                likelihood=likelihood,
                weights_col=weights_col,
            )
        if fits is not None:
            fits[distribution] = result
        rows.append(
            {
                "distribution": distribution,
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "n_episodes": result.n_episodes,
                "seconds": round(result.elapsed_seconds, 2),
                "signs_against_prior": ", ".join(signs_against_prior(result)),
            }
        )

    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table


def signs_against_prior(result: FitResult) -> list[str]:
    """The covariates whose coefficient points against its declared expected sign.

    Read on the scale parameter, where every AFT family in the comparison puts its
    covariates and where a positive coefficient lengthens survival. Only the signs declared
    in ``EXPECTED_SIGNS`` count; a covariate without one cannot be against it.
    """
    summary = result.fitter.summary.loc[result.fitter._primary_parameter_name]
    return [
        str(name)
        for name, coefficient in summary["coef"].items()
        if EXPECTED_SIGNS.get(str(name), 0) * float(coefficient) < 0
    ]


def exponential_is_rejected(result: FitResult) -> dict[str, float]:
    """Test the exponential against the Weibull, at no cost, from a fit in hand.

    The exponential is a Weibull with shape one, so it **nests** -- and a nested
    hypothesis does not need its own fit. Testing it is a Wald test on a parameter the
    Weibull fit has already estimated, where an AIC comparison would have cost a second
    full estimation to reach a weaker conclusion.

    lifelines parameterises the shape as ``log rho``, so the constant-hazard hypothesis
    is exactly ``log rho = 0`` and the test reads straight off the coefficient table.
    On the whole population the statistic runs to the hundreds, which is not a close call;
    ``docs/reports/methodology.md`` gives the run's own value, and is the only place a
    number for it is written down.

    The wider point is worth keeping: **where families nest, compare them by a test
    rather than by information criteria**. Only families that do not nest --
    Weibull against log-normal against log-logistic -- genuinely require a fit each.
    """
    if result.distribution != "weibull":
        message = f"The exponential nests inside the Weibull, not the {result.distribution}."
        raise ValueError(message)

    params = result.fitter.params_
    log_rho = float(params.loc[("rho_", "Intercept")])
    standard_error = float(result.fitter.standard_errors_.loc[("rho_", "Intercept")])
    statistic = log_rho / standard_error
    return {
        "log_rho": log_rho,
        "rho": float(np.exp(log_rho)),
        "standard_error": standard_error,
        "z": statistic,
        "p_value": float(2.0 * stats.norm.sf(abs(statistic))),
    }


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


#: The covariate the shape test lets into the shape parameter. ``occupancy`` is the one
#: stratum whose survival curves cross -- investor loans default faster early and slower
#: late, 0.9522 against 0.9533 surviving at 60 months and 0.9258 against 0.9246 at 91 --
#: and no scale factor maps one such curve onto the other, so it is where a shape that
#: varies has something to find. The report used to relax the first covariate of the
#: specification, ``credit_score``, and so answered a question nobody had asked.
SHAPE_COVARIATE: Final = "occupancy"


def shape_formula(covariates: Sequence[str], references: dict[str, str]) -> tuple[str, str]:
    """The covariate the shape test relaxes, and the ancillary formula that relaxes it.

    ``SHAPE_COVARIATE``, against its treatment reference, when the specification has it;
    the first covariate otherwise, so that a specification without it is still tested.
    """
    name = SHAPE_COVARIATE if SHAPE_COVARIATE in covariates else covariates[0]
    if name in references:
        return name, f"C({name}, Treatment('{references[name]}'))"
    return name, name


def shape_depends_on_covariates(
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    ancillary_formula: str,
    *,
    distribution: str = "weibull",
    likelihood: Likelihood = Likelihood.INTERVAL_CENSORED,
    weights_col: str | None = None,
    fitted: FitResult | None = None,
    fit: Callable[..., FitResult] = fit_aft,
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

    ``fitted`` is the restricted model, when the caller already has it. It is the
    default specification by construction -- same panel, same formula, no ancillary --
    so refitting it is pure duplication.
    """
    restricted = fitted or fit(
        encoded,
        covariates,
        formula,
        distribution=distribution,
        likelihood=likelihood,
        weights_col=weights_col,
    )
    # Started from the restricted model. The full one is the same scale coefficients with
    # shape coefficients added at zero, which is where Newton converges in a few steps --
    # 0.7 minutes against 4.2 from cold on four quarters of the book -- instead of SLSQP's
    # hundred-odd evaluations from nothing, 91 minutes on the whole training half.
    full = fit(
        encoded,
        covariates,
        formula,
        distribution=distribution,
        likelihood=likelihood,
        weights_col=weights_col,
        ancillary=ancillary_formula,
        initial_point=restricted.fitter.params_,
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
    "credit_score": +1,  # better credit survives longer
    "original_ltv": -1,  # more leverage fails sooner
    "original_cltv": -1,
    "debt_to_income": -1,  # more debt burden fails sooner
    "ltv_change": -1,  # leverage rising after origination fails sooner
    "unemployment_change": -1,  # unemployment above origination fails sooner
    "financial_conditions": -1,  # tighter financial conditions fail sooner
    "insurance_coverage": +1,  # insured loans are underwritten against a stricter standard
    # Macro candidates. Only where theory actually commits to a direction: a
    # covariate listed here with no clear prior would be eliminated for disagreeing
    # with a guess, which is worse than not testing it.
    "equity_volatility": -1,  # high implied volatility is a stressed economy
    # Volatility risen since origination is stress the loan was not written in.
    "volatility_change": -1,
    "house_price_growth": +1,  # rising house prices build equity
    #
    # ``mortgage_rate_decline`` and ``policy_rate_change`` were briefly given revised signs here, on
    # the strength of their marginal orderings and a mechanism about fixed-rate books.
    # That revision is **retracted**: conditional on the rest of the specification both
    # effects are inside the noise, and the marginal ordering that justified it was the
    # macro cycle. They are eliminated rather than re-signed -- see ``config.ELIMINATED``.
}

#: Covariates deliberately left out of ``EXPECTED_SIGNS``, with the reason. Listed so
#: the omission reads as a decision rather than an oversight.
#:
#: ``mortgage_rate_decline`` is the near miss. The sign above is the dominant channel -- rates
#: below the note rate mean refinancing is available and the payment burden is
#: easier -- but the opposite channel is real: the borrowers who *cannot* refinance
#: when everyone else can are adversely selected, and they are the ones left in the
#: book. The constraint is kept because the first channel dominates in the
#: literature, and this note is here because it is a prior, not a finding.
AMBIGUOUS_SIGNS: Final[dict[str, str]] = {
    "yield_curve_slope": "a steep curve is both cheap short funding and an expected slowdown",
    "inflation_rate": "erodes the real debt, squeezes the real income",
    "inflation_change": "the same two channels, measured against the loan's own start",
    "debt_to_income": "kept as negative, but it is measured at origination and never updated",
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

    The regressions are posed on the weighted covariance matrix rather than on the rows.
    With an intercept they are the same least-squares problems, and the matrix is added
    up a block at a time by :func:`weighted_covariance`.
    """
    return _inflation(weighted_covariance(frame, columns, weight=weight), columns)


#: Rows added up at a time when accumulating weighted cross-products. The moments are
#: sums, so the block size changes the memory and nothing else.
MOMENT_BLOCK_ROWS: Final = 5_000_000


def weighted_covariance(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    weight: str | None = None,
    rows: int = MOMENT_BLOCK_ROWS,
) -> pd.DataFrame:
    """Exposure-weighted covariance of ``columns``, added up a block at a time.

    A weight total, the weighted sums and the weighted cross-products are everything a
    variance inflation factor or a correlation needs, and all three are sums. The first
    version copied the covariates out whole and then again for every covariate it
    regressed: on the training half of the exact key, tens of gigabytes to produce a
    matrix a dozen entries wide.
    """
    names = list(columns)
    total = 0.0
    first = np.zeros(len(names))
    second = np.zeros((len(names), len(names)))
    for start in range(0, len(frame), rows):
        block = frame.iloc[start : start + rows]
        values = block.loc[:, names].to_numpy(dtype=float)
        weights = block[weight].to_numpy(dtype=float) if weight else np.ones(len(block))
        total += float(weights.sum())
        first += weights @ values
        second += values.T @ (values * weights[:, None])
    mean = first / total
    covariance = second / total - np.outer(mean, mean)
    return pd.DataFrame(covariance, index=names, columns=names)


def inflation_from_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """The VIF of every covariate in a covariance or correlation matrix, largest first.

    For a matrix computed once and saved -- ``creditsurv select`` writes the correlation of
    every candidate -- so the factors can be read back without the rows. They do not depend
    on the scale of the covariates, so a correlation matrix gives what a covariance would.
    """
    return _inflation(covariance, [str(name) for name in covariance.columns])


def _inflation(covariance: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The VIF of each of ``columns``, from their covariance, largest first.

    ``R²`` of a covariate on the others is the share of its variance that their
    covariances explain, which is what the regression on the rows computes.
    """
    names = list(columns)
    matrix = covariance.loc[names, names].to_numpy(dtype=float)
    rows = []
    for index, name in enumerate(names):
        variance = float(matrix[index, index])
        others = [position for position in range(len(names)) if position != index]
        r_squared = 0.0
        if others and variance > 0.0:
            cross = matrix[others, index]
            explained, *_ = np.linalg.lstsq(matrix[np.ix_(others, others)], cross, rcond=None)
            r_squared = min(max(float(cross @ explained) / variance, 0.0), 1.0)
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
    # Once: every later step's covariance is a submatrix of this one, so no step reads
    # the rows again.
    covariance = weighted_covariance(frame, columns, weight=weight)

    while len(surviving) > 1:
        inflation = _inflation(covariance, surviving)
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
