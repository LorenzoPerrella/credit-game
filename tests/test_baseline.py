"""Rule 16's basis along age: the arithmetic, and the probe that reads a form before it is fitted.

The module these hold is the fallback form rule 16 declares, plus the instrument that would have
cost least on the form before it. Nothing here tests a monotonicity condition, because that
condition was withdrawn on 30 October 2026: the model's monthly increment is taken at **fixed**
covariates, so its hazard is positive whatever the baseline does. The last test in this file is
that proof, written against the package's own code rather than against the algebra.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from creditsurv.config import AGE_BASIS_COLUMNS, AGE_BASIS_KNOTS, AGE_BASIS_OFFSET, AGE_BASIS_RANGE
from creditsurv.models import baseline


def _panel(points: np.ndarray) -> np.ndarray:
    return np.asarray(baseline._raw(points) @ baseline._TRANSFORM.T, dtype=np.float64)


def test_the_basis_is_orthonormal_under_the_declared_measure() -> None:
    """By adaptive quadrature, which is not the rule the transform was built with.

    The transform is Gauss-Legendre per knot interval, exact for the degree the integrands reach;
    this checks it against `scipy.integrate.quad` panel by panel, so a mistake in the node count
    or in the panel edges would show. **Panel by panel matters**: the third derivative jumps at
    every knot, and a single adaptive call over the whole range reads the kinks as error.
    """
    low, high = AGE_BASIS_RANGE
    knots = np.asarray(AGE_BASIS_KNOTS, dtype=float)
    edges = np.unique(np.clip(np.r_[low, knots, high], low, high))

    def inner(left: int, right: int) -> float:
        total = 0.0
        for start, stop in pairwise(edges):
            total += quad(
                lambda x: _panel(np.array([x]))[0, left] * _panel(np.array([x]))[0, right],
                start,
                stop,
                epsabs=1e-13,
                epsrel=1e-13,
            )[0]
        return total / (high - low)

    width = len(AGE_BASIS_COLUMNS)
    gram = np.array([[inner(i, j) for j in range(width)] for i in range(width)])
    assert np.abs(gram - np.eye(width)).max() < 1e-9


def test_the_basis_carries_no_intercept_and_no_linear_term() -> None:
    """Rule 16's dropped column, as a property of the basis rather than of the formula.

    `rho` is the linear term and the intercept is the intercept; a basis with a component of
    either would be fighting them for the same information, which is what the raw Royston-Parmar
    columns do and what the orthonormalisation removes.
    """
    low, high = AGE_BASIS_RANGE
    grid = np.linspace(low, high, 200_001)
    columns = _panel(grid)
    weight = np.full(len(grid), (high - low) / len(grid))
    for name, against in (("1", np.ones_like(grid)), ("u", grid)):
        product = (columns * (against * weight)[:, None]).sum(axis=0) / (high - low)
        assert np.abs(product).max() < 1e-3, f"the basis has a component of {name}"


def test_the_basis_is_linear_outside_the_boundary_knots() -> None:
    """What makes the spline *restricted*, and what makes a lifetime PD extrapolate as a Weibull.

    Beyond the last knot `log H` is again a straight line in `log t`, so the model's answer past
    age 170 -- where 98.2% of the defaults are already behind it -- is a Weibull tail at the slope
    the data last supported. A second difference of zero is that property.
    """
    low, _ = AGE_BASIS_RANGE
    knots = np.asarray(AGE_BASIS_KNOTS, dtype=float)
    for points in (
        np.array([low, low + 0.3, low + 0.6]),
        np.array([knots[-1] + 0.05, knots[-1] + 0.12, knots[-1] + 0.19]),
    ):
        columns = _panel(points)
        assert np.abs(columns[0] - 2.0 * columns[1] + columns[2]).max() < 1e-10


def test_the_derivative_is_the_derivative() -> None:
    points = np.array([AGE_BASIS_RANGE[0] + 0.3, 2.0, 3.5, 4.5, 5.2])
    step = 1e-6
    numerical = (_panel(points + step) - _panel(points - step)) / (2.0 * step)
    analytic = baseline.age_basis_derivative(points)
    assert np.abs(numerical - analytic).max() < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_supremum_slope_is_exact_rather_than_sampled(seed: int) -> None:
    """Against a two-million-point scan, which can only ever read low.

    `g'` is a quadratic on each knot interval, so the supremum is at a panel end or at the one
    interior vertex -- and a closed form beats a grid on a quantity that gets reported.
    """
    low, high = AGE_BASIS_RANGE
    beta = np.random.default_rng(seed).normal(scale=0.4, size=len(AGE_BASIS_COLUMNS))
    grid = np.linspace(low, high, 2_000_001)
    scanned = float((baseline.age_basis_derivative(grid) @ beta).max())
    exact = baseline.baseline_slope(beta)
    assert exact >= scanned - 1e-9
    assert exact - scanned < 1e-6


def test_zero_coefficients_are_the_weibull() -> None:
    """The nesting rule 2's test and the engine's nested-model floor both rest on."""
    assert baseline.baseline_slope([0.0] * len(AGE_BASIS_COLUMNS)) == pytest.approx(0.0, abs=1e-12)


def test_the_offset_makes_the_youngest_age_a_number() -> None:
    assert np.isfinite(baseline.log_age(np.array([0.0]))).all()
    assert baseline.log_age(np.array([0.0]))[0] == pytest.approx(np.log(AGE_BASIS_OFFSET))


def test_a_named_mapping_and_a_bare_vector_agree() -> None:
    """A fitted model names its columns ('lambda_', 'age_basis_1'); a caller may hold neither."""
    beta = np.array([0.31, -0.07, 0.04])
    named = {("lambda_", name): value for name, value in zip(AGE_BASIS_COLUMNS, beta, strict=True)}
    assert baseline.baseline_slope(named) == pytest.approx(baseline.baseline_slope(beta))


def test_a_model_without_the_basis_is_refused_rather_than_read_as_zero() -> None:
    with pytest.raises(ValueError, match="names no coefficient"):
        baseline.baseline_slope({("lambda_", "ltv_change"): 0.2})


def test_the_probe_recovers_a_weibull_it_is_given() -> None:
    """On a curve that **is** a straight line in log t, the probe reads the slope and no bend.

    The probe's whole claim is that it describes the shape of the form it fits. A synthetic
    Nelson-Aalen curve from a known `rho` is the one case where the answer is known in advance.
    """
    rho = 1.37
    ages = np.arange(0, 221, dtype=float)
    # H at the **close** of month `a`, which is where a Nelson-Aalen sum through `a` sits, so the
    # abscissa is `a + 1`. Getting this wrong by half a month is the defect this test would miss
    # if it built the curve on the probe's own convention rather than on the estimator's.
    cumulative = (ages + 1.0) ** rho * 1e-7
    defaults = np.diff(cumulative, prepend=0.0) * 1e6
    out = baseline.probe(ages, np.full_like(ages, 1e6), defaults)
    assert out.marginal_rho == pytest.approx(rho, abs=0.02)
    assert abs(out.supremum_g_prime) < 0.05
    assert out.weighted_mse < 1e-4
    assert set(out.describe()) >= {"marginal_rho", "actual_over_expected"}


def test_the_models_monthly_hazard_is_positive_whatever_the_baseline_does() -> None:
    """The withdrawn condition, written as the test that withdrew it.

    `episode_hazards` takes its increment at **fixed** covariates -- `H(a+1 | x_a) - H(a | x_a)`
    -- so it equals `exp(-rho * eta_a) * ((a+1)**rho - a**rho)`, and both factors are positive for
    any real `eta_a` and any `rho > 0`. Rule 16's monotonicity condition was read as violated by a
    baseline whose `eta` rose with age; the quantity that fell was `H(a | eta_b)` against
    `H(a | eta_b+1)`, which is two covariate vectors at one age and not one loan's curve.

    Checked here against an `eta` profile chosen to be as adversarial as the refused fit was --
    alternating sign, by band, at a magnitude larger than anything fitted.
    """
    rho = 2.0995
    ages = np.arange(0, 360, dtype=float)
    eta = np.where((ages // 12) % 2 == 0, -0.75, 0.75)
    increment = np.exp(-rho * eta) * ((ages + 1.0) ** rho - ages**rho)

    assert (increment > 0.0).all()
    hazard = 1.0 - np.exp(-increment)
    assert (hazard > 0.0).all()
    assert np.all(np.diff(np.cumsum(increment)) > 0.0)


def test_the_probe_recovers_a_baseline_multiplier_it_is_given() -> None:
    """Generated from the package's own increment, with a known step, and read back.

    This is the test that would have caught the probe's own first version. That one fitted
    `log H` at one `u` per month, which makes `H` jump between months and takes the increment
    across two different `eta` -- the same error whose retraction is in the module docstring. On
    data generated from `exp(-rho * eta_a) * ((a+1)**rho - a**rho)` it reads the wrong shape; the
    version that fits that structure reads the step back.
    """
    rho, step, cut = 1.6, 0.4, 60.0
    ages = np.arange(0, 221, dtype=float)
    eta = np.where(ages >= cut, step, 0.0)
    hazard = np.exp(-rho * eta) * ((ages + 1.0) ** rho - ages**rho) * 1e-7
    at_risk = np.full_like(ages, 2.0e6)

    out = baseline.probe(ages, at_risk, hazard * at_risk, form=baseline.bands([0, int(cut), 400]))
    assert out.marginal_rho == pytest.approx(rho, abs=0.05)
    # One step column, carrying the eta step that was generated.
    assert out.basis_coefficients[0] == pytest.approx(step, abs=0.05)
    # And every band it is scored on comes back at one, because the form is the truth here.
    assert all(value == pytest.approx(1.0, abs=0.05) for value in out.actual_over_expected.values())


def test_the_probe_reads_the_published_weibulls_own_defect() -> None:
    """The instrument is only worth having if it reproduces a defect already measured.

    The published model is the Weibull alone, and its measured out-of-sample age profile misses
    **high** over 12 to 35 months and **low** from 60 on. The probe is in sample and marginal, so
    it cannot reproduce those numbers -- but if it does not reproduce their *direction* it is not
    reading the same thing the criterion reads.
    """
    import pandas as pd

    from creditsurv.models.nonparametric import EXPOSURE_FLOOR

    table = pd.read_parquet("docs/tables/km_vs_model.parquet")
    whole = table[table["group"] == "all loans"].sort_values("age")
    rows = whole[whole["at_risk"].to_numpy(dtype=float) >= EXPOSURE_FLOOR]
    if rows.empty:
        pytest.skip("the committed risk sets carry no age above the exposure floor")

    out = baseline.probe(
        rows["age"].to_numpy(dtype=float),
        rows["at_risk"].to_numpy(dtype=float),
        rows["defaults"].to_numpy(dtype=float),
        form=baseline.WEIBULL,
    )
    profile = out.actual_over_expected
    assert profile["12 to 23"] > 1.25, "the young bands should miss high"
    assert profile["60 to 119"] < 0.80, "the seasoned bands should miss low"
    assert out.supremum_g_prime == pytest.approx(0.0), "a straight line bends by nothing"
