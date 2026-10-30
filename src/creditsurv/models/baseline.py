"""Rule 16's flexible baseline along age: the basis, and the probe that reads a form cheaply.

The Weibull writes `H = exp(rho * (log t - eta))`, so `log H` is a **straight line** in `log t`
with slope `rho`. On this book that quantity is not constant: off the Nelson-Aalen curve of the
estimation table it reads 1.67 at six months of age, **1.81** at twelve, 1.76 at twenty-four,
1.52 at thirty-six, 1.25 at sixty, 1.15 at eighty-four and **1.10** at a hundred and twenty. It
does not wobble around a constant -- it halves. Fitting the straight line costs **2.6x** the
exposure-weighted mean square error of a single bend, 0.283 against 0.108, and one parameter is
being asked to be two numbers. That is rule 16's premise and nothing here has disturbed it.

So the baseline gets a function of the age on the **scale** side, which rule 16 admits because
`period = cohort + age` already puts the age in the calendar key -- the same identity rule 14
invokes to keep time-varying covariates out of the shape. Rule 16 fixes the order the forms are
tried in: **bands first**, by rule 12's mechanism, and a restricted cubic basis only if bands do
not reach section 5's out-of-sample age criterion. This module builds the basis for that fallback
and the probe that reads any declared form in milliseconds.

**What this module does not do, and the correction that says why.** An earlier version carried a
monotonicity *certificate*, on the argument that a cumulative hazard which falls is not one. The
condition it tested was withdrawn on 30 October 2026, because the model's monthly hazard is
non-negative for **every** `eta` profile by construction: `episode_hazards`
(`models/aft.py:492-493`) evaluates the cumulative hazard at `a` and `a + 1` with the covariates
the row itself carries, so the increment is `exp(-rho * eta_a) * ((a+1)**rho - a**rho)` and both
factors are positive. This model is an AFT with **time-varying covariates**, integrated month by
month at fixed covariates within each month, and `moratorium_regime`, `ltv_change` and
`unemployment_change` have made `eta` step between consecutive months since they entered it. The
age is no different from them. :func:`baseline_slope` survives as a **description** of a form's
implied shape -- it is the quantity the Weibull holds constant -- and not as a gate.

**What is true about the banded form**: it makes the **hazard** discontinuous in age, 23.8 per
100,000 loan-months at age 35 against 17.4 at 36 through the fitted model, so a term structure
built from it has kinks at ages a reporting grouper chose. That prefers a smooth basis on grounds
of smoothness, which decides a tie rather than refusing a form.

**The basis is orthonormalised, and it has to be.** The raw Royston-Parmar curvature columns carry
exposure-weighted **VIF 2,142 / 12,594 / 4,671** on this book, with a correlation condition number
of **57,402** -- and `selection.stepwise_vif` ranks protection rather than respecting it, so step 6
would delete two of the three columns of one function. The transform here is computed from the
declared knots alone by exact quadrature and reads **no data**: it is a change of basis inside a
fixed column span, so it cannot change the function, the likelihood or any prediction -- only the
coordinates and the conditioning the polish's `lstsq` and `pinv` depend on. Measured, it brings
the same VIFs to **1.40 / 1.12 / 1.37** and the condition number to **3.37**.

**The linear column is dropped**, which is rule 16's first declared condition. With `rho` free the
Weibull's `rho` *is* the linear term: `span{1, u, v_1, v_2, v_3}` is reached exactly either way,
and carrying `u` as well would leave the pair identified only through the `O(1/age)` difference
between `log(age)` and `log(age + 1/2)`. So the basis is spanned by the curvature columns and zero
coefficients are the Weibull, exactly nested, which is what makes the likelihood-ratio test and
the nested-model floor apply unchanged.

**And a restricted basis extrapolates as a Weibull**, linear beyond its boundary knots by
construction, so beyond age 170 -- where 98.2% of the defaults are already behind it -- `log H` is
again a straight line in `log t` with the slope the data last supported. A lifetime PD runs to 360
months, and that is the whole of what the model would say out there.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from creditsurv.config import (
    AGE_BASIS_COLUMNS,
    AGE_BASIS_KNOTS,
    AGE_BASIS_OFFSET,
    AGE_BASIS_RANGE,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

#: Gauss-Legendre nodes per knot interval for the orthonormalising quadrature.
#:
#: The inner products are of two piecewise cubics, so degree six, and an `n`-node rule is exact to
#: degree `2n - 1`. Four nodes would do it; six is taken because the cost is a hundred
#: multiplications once per process and being exact rather than nearly exact is what lets this
#: transform be called data-free and reproducible from the declared constants.
_QUADRATURE_NODES: int = 6


def log_age(ages: np.ndarray) -> np.ndarray:
    """`u = log(age + 1/2)`, the midpoint of the month the row is at risk over.

    The offset is not a tuned parameter: at age zero `log(age)` is not a number, the interval the
    row covers is `(age, age + 1]`, and its midpoint is the one point in it that is not an
    endpoint. Any other offset would be a choice; this one is the interval's own centre.
    """
    return np.log(np.asarray(ages, dtype=np.float64) + AGE_BASIS_OFFSET)


def _truncated_power(u: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Royston-Parmar's natural cubic spline basis, curvature columns only.

    `z_j(u) = (u - k_j)_+^3 - lam_j (u - k_1)_+^3 - (1 - lam_j)(u - k_m)_+^3` with
    `lam_j = (k_m - k_j) / (k_m - k_1)`, for each interior knot `k_j`. The two correction terms are
    what make the spline *restricted*: they force the second derivative to zero at both boundary
    knots, so the function is linear outside them.
    """
    k = np.asarray(knots, dtype=np.float64)
    first, last = k[0], k[-1]
    columns = []
    for interior in k[1:-1]:
        weight = (last - interior) / (last - first)
        columns.append(
            np.clip(u - interior, 0.0, None) ** 3
            - weight * np.clip(u - first, 0.0, None) ** 3
            - (1.0 - weight) * np.clip(u - last, 0.0, None) ** 3
        )
    return np.column_stack(columns)


def _truncated_power_derivative(u: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """`dz_j/du`: piecewise quadratic inside the boundary knots, constant outside them."""
    k = np.asarray(knots, dtype=np.float64)
    first, last = k[0], k[-1]
    columns = []
    for interior in k[1:-1]:
        weight = (last - interior) / (last - first)
        columns.append(
            3.0
            * (
                np.clip(u - interior, 0.0, None) ** 2
                - weight * np.clip(u - first, 0.0, None) ** 2
                - (1.0 - weight) * np.clip(u - last, 0.0, None) ** 2
            )
        )
    return np.column_stack(columns)


def _raw(u: np.ndarray) -> np.ndarray:
    """`[1, u, z_2, ..., z_{m+1}]`, the span the orthonormal basis is a rotation of."""
    return np.column_stack([np.ones_like(u), u, _truncated_power(u, AGE_BASIS_KNOTS)])


def _raw_derivative(u: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.zeros_like(u),
            np.ones_like(u),
            _truncated_power_derivative(u, AGE_BASIS_KNOTS),
        ]
    )


def _gram() -> np.ndarray:
    """The L2 Gram matrix of `[1, u, z_2, ...]` on the declared range, by exact quadrature.

    Uniform measure, integrated knot interval by knot interval so that no panel straddles a point
    where the third derivative jumps -- within a panel every integrand is a polynomial of degree
    six and the rule is exact. The measure is uniform rather than the exposure's own, because an
    exposure-weighted transform would read the estimation table and this one must be reproducible
    from the declared constants alone. The exposure-weighted alternative reaches a VIF of exactly
    1.0 against this one's 1.40, which is not worth a basis nobody can recompute from the rules.
    """
    low, high = AGE_BASIS_RANGE
    knots = np.asarray(AGE_BASIS_KNOTS, dtype=float)
    edges = np.unique(np.clip(np.r_[low, knots, high], low, high))
    nodes, weights = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    total = None
    for start, stop in pairwise(edges):
        middle, half = 0.5 * (start + stop), 0.5 * (stop - start)
        points = middle + half * nodes
        design = _raw(points)
        block = (design * (weights * half)[:, None]).T @ design
        total = block if total is None else total + block
    assert total is not None
    return np.asarray(total / (high - low), dtype=np.float64)


def _transform() -> np.ndarray:
    """The curvature rows of the inverse Cholesky factor: `v = T @ [1, u, z_2, ...]`.

    `G = L L^T`, so the functions `L^-1 [1, u, z...]` are orthonormal under the same inner product
    and the `k`-th is a combination of the first `k + 1` originals -- Gram-Schmidt in the order
    they are written. Dropping the first two rows drops the ones that *are* `1` and `u`, leaving a
    basis orthogonal to both: the intercept and `rho * u` keep what belongs to them, and `g`
    carries only curvature.
    """
    return np.asarray(np.linalg.inv(np.linalg.cholesky(_gram()))[2:, :], dtype=np.float64)


_TRANSFORM = _transform()


def age_basis(ages: np.ndarray) -> np.ndarray:
    """The declared basis columns, one row per age, in `AGE_BASIS_COLUMNS` order."""
    return np.asarray(_raw(log_age(ages)) @ _TRANSFORM.T, dtype=np.float64)


def age_basis_derivative(u: np.ndarray) -> np.ndarray:
    """`dv_k/du` at each `u`, in the same column order."""
    derivative = _raw_derivative(np.asarray(u, dtype=np.float64)) @ _TRANSFORM.T
    return np.asarray(derivative, dtype=np.float64)


def _slope_candidates() -> np.ndarray:
    """Every `u` at which `sup g'` can be attained, whatever the coefficients.

    `g'` is a quadratic on each knot interval and constant outside the boundary knots, so its
    extremum over the range is at a panel endpoint or at the one interior vertex of a panel. The
    vertices depend on the coefficients, so they are found per call; this returns the endpoints.
    """
    low, high = AGE_BASIS_RANGE
    return np.unique(np.clip(np.r_[low, np.asarray(AGE_BASIS_KNOTS, dtype=float), high], low, high))


def baseline_slope(coefficients: Sequence[float] | Mapping[Any, float] | np.ndarray) -> float:
    """`sup g'` over the declared age range, exactly -- a description of the form, not a gate.

    `g'` is how fast the baseline bends away from the straight line the Weibull would draw, so
    this is the largest departure a fitted form takes anywhere on the range. It is reported
    because it is readable and because `rho * (1 - g')` is the slope of `log H` the form implies
    at fixed covariates -- the quantity the Weibull holds constant and this book halves.

    **It is not a condition.** An earlier version refused a fit on `sup g' >= 1`, on the argument
    that the cumulative hazard would otherwise fall; the model's increment is taken at fixed
    covariates, so its hazard is positive whatever this reads. The module docstring carries the
    retraction.

    The supremum is exact rather than sampled, because `g'` is a quadratic on each panel:
    evaluating it at the panel's two ends and its midpoint determines that quadratic, and its
    vertex is then the only other place an extremum can sit.
    """
    beta = _ordered(coefficients)
    edges = _slope_candidates()
    points = [edges]
    for start, stop in pairwise(edges):
        middle = 0.5 * (start + stop)
        left, centre, right = (
            float((age_basis_derivative(np.array([x])) @ beta)[0]) for x in (start, middle, stop)
        )
        # The quadratic through three equally spaced points, in the panel's own coordinate.
        curvature = left - 2.0 * centre + right
        if abs(curvature) <= np.finfo(float).eps * max(1.0, abs(centre)) * 8:
            continue
        offset = 0.5 * (right - left) / curvature
        if -1.0 < offset < 1.0:
            points.append(np.array([middle - offset * 0.5 * (stop - start)]))
    everywhere = np.concatenate(points)
    return float(np.max(age_basis_derivative(everywhere) @ beta))


def _ordered(
    coefficients: Sequence[float] | Mapping[Any, float] | np.ndarray,
) -> np.ndarray:
    """The basis coefficients in `AGE_BASIS_COLUMNS` order, from a vector or a named mapping.

    A mapping is matched by **suffix**, because a fitted model names its columns
    `('lambda_', 'age_basis_1')` and a caller holding a bare vector should not have to know that.
    """
    if hasattr(coefficients, "items"):
        found = {}
        for key, value in coefficients.items():
            for name in AGE_BASIS_COLUMNS:
                if str(key).endswith(name) or str(key).endswith(f"{name}')"):
                    found[name] = float(value)
        missing = [name for name in AGE_BASIS_COLUMNS if name not in found]
        if missing:
            message = (
                f"The fitted model names no coefficient for {', '.join(missing)}, so rule 16's "
                "monotonicity certificate cannot be computed. A model without the basis is a "
                "Weibull, whose baseline is monotone for any positive shape; a model that should "
                "carry it and does not is a fit of a different specification."
            )
            raise ValueError(message)
        return np.array([found[name] for name in AGE_BASIS_COLUMNS], dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64).ravel()
    if beta.size != len(AGE_BASIS_COLUMNS):
        message = (
            f"{beta.size} coefficients for {len(AGE_BASIS_COLUMNS)} basis columns "
            f"({', '.join(AGE_BASIS_COLUMNS)})."
        )
        raise ValueError(message)
    return beta


@dataclass(frozen=True)
class Form:
    """A declared baseline form, as the probe needs to see it.

    `columns` must open with `[1, u]` so that the second coefficient is the marginal `rho` -- the
    one number a Weibull is forced to pick -- and whatever follows is the form's own freedom.
    `derivative` is of the same columns, and is what the slope reading is taken from.

    Three forms matter here and the probe reads all three: `WEIBULL` (the straight line, which is
    the model as published), `BASIS` (rule 16's restricted cubic fallback), and `bands` (rule 16's
    first choice, which the selection fitted). A step form has a derivative of zero inside every
    band, so its slope reading is `rho` and `sup g'` is zero -- which is true and is the point:
    **the slope cannot see a step**, and the thing a band form does to the hazard is invisible to
    the only summary a smooth form needs. That is the clearest statement of why the smooth form
    is preferred, and it falls out of the instrument rather than being asserted.
    """

    name: str
    columns: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]


WEIBULL: Final = Form(
    name="weibull",
    columns=lambda u: np.column_stack([np.ones_like(u), u]),
    derivative=lambda u: np.column_stack([np.zeros_like(u), np.ones_like(u)]),
)


def bands(edges: Sequence[int]) -> Form:
    """Rule 16's first form: one level per age band, against the first band.

    The columns are indicators of `age >= edge` rather than of membership, so that dropping the
    reference band is dropping a column and the remaining coefficients are steps rather than
    levels. It changes no fitted function -- the span is the same -- and makes the step at each
    boundary readable directly.
    """
    cuts = np.asarray(edges, dtype=np.float64)[1:]

    def columns(u: np.ndarray) -> np.ndarray:
        age = np.exp(u) - 1.0
        return np.column_stack([np.ones_like(u), u, *((age >= cut).astype(float) for cut in cuts)])

    def derivative(u: np.ndarray) -> np.ndarray:
        zero = np.zeros((len(np.atleast_1d(u)), len(cuts)))
        return np.column_stack([np.zeros_like(u), np.ones_like(u), zero])

    return Form(name="bands", columns=columns, derivative=derivative)


BASIS: Final = Form(
    name="basis", columns=lambda u: _raw(u), derivative=lambda u: _raw_derivative(u)
)


@dataclass(frozen=True)
class Probe:
    """What a declared baseline form implies, read off the book's own cumulative hazard.

    Carries the inputs that produced the numbers, as `anchoring.Anchor` does, so a register
    entry quoting it can be checked against the table it was computed from.
    """

    form: str
    knots: tuple[float, ...]
    ages: tuple[float, float]
    loan_months: float
    defaults: float
    marginal_rho: float
    weighted_mse: float
    minimum_log_slope: float
    supremum_g_prime: float
    basis_coefficients: tuple[float, ...]
    actual_over_expected: dict[str, float]

    def describe(self) -> dict[str, object]:
        """Flat and rounded, for a report or a register entry."""
        return {
            "form": self.form,
            "knots": list(self.knots),
            "ages": list(self.ages),
            "loan_months": round(self.loan_months),
            "defaults": round(self.defaults),
            "marginal_rho": round(self.marginal_rho, 4),
            "weighted_mse": round(self.weighted_mse, 6),
            "minimum_log_slope": round(self.minimum_log_slope, 4),
            "supremum_g_prime": round(self.supremum_g_prime, 4),
            "basis_coefficients": [round(value, 6) for value in self.basis_coefficients],
            "actual_over_expected": self.actual_over_expected,
        }


def probe(
    ages: np.ndarray,
    at_risk: np.ndarray,
    defaults: np.ndarray,
    *,
    form: Form | None = None,
    age_bands: Sequence[int] | None = None,
    shapes: np.ndarray | None = None,
) -> Probe:
    """Fit a declared baseline form to the book's own monthly hazard, before any heavy fit.

    **This exists because a form was selected over 14,007 s and nobody had looked at the profile
    it would imply.** The probe is the cheap reading of a declared form against the same data the
    heavy fit will see: the slope it produces, the error it leaves, and -- the part that matters
    -- actual over expected by the age bands section 5's criterion is scored on.

    **It fits the object the package computes, which is not the curve it looks like.** The
    monthly increment `episode_hazards` builds is

        increment(a) = exp(-rho * eta_a) * ((a+1)**rho - a**rho)

    with `eta_a` the **cell's own** value, constant across the month. So the baseline is a
    per-month *multiplier* on a Weibull increment and not a reshaping of `log H`'s own curve, and

        log increment(a) = k - rho * g(u_a) + log((a+1)**rho - a**rho)

    which is linear in `(k, g)` once `rho` is fixed. The fit is therefore a one-dimensional search
    over `rho` with a weighted linear solve inside it -- a few hundred solves, milliseconds.

    **An earlier version of this function fitted `log H` at a single `u` per month instead**, which
    makes `H` jump between months and takes the increment across two different `eta` -- the very
    error whose retraction is in the module docstring, reproduced inside the instrument built to
    avoid it. It read the band form as 1 of 6 bands in band where the structure the package fits
    reads it quite differently, and it is recorded here because an instrument that models
    something other than the model is worse than no instrument.

    Weighted by loan-months at risk. **Declared not to be evidence for a form, only against one**:
    it is in sample and marginal, where section 5's criterion is out of sample.
    """
    ages = np.asarray(ages, dtype=np.float64)
    at_risk = np.asarray(at_risk, dtype=np.float64)
    defaults = np.asarray(defaults, dtype=np.float64)
    order = np.argsort(ages)
    ages, at_risk, defaults = ages[order], at_risk[order], defaults[order]

    chosen = BASIS if form is None else form
    low, high = AGE_BASIS_RANGE
    centre = log_age(ages)
    observed = np.divide(defaults, at_risk, out=np.zeros_like(defaults), where=at_risk > 0)
    kept = (observed > 0) & (centre >= low) & (centre <= high)
    if not kept.any():
        message = "No age in the risk sets carries both exposure and a realised default."
        raise ValueError(message)

    # `eta` is the cell's own value, so the form is read at the month's declared midpoint -- the
    # same point `features` stores the covariate at -- and not at the interval's two ends.
    design = chosen.columns(centre[kept])
    target = np.log(observed[kept])
    weight = np.sqrt(at_risk[kept])
    span = ages[kept]

    best: tuple[float, float, np.ndarray] | None = None
    grid = np.linspace(0.2, 6.0, 1161) if shapes is None else np.asarray(shapes, dtype=np.float64)
    for rho in grid:
        # The Weibull's own increment over the month, which the multiplier scales.
        increment = np.log((span + 1.0) ** rho - span**rho)
        # `g` enters as `-rho * g`, so the solved coefficients are `-rho` times the form's.
        columns = np.column_stack([np.ones_like(span), design[:, 2:]])
        solved, *_ = np.linalg.lstsq(
            columns * weight[:, None], (target - increment) * weight, rcond=None
        )
        residual = float(
            at_risk[kept] @ (target - increment - columns @ solved) ** 2 / at_risk[kept].sum()
        )
        if best is None or residual < best[1]:
            best = (float(rho), residual, solved)
    assert best is not None
    rho, error, solved = best

    # Back to the form's own coefficients, and to the slope reading in `g'` units.
    raw = np.concatenate([[solved[0]], [0.0], -solved[1:] / rho])
    slope = _minimum_log_slope(np.concatenate([[0.0], [rho], solved[1:]]), chosen)
    implied = 1.0 - slope / rho if rho else float("inf")
    curvature = raw[2:] @ np.linalg.pinv(_TRANSFORM[:, 2:]) if chosen is BASIS else raw[2:]

    fitted = np.exp(
        np.log((span + 1.0) ** rho - span**rho)
        + np.column_stack([np.ones_like(span), design[:, 2:]]) @ solved
    )
    expected = at_risk[kept] * fitted
    return Probe(
        form=chosen.name,
        knots=tuple(AGE_BASIS_KNOTS) if chosen is BASIS else (),
        ages=(float(span.min()), float(span.max())),
        loan_months=float(at_risk[kept].sum()),
        defaults=float(defaults[kept].sum()),
        marginal_rho=rho,
        weighted_mse=error,
        minimum_log_slope=slope,
        supremum_g_prime=implied,
        basis_coefficients=tuple(float(value) for value in curvature),
        actual_over_expected=_by_band(span, defaults[kept], expected, bands=age_bands),
    )


def _minimum_log_slope(solution: np.ndarray, form: Form) -> float:
    """`min d log H / du` over the declared range, exactly, for a probe's own coefficients.

    The same panel-by-panel argument as :func:`baseline_slope`, on the raw span rather than the
    orthonormal one: `d log H / du` is a quadratic on each knot interval and constant outside.
    """
    edges = _slope_candidates()
    points = [edges]
    for start, stop in pairwise(edges):
        middle = 0.5 * (start + stop)
        left, centre, right = (
            float((form.derivative(np.array([x])) @ solution)[0]) for x in (start, middle, stop)
        )
        bend = left - 2.0 * centre + right
        if abs(bend) <= np.finfo(float).eps * max(1.0, abs(centre)) * 8:
            continue
        offset = 0.5 * (right - left) / bend
        if -1.0 < offset < 1.0:
            points.append(np.array([middle - offset * 0.5 * (stop - start)]))
    everywhere = np.concatenate(points)
    return float(np.min(form.derivative(everywhere) @ solution))


def _by_band(
    ages: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    bands: Sequence[int] | None = None,
) -> dict[str, float]:
    """Actual over expected by the age bands rule 5's criterion is scored on."""
    from creditsurv.config import AGE_BANDS

    edges = np.asarray(bands if bands is not None else AGE_BANDS, dtype=np.int64)
    index = np.clip(np.searchsorted(edges, ages, side="right") - 1, 0, len(edges) - 2)
    table = {}
    for position in range(len(edges) - 1):
        inside = index == position
        if not inside.any() or expected[inside].sum() <= 0:
            continue
        label = f"{edges[position]} to {edges[position + 1] - 1}"
        table[label] = round(float(actual[inside].sum() / expected[inside].sum()), 4)
    return table
