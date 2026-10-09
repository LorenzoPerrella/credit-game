//! The interval-censored AFT likelihood, one row at a time, as a fused loop.
//!
//! This is the second implementation of `creditsurv.models.kernel`, and the NumPy one is
//! normative: what is written here reproduces it, including lifelines' clips and the grouping of
//! the second-order terms, and `tests/test_kernel.py` compares the two.
//!
//! **What crosses the boundary** is numpy arrays of fixed dtype and nothing else -- `f64` for the
//! parameters and the two design tables, `u32` for the key codes, `u16` for the loan age, `bool`
//! for the event flag -- and back a scalar, an `f64[p]` and an `f64[p, p]`. No Python objects, no
//! pandas, no lifelines. Inside: no I/O, no logging, no configuration and no rule of
//! `docs/rules.md`. The loop over rows, and the small contractions that turn its two
//! accumulators into a gradient and a curvature.
//!
//! **Why it exists.** A row depends on exactly two scalars -- its own linear predictor and the
//! shape's single coefficient -- so its derivatives are six numbers whatever the parameter count,
//! and the NumPy path carries them as six arrays over a chunk: 178 array operations where the
//! mathematics needs forty, each one allocating and traversing. Measured, a Hessian over the
//! training half spends 67% of its 409 ns a row in that chain and the arithmetic itself is a
//! minority of it. Here the six components are registers.
//!
//! **One declared difference from the NumPy, and it is in the sign of a nan.** The Python chain
//! carries a structural zero as the literal `0.0` and skips the term, so `0 * inf` never happens
//! where a component is zero by construction; an *array* element that is merely numerically zero
//! gets no such treatment and `0.0 * inf` is a `nan` there. This loop takes the shortcut on the
//! value rather than on the construction, so it returns `0.0` where the NumPy returns `nan`, far
//! outside the region any fit converges in. It is reachable only where a derivative has already
//! overflowed, which is past the wall the engine backtracks from.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// lifelines' `safe_exp` ceiling: `exp` is never asked for more than this, and its derivative is
/// reported as the capped value rather than as zero.
const MAX_EXPONENT: f64 = 709.782712893384 - 75.0;

/// The interval probability's clip, from `_log_likelihood_interval_censoring`.
const INTERVAL_FLOOR: f64 = 1e-25;
const INTERVAL_CEILING: f64 = 1.0 - 1e-25;

/// The survival function's clip, from `ParametricRegressionFitter._survival_function`. The
/// Weibull fitter overrides that method and does **not** clip; the log-logistic inherits it.
const SURVIVAL_FLOOR: f64 = 1e-12;
const SURVIVAL_CEILING: f64 = 1.0 - 1e-12;

const WEIBULL: u8 = 0;
const LOGLOGISTIC: u8 = 1;

/// A product that takes the Python chain's shortcut: a zero factor gives zero, never a `nan`.
#[inline(always)]
fn times(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        0.0
    } else {
        left * right
    }
}

/// A quantity and its first and second derivatives in the two scalars a row depends on.
///
/// `e` is the row's own scale predictor and `r` the shape's single coefficient. Six numbers,
/// independent of how many parameters the model has, which is why a Hessian costs what a gradient
/// costs here.
#[derive(Clone, Copy)]
struct Jet {
    v: f64,
    de: f64,
    dr: f64,
    dee: f64,
    der: f64,
    drr: f64,
}

impl Jet {
    #[inline(always)]
    fn constant(v: f64) -> Self {
        Jet {
            v,
            de: 0.0,
            dr: 0.0,
            dee: 0.0,
            der: 0.0,
            drr: 0.0,
        }
    }

    #[inline(always)]
    fn seed_scale(v: f64) -> Self {
        Jet {
            v,
            de: 1.0,
            dr: 0.0,
            dee: 0.0,
            der: 0.0,
            drr: 0.0,
        }
    }

    #[inline(always)]
    fn seed_shape(v: f64) -> Self {
        Jet {
            v,
            de: 0.0,
            dr: 1.0,
            dee: 0.0,
            der: 0.0,
            drr: 0.0,
        }
    }

    #[inline(always)]
    fn minus(self, other: Jet) -> Jet {
        Jet {
            v: self.v - other.v,
            de: self.de - other.de,
            dr: self.dr - other.dr,
            dee: self.dee - other.dee,
            der: self.der - other.der,
            drr: self.drr - other.drr,
        }
    }

    #[inline(always)]
    fn plus(self, other: Jet) -> Jet {
        Jet {
            v: self.v + other.v,
            de: self.de + other.de,
            dr: self.dr + other.dr,
            dee: self.dee + other.dee,
            der: self.der + other.der,
            drr: self.drr + other.drr,
        }
    }

    #[inline(always)]
    fn mul(self, other: Jet) -> Jet {
        Jet {
            v: times(self.v, other.v),
            de: times(self.de, other.v) + times(self.v, other.de),
            dr: times(self.dr, other.v) + times(self.v, other.dr),
            dee: (times(self.dee, other.v) + times(self.v, other.dee))
                + times(2.0, times(self.de, other.de)),
            der: (times(self.der, other.v) + times(self.v, other.der))
                + (times(self.de, other.dr) + times(self.dr, other.de)),
            drr: (times(self.drr, other.v) + times(self.v, other.drr))
                + times(2.0, times(self.dr, other.dr)),
        }
    }

    /// `f(self)`, given `f` at this value and its first two derivatives there.
    ///
    /// The second-order terms are grouped as `(f'' * d) * d` rather than `f'' * (d * d)`, which
    /// is not a style: the two differ where one factor overflows. Far out on the prepayment
    /// surface a cumulative hazard reaches 1.3e275 and its derivative -2.6e276, so the square is
    /// `inf` -- and a survival of exactly zero times `inf` is `nan`, where the other association
    /// gives zero.
    #[inline(always)]
    fn chain(self, value: f64, first: f64, second: f64) -> Jet {
        let curved_e = times(second, self.de);
        Jet {
            v: value,
            de: times(first, self.de),
            dr: times(first, self.dr),
            dee: times(first, self.dee) + times(curved_e, self.de),
            der: times(first, self.der) + times(curved_e, self.dr),
            drr: times(first, self.drr) + times(times(second, self.dr), self.dr),
        }
    }

    /// This jet times a constant -- a mask or a weight, nothing that carries derivatives.
    #[inline(always)]
    fn scaled(self, by: f64) -> Jet {
        Jet {
            v: times(self.v, by),
            de: times(self.de, by),
            dr: times(self.dr, by),
            dee: times(self.dee, by),
            der: times(self.der, by),
            drr: times(self.drr, by),
        }
    }
}

/// lifelines' `safe_exp`: the argument is capped, the derivative is not zeroed.
#[inline(always)]
fn safe_exp(jet: Jet) -> Jet {
    let value = jet.v.min(MAX_EXPONENT).exp();
    jet.chain(value, value, value)
}

/// `safe_exp(-jet)`, from `f' = -f` and `f'' = f`, without negating the jet first.
#[inline(always)]
fn safe_exp_of_minus(jet: Jet) -> Jet {
    let value = (-jet.v).min(MAX_EXPONENT).exp();
    jet.chain(value, -value, value)
}

#[inline(always)]
fn exp_jet(jet: Jet) -> Jet {
    let value = jet.v.exp();
    jet.chain(value, value, value)
}

/// `exp(-jet)`, with no cap -- which is how the log-logistic writes its survival.
#[inline(always)]
fn exp_of_minus(jet: Jet) -> Jet {
    let value = (-jet.v).exp();
    jet.chain(value, -value, value)
}

#[inline(always)]
fn log_jet(jet: Jet) -> Jet {
    let value = jet.v.ln();
    let first = 1.0 / jet.v;
    jet.chain(value, first, -first * first)
}

/// `clip`, whose gradient autograd zeroes wherever the answer is a bound.
#[inline(always)]
fn clipped(jet: Jet, floor: f64, ceiling: f64) -> Jet {
    let value = jet.v.clamp(floor, ceiling);
    let inside = if value != floor && value != ceiling {
        1.0
    } else {
        0.0
    };
    jet.chain(value, inside, 0.0)
}

/// `logaddexp(jet, 0)` -- the log-logistic's cumulative hazard, as numpy computes the value.
#[inline(always)]
fn logaddexp_zero(jet: Jet) -> Jet {
    let value = jet.v.max(0.0) + (-(jet.v.abs())).exp().ln_1p();
    let first = 1.0 / (1.0 + (-jet.v).exp());
    jet.chain(value, first, first * (1.0 - first))
}

/// The shape's own jet, which does not depend on the row and is taken once for the whole pass.
///
/// `safe_exp` for the Weibull and a plain `exp` for the log-logistic -- the two families do it
/// differently, and it is a single number either way.
#[inline(always)]
fn rate_of(family: u8, shape: f64) -> Jet {
    let coefficient = Jet::seed_shape(shape);
    if family == WEIBULL {
        safe_exp(coefficient)
    } else {
        exp_jet(coefficient)
    }
}

/// The scale as the family reads it: as it stands for the Weibull, and through
/// `log(safe_exp(eta))` for the log-logistic, written as lifelines writes it rather than
/// simplified to `eta` -- the two differ in the last digits, and above the cap altogether.
#[inline(always)]
fn scale_of(family: u8, eta: f64) -> Jet {
    let scale = Jet::seed_scale(eta);
    if family == WEIBULL {
        scale
    } else {
        log_jet(safe_exp(scale))
    }
}

#[inline(always)]
fn cumulative(family: u8, log_time: f64, scale: Jet, rate: Jet) -> Jet {
    let inner = rate.mul(Jet::constant(log_time).minus(scale));
    if family == WEIBULL {
        safe_exp(inner)
    } else {
        logaddexp_zero(inner)
    }
}

#[inline(always)]
fn survival(family: u8, hazard: Jet) -> Jet {
    if family == WEIBULL {
        safe_exp_of_minus(hazard)
    } else {
        clipped(exp_of_minus(hazard), SURVIVAL_FLOOR, SURVIVAL_CEILING)
    }
}

/// One row's log-likelihood and its derivatives in `eta` and the shape's coefficient.
///
/// The interval-censored term and the left-truncation term, exactly as
/// `_log_likelihood_interval_censoring` writes them for a panel where no observation is exact:
///
/// ```text
/// log(clip(S(start) - S(stop), 1e-25, 1 - 1e-25))  +  H(entry) if entry > 0
/// ```
///
/// The second term is **not** inside the clip, and that asymmetry is why the objective is
/// unbounded below far from the data.
#[inline(always)]
fn row_likelihood(
    family: u8,
    log_entry: f64,
    log_start: f64,
    log_stop: f64,
    truncated: f64,
    eta: f64,
    rate: Jet,
) -> Jet {
    let scale = scale_of(family, eta);
    let entry = cumulative(family, log_entry, scale, rate);
    let opened = survival(family, cumulative(family, log_start, scale, rate));
    let closed = survival(family, cumulative(family, log_stop, scale, rate));
    let interval = clipped(opened.minus(closed), INTERVAL_FLOOR, INTERVAL_CEILING);
    log_jet(interval).plus(entry.scaled(truncated))
}

/// A design table times a coefficient vector, row by row.
///
/// ``rows`` is passed rather than divided out of the slice's length, and that is not a style:
/// a model with no calendar covariates has a table of `n` rows and **zero** columns, where the
/// division gives zero rows. The accumulators are sized from this, so a zero there allocated
/// them empty and the row loop wrote past the end of them.
fn predictor(table: &[f64], rows: usize, columns: usize, coefficients: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for (row, value) in out.iter_mut().enumerate() {
        let start = row * columns;
        let mut total = 0.0;
        for column in 0..columns {
            total += table[start + column] * coefficients[column];
        }
        *value = total;
    }
    out
}

/// `table.T @ weights`: a column's share of the gradient.
fn contract(table: &[f64], columns: usize, weights: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; columns];
    for (row, weight) in weights.iter().enumerate() {
        if *weight == 0.0 {
            continue;
        }
        let start = row * columns;
        for column in 0..columns {
            out[column] += table[start + column] * weight;
        }
    }
    out
}

/// `table.T @ (table * weights[:, None])`: one side's block of the curvature.
fn contract_square(table: &[f64], columns: usize, weights: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; columns * columns];
    for (row, weight) in weights.iter().enumerate() {
        if *weight == 0.0 {
            continue;
        }
        let start = row * columns;
        for a in 0..columns {
            let scaled = table[start + a] * weight;
            for b in 0..columns {
                out[a * columns + b] += scaled * table[start + b];
            }
        }
    }
    out
}

/// The accumulators one pass over the rows fills.
struct Totals {
    value: f64,
    by_loan: Vec<f64>,
    by_calendar: Vec<f64>,
    shape_first: f64,
    curved_loan: Vec<f64>,
    curved_calendar: Vec<f64>,
    crossed: Vec<f64>,
    mixed_loan: Vec<f64>,
    mixed_calendar: Vec<f64>,
    shape_second: f64,
}

#[allow(clippy::too_many_arguments)]
fn accumulate(
    family: u8,
    eta_loan: &[f64],
    eta_calendar: &[f64],
    shape: f64,
    log_entry: &[f64],
    log_following: &[f64],
    log_far: f64,
    calendar: &[f64],
    calendar_columns: usize,
    i: &[u32],
    j: &[u32],
    age: &[u16],
    event: &[bool],
    weight: &[u32],
    curvature: bool,
) -> Totals {
    let loans = eta_loan.len();
    let calendars = eta_calendar.len();
    // The shape's jet is the same for every row: one exponential and one chain, not 72.7 million.
    let rate = rate_of(family, shape);
    let mut totals = Totals {
        value: 0.0,
        by_loan: vec![0.0; loans],
        by_calendar: vec![0.0; calendars],
        shape_first: 0.0,
        curved_loan: vec![0.0; if curvature { loans } else { 0 }],
        curved_calendar: vec![0.0; if curvature { calendars } else { 0 }],
        crossed: vec![
            0.0;
            if curvature {
                loans * calendar_columns
            } else {
                0
            }
        ],
        mixed_loan: vec![0.0; if curvature { loans } else { 0 }],
        mixed_calendar: vec![0.0; if curvature { calendars } else { 0 }],
        shape_second: 0.0,
    };

    // The five row arrays are walked by iterator, which costs nothing and reads better. The
    // **tables** are indexed with their bounds checked, and that is a decision taken after
    // measuring both: unchecked access bought 2% -- a Hessian of 9.96 s against 10.18 -- and
    // the first thing it did was turn a wrong accumulator length into a segmentation fault
    // inside an ordinary fit, where a bounds check would have named the array and the index.
    // Two per cent is not what this project pays for that.
    let rows = i.iter().zip(j).zip(age).zip(event).zip(weight);
    for ((((loan, key), months), exits), count) in rows {
        let loan = *loan as usize;
        let key = *key as usize;
        let months = *months as usize;
        let count = f64::from(*count);
        let opens = log_entry[months];
        let closes = log_following[months];
        let eta = eta_loan[loan] + eta_calendar[key];
        let jet = row_likelihood(
            family,
            opens,
            if *exits { opens } else { closes },
            if *exits { closes } else { log_far },
            if months > 0 { 1.0 } else { 0.0 },
            eta,
            rate,
        );
        totals.value += count * jet.v;
        let first = count * jet.de;
        totals.by_loan[loan] += first;
        totals.by_calendar[key] += first;
        totals.shape_first += count * jet.dr;
        if !curvature {
            continue;
        }
        let second = count * jet.dee;
        let mixing = count * jet.der;
        let start = key * calendar_columns;
        let crossing = loan * calendar_columns;
        totals.curved_loan[loan] += second;
        totals.curved_calendar[key] += second;
        for column in 0..calendar_columns {
            totals.crossed[crossing + column] += second * calendar[start + column];
        }
        totals.mixed_loan[loan] += mixing;
        totals.mixed_calendar[key] += mixing;
        totals.shape_second += count * jet.drr;
    }
    totals
}

/// What crosses back: the objective, its gradient, and its curvature where one was asked for.
///
/// The three results rule 13 declares, and nothing else. A `None` curvature is an evaluation
/// the optimiser asked a value and a gradient of, which is most of them.
type Evaluated<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

/// The objective over every row, its gradient, and its curvature when one is asked for.
///
/// The parameter vector is `[loan..., calendar..., shape]`, in the order the two design tables
/// give their columns; the caller maps that onto lifelines' own order, because which index a
/// coefficient sits at is a fact about lifelines and not about this arithmetic.
#[pyfunction]
#[pyo3(signature = (
    family, loan, calendar, x_loan, x_calendar, shape, log_entry, log_following, log_far,
    i, j, age, event, weight, total_weight, curvature,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate<'py>(
    py: Python<'py>,
    family: u8,
    loan: PyReadonlyArray2<'py, f64>,
    calendar: PyReadonlyArray2<'py, f64>,
    x_loan: PyReadonlyArray1<'py, f64>,
    x_calendar: PyReadonlyArray1<'py, f64>,
    shape: f64,
    log_entry: PyReadonlyArray1<'py, f64>,
    log_following: PyReadonlyArray1<'py, f64>,
    log_far: f64,
    i: PyReadonlyArray1<'py, u32>,
    j: PyReadonlyArray1<'py, u32>,
    age: PyReadonlyArray1<'py, u16>,
    event: PyReadonlyArray1<'py, bool>,
    weight: PyReadonlyArray1<'py, u32>,
    total_weight: f64,
    curvature: bool,
) -> PyResult<Evaluated<'py>> {
    if family != WEIBULL && family != LOGLOGISTIC {
        return Err(PyValueError::new_err(
            "family must be 0 (weibull) or 1 (loglogistic)",
        ));
    }
    let loan_table = loan.as_slice()?;
    let calendar_table = calendar.as_slice()?;
    let loan_columns = loan.shape()[1];
    let calendar_columns = calendar.shape()[1];
    let loans = loan.shape()[0];
    let calendars = calendar.shape()[0];
    let coefficients_loan = x_loan.as_slice()?;
    let coefficients_calendar = x_calendar.as_slice()?;
    if coefficients_loan.len() != loan_columns || coefficients_calendar.len() != calendar_columns {
        return Err(PyValueError::new_err(
            "the coefficient vectors must match the design tables' columns",
        ));
    }
    let rows = i.as_slice()?;
    let keys = j.as_slice()?;
    let ages = age.as_slice()?;
    let events = event.as_slice()?;
    let counts = weight.as_slice()?;
    if keys.len() != rows.len()
        || ages.len() != rows.len()
        || events.len() != rows.len()
        || counts.len() != rows.len()
    {
        return Err(PyValueError::new_err(
            "the five row arrays must be the same length",
        ));
    }
    let entry = log_entry.as_slice()?;
    let following = log_following.as_slice()?;
    if entry.len() != following.len() {
        return Err(PyValueError::new_err(
            "the two log-time tables must be the same length",
        ));
    }
    for (number, months) in ages.iter().enumerate() {
        if (*months as usize) >= entry.len() {
            return Err(PyValueError::new_err(format!(
                "row {number} has age {months}, past the log-time tables' {} entries",
                entry.len()
            )));
        }
    }
    for (number, index) in rows.iter().enumerate() {
        if (*index as usize) >= loans {
            return Err(PyValueError::new_err(format!(
                "row {number} indexes loan combination {index}, past the table's {loans}"
            )));
        }
    }
    for (number, index) in keys.iter().enumerate() {
        if (*index as usize) >= calendars {
            return Err(PyValueError::new_err(format!(
                "row {number} indexes calendar key {index}, past the table's {calendars}"
            )));
        }
    }

    let eta_loan = predictor(loan_table, loans, loan_columns, coefficients_loan);
    let eta_calendar = predictor(
        calendar_table,
        calendars,
        calendar_columns,
        coefficients_calendar,
    );
    let totals = accumulate(
        family,
        &eta_loan,
        &eta_calendar,
        shape,
        entry,
        following,
        log_far,
        calendar_table,
        calendar_columns,
        rows,
        keys,
        ages,
        events,
        counts,
        curvature,
    );

    let parameters = loan_columns + calendar_columns + 1;
    let gradient_loan = contract(loan_table, loan_columns, &totals.by_loan);
    let gradient_calendar = contract(calendar_table, calendar_columns, &totals.by_calendar);
    let mut gradient = vec![0.0; parameters];
    for (column, value) in gradient_loan.iter().enumerate() {
        gradient[column] = -value / total_weight;
    }
    for (column, value) in gradient_calendar.iter().enumerate() {
        gradient[loan_columns + column] = -value / total_weight;
    }
    gradient[parameters - 1] = -totals.shape_first / total_weight;
    let value = -totals.value / total_weight;
    let gradient_out = PyArray1::from_vec(py, gradient);
    if !curvature {
        return Ok((value, gradient_out, None));
    }

    let mut hessian = vec![0.0; parameters * parameters];
    let loan_block = contract_square(loan_table, loan_columns, &totals.curved_loan);
    for a in 0..loan_columns {
        for b in 0..loan_columns {
            hessian[a * parameters + b] = loan_block[a * loan_columns + b];
        }
    }
    let calendar_block = contract_square(calendar_table, calendar_columns, &totals.curved_calendar);
    for a in 0..calendar_columns {
        for b in 0..calendar_columns {
            hessian[(loan_columns + a) * parameters + loan_columns + b] =
                calendar_block[a * calendar_columns + b];
        }
    }
    // The cross block: the loan table contracted against what each loan combination accumulated
    // of the calendar's own columns.
    for a in 0..loan_columns {
        for b in 0..calendar_columns {
            let mut total = 0.0;
            for row in 0..loans {
                total +=
                    loan_table[row * loan_columns + a] * totals.crossed[row * calendar_columns + b];
            }
            hessian[a * parameters + loan_columns + b] = total;
            hessian[(loan_columns + b) * parameters + a] = total;
        }
    }
    let mixed_loan = contract(loan_table, loan_columns, &totals.mixed_loan);
    for (column, value) in mixed_loan.iter().enumerate() {
        hessian[column * parameters + parameters - 1] = *value;
        hessian[(parameters - 1) * parameters + column] = *value;
    }
    let mixed_calendar = contract(calendar_table, calendar_columns, &totals.mixed_calendar);
    for (column, value) in mixed_calendar.iter().enumerate() {
        hessian[(loan_columns + column) * parameters + parameters - 1] = *value;
        hessian[(parameters - 1) * parameters + loan_columns + column] = *value;
    }
    hessian[(parameters - 1) * parameters + parameters - 1] = totals.shape_second;
    for entry in hessian.iter_mut() {
        *entry = -*entry / total_weight;
    }
    let curvature_out = PyArray1::from_vec(py, hessian).reshape([parameters, parameters])?;
    Ok((value, gradient_out, Some(curvature_out)))
}

#[pymodule]
fn creditsurv_kernel(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(evaluate, module)?)?;
    module.add("MAX_EXPONENT", MAX_EXPONENT)?;
    // The family codes, so the caller's own map can be held to them by a test rather than
    // agreed by convention: a swap here fits the other family under the right name.
    module.add("WEIBULL", WEIBULL)?;
    module.add("LOGLOGISTIC", LOGLOGISTIC)?;
    Ok(())
}
