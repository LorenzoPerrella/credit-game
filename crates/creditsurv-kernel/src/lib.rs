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

/// A sum that keeps what the addition lost, and adds it back at the end.
///
/// **Three of the accumulators are scalars over every row**, and 72.7 million sequential
/// additions into one `f64` is the worst summation order there is: the error grows with the
/// count, where NumPy's `bincount` and `dot` sum pairwise and it grows with its square root.
/// Measured, that put the two backends 4e-11 apart on the objective, where this project's own
/// standard for two orderings of the same sum is the **1.97e-16** that 443 blocks and 800
/// reproduce. Neumaier's compensation closes it for a handful of flops a row, and it is
/// deterministic: a fixed order, no reassociation, the same bits on every run.
///
/// The gradient and the curvature need none of it. They accumulate into 3,001 and 286,387
/// bins, so each one sums about 24,000 terms rather than 72.7 million.
#[derive(Clone, Copy, Default)]
struct Compensated {
    total: f64,
    lost: f64,
}

impl Compensated {
    #[inline(always)]
    fn add(&mut self, value: f64) {
        let sum = self.total + value;
        self.lost += if self.total.abs() >= value.abs() {
            (self.total - sum) + value
        } else {
            (value - sum) + self.total
        };
        self.total = sum;
    }

    #[inline(always)]
    fn value(self) -> f64 {
        self.total + self.lost
    }
}

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
    value: Compensated,
    by_loan: Vec<f64>,
    by_calendar: Vec<f64>,
    shape_first: Compensated,
    curved_loan: Vec<f64>,
    curved_calendar: Vec<f64>,
    crossed: Vec<f64>,
    mixed_loan: Vec<f64>,
    mixed_calendar: Vec<f64>,
    shape_second: Compensated,
}

impl Totals {
    /// An empty set of accumulators, sized for the two tables.
    fn empty(loans: usize, calendars: usize, calendar_columns: usize, curvature: bool) -> Self {
        let curved = |size: usize| vec![0.0; if curvature { size } else { 0 }];
        Totals {
            value: Compensated::default(),
            by_loan: vec![0.0; loans],
            by_calendar: vec![0.0; calendars],
            shape_first: Compensated::default(),
            curved_loan: curved(loans),
            curved_calendar: curved(calendars),
            crossed: curved(loans * calendar_columns),
            mixed_loan: curved(loans),
            mixed_calendar: curved(calendars),
            shape_second: Compensated::default(),
        }
    }

    /// Add another part's sums into this one.
    ///
    /// **Called in the parts' own order, never as they finish.** Floating-point addition is not
    /// associative, so the order of this reduction is part of what determines the answer: a
    /// pooled fit in this project once took its shares from whichever worker finished first, and
    /// two runs of the identical model split at the twelfth digit and were five significant
    /// figures apart forty evaluations later.
    fn absorb(&mut self, other: &Totals) {
        self.value.add(other.value.value());
        self.shape_first.add(other.shape_first.value());
        self.shape_second.add(other.shape_second.value());
        for (into, from) in [
            (&mut self.by_loan, &other.by_loan),
            (&mut self.by_calendar, &other.by_calendar),
            (&mut self.curved_loan, &other.curved_loan),
            (&mut self.curved_calendar, &other.curved_calendar),
            (&mut self.crossed, &other.crossed),
            (&mut self.mixed_loan, &other.mixed_loan),
            (&mut self.mixed_calendar, &other.mixed_calendar),
        ] {
            for (slot, value) in into.iter_mut().zip(from) {
                *slot += value;
            }
        }
    }
}

/// Where one part of the rows begins and ends.
///
/// Contiguous and **fixed**: the rows are cut into as many parts as there are threads, by index,
/// so the same count always cuts them the same way. Nothing is stolen and nothing is rebalanced.
fn parts(rows: usize, threads: usize) -> Vec<(usize, usize)> {
    let threads = threads.max(1).min(rows.max(1));
    let each = rows / threads;
    let extra = rows % threads;
    let mut cuts = Vec::with_capacity(threads);
    let mut at = 0;
    for part in 0..threads {
        let length = each + usize::from(part < extra);
        cuts.push((at, at + length));
        at += length;
    }
    cuts
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
    from: usize,
    to: usize,
) -> Totals {
    let loans = eta_loan.len();
    let calendars = eta_calendar.len();
    // The shape's jet is the same for every row: one exponential and one chain, not 72.7 million.
    let rate = rate_of(family, shape);
    let mut totals = Totals::empty(loans, calendars, calendar_columns, curvature);

    // The five row arrays are walked by iterator, which costs nothing and reads better. The
    // **tables** are indexed with their bounds checked, and that is a decision taken after
    // measuring both: unchecked access bought 2% -- a Hessian of 9.96 s against 10.18 -- and
    // the first thing it did was turn a wrong accumulator length into a segmentation fault
    // inside an ordinary fit, where a bounds check would have named the array and the index.
    // Two per cent is not what this project pays for that.
    let rows = i[from..to]
        .iter()
        .zip(&j[from..to])
        .zip(&age[from..to])
        .zip(&event[from..to])
        .zip(&weight[from..to]);
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
        totals.value.add(count * jet.v);
        let first = count * jet.de;
        totals.by_loan[loan] += first;
        totals.by_calendar[key] += first;
        totals.shape_first.add(count * jet.dr);
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
        totals.shape_second.add(count * jet.drr);
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
    i, j, age, event, weight, total_weight, curvature, threads = 1,
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
    threads: usize,
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
    // **The rows in as many contiguous parts as there are threads, reduced in the parts' own
    // order.** Each part sums its own in its own order and the partials are added by index, not
    // as they finish: with a fixed count the answer is the same to the last bit on every run,
    // which is what rule 13 asks of it and what a pooled fit in this project learned the hard
    // way. The GIL is released around the whole of it -- nothing here touches a Python object.
    let cuts = parts(rows.len(), threads);
    let eta_loan = eta_loan.as_slice();
    let eta_calendar = eta_calendar.as_slice();
    let totals = py.allow_threads(|| {
        // One part runs here rather than on a thread of its own, which saves a spawn and a join
        // and nothing else. It was written to explain a 15% regression on the single-threaded
        // path and did not: the machine had drifted, which the NumPy baseline confirmed by
        // drifting with it, from 27.4 s to 31.2 on the same code. Kept because spawning one
        // thread to do all the work is pointless, not because it was measured to be faster.
        if cuts.len() == 1 {
            return accumulate(
                family,
                eta_loan,
                eta_calendar,
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
                0,
                rows.len(),
            );
        }
        let mut parts: Vec<Totals> = Vec::with_capacity(cuts.len());
        std::thread::scope(|scope| {
            let mut running = Vec::with_capacity(cuts.len());
            for (from, to) in &cuts {
                let (from, to) = (*from, *to);
                running.push(scope.spawn(move || {
                    accumulate(
                        family,
                        eta_loan,
                        eta_calendar,
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
                        from,
                        to,
                    )
                }));
            }
            for handle in running {
                parts.push(handle.join().expect("a kernel thread panicked"));
            }
        });
        let mut whole = Totals::empty(loans, calendars, calendar_columns, curvature);
        for part in &parts {
            whole.absorb(part);
        }
        whole
    });

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
    gradient[parameters - 1] = -totals.shape_first.value() / total_weight;
    let value = -totals.value.value() / total_weight;
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
    hessian[(parameters - 1) * parameters + parameters - 1] = totals.shape_second.value();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// What the compensation is for, on a sum that defeats the plain one.
    ///
    /// Ten million ones added to 1e16: every addition falls below the accumulator's last bit
    /// and the plain sum keeps none of them. The real answer is 1.001e16.
    #[test]
    fn a_compensated_sum_keeps_what_the_addition_loses() {
        let mut plain = 1e16_f64;
        let mut kept = Compensated::default();
        kept.add(1e16);
        for _ in 0..10_000_000 {
            plain += 1.0;
            kept.add(1.0);
        }
        assert_eq!(plain, 1e16, "the plain sum keeps none of them");
        assert_eq!(kept.value(), 1.0e16 + 1.0e7);
    }

    /// The same sequence twice is the same bits: rule 13's determinism, at the one place it
    /// could have been lost.
    #[test]
    fn a_compensated_sum_is_the_same_bits_every_time() {
        let run = || {
            let mut kept = Compensated::default();
            for step in 0..100_000 {
                kept.add(f64::from(step) * 1e-7 - 3.0);
            }
            kept.value()
        };
        assert_eq!(run().to_bits(), run().to_bits());
    }

    /// A zero factor gives zero and never a `nan`, which is the Python chain's own shortcut.
    #[test]
    fn a_zero_factor_gives_zero_even_against_an_infinity() {
        assert_eq!(times(0.0, f64::INFINITY), 0.0);
        assert_eq!(times(f64::INFINITY, 0.0), 0.0);
        assert_eq!(times(2.0, 3.0), 6.0);
        assert!(times(f64::NAN, 2.0).is_nan());
    }

    /// `safe_exp` caps its argument and reports the derivative of the capped value, which is
    /// what lifelines' own custom gradient does -- not the zero the mathematics would give.
    #[test]
    fn the_capped_exponential_reports_the_capped_derivative() {
        let far = safe_exp(Jet::seed_scale(1e6));
        assert_eq!(far.v, MAX_EXPONENT.exp());
        assert_eq!(far.de, MAX_EXPONENT.exp());
        assert_eq!(far.dee, MAX_EXPONENT.exp());
    }

    /// `exp(-x)` through the chain is `safe_exp` of the negated jet, to the last bit.
    #[test]
    fn the_negated_exponential_is_the_exponential_of_the_negation() {
        let jet = Jet {
            v: 2.5,
            de: -1.5,
            dr: 0.75,
            dee: 0.5,
            der: -0.25,
            drr: 0.125,
        };
        let folded = safe_exp_of_minus(jet);
        let spelled = safe_exp(Jet {
            v: -jet.v,
            de: -jet.de,
            dr: -jet.dr,
            dee: -jet.dee,
            der: -jet.der,
            drr: -jet.drr,
        });
        assert_eq!(folded.v.to_bits(), spelled.v.to_bits());
        assert_eq!(folded.de.to_bits(), spelled.de.to_bits());
        assert_eq!(folded.dr.to_bits(), spelled.dr.to_bits());
        assert_eq!(folded.dee.to_bits(), spelled.dee.to_bits());
        assert_eq!(folded.der.to_bits(), spelled.der.to_bits());
        assert_eq!(folded.drr.to_bits(), spelled.drr.to_bits());
    }

    /// `logaddexp(x, 0)` as numpy computes it, on both sides of zero and far out.
    #[test]
    fn logaddexp_matches_the_formula_numpy_uses() {
        for x in [-800.0, -40.0, -1.0, 0.0, 1.0, 40.0, 800.0] {
            let jet = logaddexp_zero(Jet::seed_scale(x));
            let expected = x.max(0.0) + (-(x.abs())).exp().ln_1p();
            assert_eq!(jet.v, expected, "at {x}");
            assert!(jet.v.is_finite(), "at {x}");
        }
    }
}
