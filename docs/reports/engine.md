# The fit engine's envelope

*Measured on this machine (4 physical cores, 16 GB) against the production cell table,
`cells_exclude.parquet`, on the training window the selection used. Every number here was
taken from inside the process with `resource.getrusage(RUSAGE_SELF)`, never from
`/usr/bin/time -l` on `uv run`, which measures the wrapper and reported 10 MB for a 5 GB
run.*

This report exists because the next thing the model needs is finer bands, and finer bands
cost a full re-selection for two families. At the costs below that cannot be paid, so the
costs are the subject.

## What a fit actually spends, before any change

One block of the real table, the real formula, **26 parameters** -- 25 on the scale and an
intercept-only shape. Best of three for the value-and-gradient, one run for the Hessian.

| cells a batch | rows | design expansion | value+gradient | Hessian | v+g per row | Hessian per row |
|---|---|---|---|---|---|---|
| 50,000 | 49,079 | 7.2 ms | 40.5 ms | 273.2 ms | 0.83 µs | 5.57 µs |
| 250,000 | 242,419 | 39.6 ms | 234.9 ms | 1,373.4 ms | 0.97 µs | 5.67 µs |
| 1,000,000 | 967,791 | 290.7 ms | 1,287.9 ms | 7,593.1 ms | 1.33 µs | 7.85 µs |

Three things in that table correct what the project had written down.

**The published 66 ms is stale, by 3.6x.** `CLAUDE.md` records "a value 28 ms, a value with
its gradient 66 ms, a Hessian 1,381 ms" on a 250,000-row block with **19 parameters**. Rule
12 turned three continuous covariates into band factors and the design went to 26 columns:
the same measurement is now **234.9 ms**. The Hessian is unchanged at 1,373 ms. Most of the
gap between the micro-benchmark and the logs -- an evaluation on the training half reading
~180 s where 66 ms a block predicted ~16-21 s -- was never a mystery in the engine. It was a
number taken before the specification changed and not retaken after.

**"A Hessian is 49 times the value" measures a call the engine never makes.** `minimize` is
always called with `jac=True`, so every optimiser evaluation is a value *and* gradient; the
only value-only consumers are the polish's line search and Newton's entry check, and both
discard the gradient rather than ask for a cheaper call. Against what the optimiser actually
pays, a Hessian is **5.8x a value-and-gradient**, not 49x a value. The ratio that matters for
choosing an optimiser is the smaller one.

**The cost per row rises with the block size.** From 49,000 rows to 968,000 a
value-and-gradient goes from 0.83 to 1.33 µs a row -- a **1.6x penalty** -- and the design
expansion alone from 0.147 to 0.300. At 26 columns a million-row block allocates 208 MB of
design, about 700 MB of autograd tape and up to five filtered copies of the design inside
lifelines' likelihood, **on every evaluation and again on every Hessian**. The runs whose
logs read ~180 s an evaluation were million-row-block runs at a 14.56 GB footprint; the
published run used 250,000-cell batches. So the residue is block-size superlinearity and
memory pressure, and there is no unattributed factor left to find.

Scaled to the training half of 59,663,961 rows at 250,000-cell batches, one worker: a
value-and-gradient is **57.8 s**, a Hessian **338 s**. A cold fit of 50 to 121 evaluations
plus three or four Hessians is the hour it is observed to be.

## What the likelihood actually distinguishes

Measured on the training window -- 72,671,500 cells over 2.11 billion loan-months:

| | |
|---|---|
| distinct **loan**-covariate combinations | **3,001** |
| distinct **(origination month, age)** pairs | **38,384** |
| the same with the LTV band, which `ltv_change` needs | **153,309** |
| rows the likelihood distinguishes, cause `default` | **53,273,105** of 72,671,500 (1.36x) |

The 1.36x is the exact merge: the model never reads `delinquency_state`, and for one cause
the three-state outcome collapses to a boolean, so rows agreeing on the design and on their
bounds are one row with the weights summed. It is smaller than it looks like it should be,
because the cell space is sparse: the average multiplicity across the four payment states
and the three outcomes is only 1.62.

The other three rows are the structural finding. **Every column of the design is a function
of the loan combination `i` or of the calendar key `j`, and never of both.** The couplings
that could have broken this are `ltv_change`, which is the LTV band's midpoint times a
house-price ratio, and `mortgage_rate_decline`, whose benchmark is chosen by the term: both
are resolved by putting the LTV band and the term inside `j`. (`refinance_incentive` and
`origination_spread` need the loan's own note rate, which is not in the key, and are already
unfittable here for that reason.) So

    eta = A[i] + B[j]

with `A` over 3,001 entries and `B` over 153,309, rebuilt per evaluation for about 1.1
million flops -- against expanding a 26-column design over 53 million rows.

And the time structure collapses with it. The interval is always `[a, a+1]` for an exit and
`[a+1, ∞)` for a survivor, the entry is always the age, and exact observations never occur
(`panel.py:147` sets them `False` with no condition). With `u = log a - eta` and
`L = log((a+1)/a)`, both lookups on at most 361 ages:

    H(a) = exp(rho·u),   d = H(a+1) - H(a) = H(a)·(exp(rho·L) - 1)
    survivor:  ll = -d
    exit:      ll = log(1 - exp(-d))

**One exponential a row** for the 97.6% of rows that are survivors, two for an exit. The
whole per-row state is `(i: u16, j: u32, one event bit, weight: u32)` -- about 10 bytes, so
530 MB for the entire training half, resident, in one process.

## What that arithmetic costs, measured

A chunked NumPy implementation of the above, on arrays of the real cardinalities, 2²⁰ rows a
chunk, single-threaded:

| rows | value+gradient | with the Hessian | per row, v+g | per row, with H | footprint |
|---|---|---|---|---|---|
| 242,419 | 10.4 ms | 46.3 ms | 43 ns | 191 ns | — |
| 5,000,000 | 284.6 ms | 821.3 ms | 57 ns | 164 ns | 0.28 GB |
| 53,273,105 | **2.82 s** | **8.67 s** | 53 ns | 163 ns | **0.98 GB** |

Per row, against the table at the top: **18x** on a value-and-gradient and **41x** on a
value-and-gradient-and-Hessian. The cost per row is flat in the number of rows -- 43, 57, 53
ns -- where the present engine's rises by 1.6x, because nothing here is allocated per
evaluation. A Hessian stops being 5.8 times a value-and-gradient and becomes 3.1.

Three honest qualifications. The curvature scalars in the prototype are placeholders with
the right **cost** -- the same number of passes and accumulations -- not the final algebra;
deriving that exactly, and holding it to autograd, is the work itself. The indices are drawn
at random, which is the worst case for the gathers `A[i]`, `B[j]` and `X_cal[j]`; the real
table is written in key order, so the real thing should be faster rather than slower. And
this is one thread: the present engine's 57.8 s is also one worker.

What it changes downstream: with an exact Hessian at three times the cost of a gradient,
Newton from the first step replaces a 50-to-121-evaluation SLSQP path, and a fit becomes
about ten to fifteen iterations of 8.7 s.

## The gate for compiling anything

The compiled kernel is measured **against the NumPy above**, not against autograd. Measuring
it against autograd would credit a compiled language with removing a tape that NumPy already
removed. It must be at least three times faster on value-and-gradient-and-Hessian over the
whole training half, and hold resident memory under 1 GB, or it is abandoned and the number
is reported here.
