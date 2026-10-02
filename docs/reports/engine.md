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

`creditsurv.models.kernel` writes the likelihood out: a row depends on exactly two scalars --
its own `eta` and the shape's single coefficient -- so its derivatives are six numbers whatever
the parameter count, and they are carried through a second-order forward chain rather than a
tape. The gradient is two scatter-adds into the two tables; the curvature is those plus a cross
accumulator and three small matrix products. On arrays of the real cardinalities, one thread:

| rows | value+gradient | with the Hessian | per row, v+g | per row, with H | rows and tables |
|---|---|---|---|---|---|
| 242,419 | 0.061 s | 0.171 s | 253 ns | 705 ns | 0.01 GB |
| 53,273,105 | **10.2 s** | **29.9 s** | 192 ns | 562 ns | **0.81 GB** |

Against the table at the top -- 970 ns a row for a value-and-gradient and 6,634 with a Hessian
-- that is **5.1x** and **11.8x**. A pass over the training half goes from 57.8 s to 11.4, and
from 396 s to 33.5 with the curvature, so a Newton fit of a dozen iterations is about seven
minutes on one core where a cold SLSQP path is an hour.

**Two numbers in an earlier draft of this report were wrong, and they were wrong in the
flattering direction.** It projected 18x and 41x from a prototype that computed the interval
probability as a single exponential, `d = H(a+1) - H(a)`. That is algebraically equivalent to
what lifelines computes only where no clip binds; reproducing the clips needs three cumulative
hazards and two survivals, and the prototype's 53 ns a row became 192. The projection was an
estimate of arithmetic that does not reproduce the thing being replaced.

Two things the profile settled rather than the design. **The second derivatives are optional**:
they are six of the eleven multiplications a chain rule performs and nine of the fourteen a
product does, and carrying them through a gradient-only evaluation cost 420 ns a row against
212. And **the chunk size hardly matters**: swept from 2,048 to 1,048,576 rows the figure reads
289, 347, 241, 212, 217 and 265 ns, so small chunks pay numpy's per-call overhead, large ones
leave cache, and neither is worth more than about 30%. What the chunking is for is a working set
that does not grow with the table.

The row itself is **fifteen bytes** -- four for each index, two for the age, one for the exit and
four for the weight, which is a count of loan-months and therefore an integer -- against the 208
bytes a row an expanded 26-column design costs. The whole training half is 0.81 GB of rows and
tables, with a peak of 1.31 GB including the chunk's working set.

## The same fit, on the whole book

The equivalence tests run on a fixture. This is the same question at the size the engine exists
for: the specification the selection ended on, refitted through the written-out kernel on the
production table, started from the optimum already in the cache. If the two paths are the same
estimator, Newton has nothing to do.

| | |
|---|---|
| rows | **72,671,500**, and 1,440,771 exits -- the cached fit's own counts |
| log-likelihood | **-10,691,177.687932** against the cache's -10,691,177.687932, 5.55e-16 relative |
| coefficients | **0** standard errors moved |
| standard errors | 7.45e-13 relative at worst |
| rows held | **1.09 GB**, 15 bytes a row, 443 blocks, **one** process |
| footprint | 2.49 GB |
| one value-and-gradient | **17 s** |
| the whole fit after the scan | **53 s** |
| the scan | **12.1 minutes** |

The standard errors are the stronger half of that. They come from the curvature, so agreeing to
seven parts in 10^13 is the analytic Hessian confirmed against autograd's on 72 million rows --
not on a fixture, and not on the gradient alone.

**And it says where the time now is.** Fitting is 53 seconds and reading is twelve minutes. The
scan was always this expensive; it was paid by four processes at once and nobody noticed, because
an evaluation cost as much again. What it buys is spent immediately: every fit of a selection
re-reads the whole cell file, re-derives the macro family, rebuilds the design and throws all of
it away, **once per candidate**, and the fifteen step-7 fits of one logged run each began by
recomputing the identical base objective to twelve digits.

That is the next thing to fix, and the kernel is what makes it possible rather than what makes
it necessary. The encoded rows are 1.09 GB, so they can simply be kept; and because every design
column is a function of the loan combination or of the calendar key, a *different* model's two
tables are built by putting the formula through frames of **3,001 and 153,309 rows**. One scan,
then every candidate fitted without touching the 72 million again.

## The gate for compiling anything

The compiled kernel is measured **against the NumPy above**, not against autograd. Measuring it
against autograd would credit a compiled language with removing a tape that NumPy already
removed.

The profile says where the remaining time goes, and it is not arithmetic: `_times` -- one array
multiply -- was 46% of a call, because the chain rule performs about 180 of them per chunk where
the mathematics needs perhaps 40, and each one allocates and traverses an array. The primitives
themselves are 1.3 ns a row for an exponential, 0.3 for a multiply, 1.7 for a gather and 3.1 for
a scatter-add into 153,309 bins. A fused loop holds the six jet components in registers, lets
the compiler delete the structural zeros outright, and traverses the fifteen bytes of a row once.

So the gate, if it is still worth passing: **value+gradient+Hessian over the whole half at
least three times faster than 29.9 s, with resident memory no higher than the 1.31 GB measured
here.**

**But the measurement above has moved it down the queue.** A fit is now 53 seconds of
arithmetic behind twelve minutes of reading, so compiling the evaluation would take the 53 to
perhaps fifteen and leave the twelve minutes exactly where they are. Removing the re-scan is
worth an order of magnitude on a selection where the compiled kernel is worth a few per cent,
and it should be done first. The gate stands; the priority does not.
