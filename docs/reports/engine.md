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

The 153,309 is what the calendar *can* distinguish; the key a reading actually builds is
**152,565** for the published model's six macro columns, because a few hundred (month, age,
band) triples land on identical macro values and are one key. A reading covering all fifteen
candidates finds **286,387**, which is a different quantity and not a disagreement: see the end
of "One reading, many fits".

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
| 72,671,500, the half as it is fitted | **12.95 s** | **29.69 s** | 178 ns | 409 ns | 1.10 GB |

The third row is the one to quote and the one the gate below is declared against: the whole
training half, unmerged, at the 26 parameters rule 12 produced, read from the cached encoding
in 0.45 s and expanded from the keys in 0.48. The process holds **467 MB** -- the rows are a
memory map, so the gigabyte of them is never resident.

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

### The chain's own waste, removed -- and what that says about the rest

The chain performed **178 array operations a chunk** where the mathematics needs about forty,
and four of them were waste rather than generality:

* a row reads the cumulative hazard at three times and the survival at **two**, and the
  entry's survival was computed and thrown away -- 19 operations including two exponentials;
* the log-logistic's `log(safe_exp(eta))` does not depend on the time and was recomputed for
  each of the three -- 52 operations;
* `safe_exp(-H)` negated six arrays to get its argument, where the chain gives `exp(-x)` from
  `f' = -f` and `f'' = f` directly;
* and `a - b` was `a + (-b)`, twelve arrays for six.

Removed, with the answer identical to the last bit at three points including one far out where
every clip binds, on the whole training half at 26 parameters:

| | value+gradient | with the Hessian |
|---|---|---|
| Weibull, before | 186 ns/row | 471 ns/row |
| Weibull | **166** | **415** |
| log-logistic, before | 298 | 630 |
| log-logistic | **271** | **557** |

**12% on a Hessian in both families** -- and the interesting part is that it is only 12% for a
quarter fewer operations. Timing the halves separately, on the real tables, says where a
Hessian's 409 ns a row goes: the **jet 67%**, the **scatter-adds 18%**, the gathers and the
masks 4%, and the rest the dots and the final matrix products. So the chain is the cost, but
its cost is not its arithmetic: 178 traversals of a 512 KB array should be about 5 ms a chunk
and the jet takes 19. What is left is **allocation and dispatch** -- a fresh array per
operation, and numpy's per-call overhead -- which is also why the chunk sweep has a floor in
the middle: small chunks pay the dispatch and large ones pay the memory. Neither is arithmetic,
and neither is reachable from Python.

The sweep was retaken after the change and its floor moved from 131,072 rows to 65,536, worth
nothing on the Weibull and 2% on the log-logistic. **Not taken**: the chunk is where the sums
are cut, so moving it re-partitions every sum and a re-run of a cached fit would differ in its
last digits. Two percent does not buy that.

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

## One reading, many fits

The reading is 93% of a fit, so it is read once and every model is fitted from it. Measured on
the production table, one process:

| | |
|---|---|
| the reading, with **no formula involved** | **10.88 minutes**, 1.09 GB of rows, 2.88 GB footprint |
| the keys it found | **3,001** loan combinations, **152,565** calendar |
| the selected model, warm from the cached optimum | **0.89 min**, 1 evaluation, 0 Newton steps |
| a nested candidate, warm from its parent | **5.67 min**, 6 evaluations, 5 Newton steps |
| the selected model, cold | 43.79 min, **142 evaluations**, 2 Newton steps |

The warm fit lands **9.9e-15** standard errors from the optimum already in the cache and the
cold one 0.000917, both at the same log-likelihood to four decimals of ten million. The first
evaluation of the warm fit reads 0.005060834733 -- the same twelve digits the scan path read.
So the encoding is the same estimator at full scale, not only on a fixture.

**The reading itself barely moved**: 10.88 minutes against the 12.1 of a scan that also built a
design, accumulated moments over 26 columns and compacted it. What is left is parquet, the
macro family and the key coding, and it is close to irreducible. The gain is not that a reading
got cheaper; it is that a selection pays **three** of them -- the training half and the two
origination-year halves of step 9 -- where it used to pay one per candidate, about thirty.

Finding one of those three took vectorising a lookup: walking each block in Python to find
which combinations were new to it is two passes over 250,000 rows for each of 443 blocks, 221
million interpreted iterations, and it made an encoding *slower* than the scan. `np.unique`
with `return_index` does it in one sorted call.

### The reading is written once, and the second one is half a second

A reading involves no formula, so nothing in it depends on the model: it is parquet, the macro
family and the key coding. Written to disk and mapped back it stops being a cost at all.
Measured on the production training half, the same covering set a selection needs:

| | |
|---|---|
| the reading | **693.9 s** and **717.9** on a second run, 72,671,500 cells, 1,440,771 defaults, 443 blocks |
| its footprint | **2.37 GB** -- one batch expanded, beside the rows it keeps |
| the same reading in the table above | 10.88 min and 2.88 GB, measured separately |
| mapped back, in a new process | **0.4 s**, a **1,735x** saving, and 7.98 s for the whole process |
| its footprint | **0.63 GB**: the 1.09 GB of rows is mapped, not resident |
| what it occupies | **1.0 GB** on disk, against 448 MB of parquet it was read from |
| two independent readings | the five row arrays **byte-identical**, `shasum` on 1.09 GB |
| the fit from it | the same log-likelihood, coefficients and standard errors, **bit for bit** |

The third row is the same reading measured earlier in this report, and the spread -- 10.88 to
11.97 minutes, 2.37 to 2.88 GB -- is the repeatability to expect of a figure dominated by
parquet and a laptop's page cache. It is not a disagreement and nothing here turns on it.

The last two rows are the claim worth making, and the second is the one that matters. A reading that came back *almost* the
same would be a different model published under the same name, and it can be exact because
nothing is recomputed: the rows come back as views of a memory map, the keys in the order their
codes were handed out in, and the blocks cut where they were cut -- and the blocks are where the
sums are cut, which is the one thing a reading can change about an objective, floating-point
addition not being associative. That is also why the batch size is in the name.

Restoring the key tables is **replaying the keys, not the rows**: the key frames come back in
the order the codes were handed out in and are registered in that order, so every combination
keeps the index the saved rows point at. The categorical levels are re-applied from the
declaration rather than taken from parquet, because what a formula's expansion reads off a key
frame is its *dtypes*: a column that came back as plain strings would hand `C(purpose)` whatever
reference level pandas sorted first.

**The name is the reading, not the run.** It covers the cell table's identity, the two covariate
lists, the cause, the parity, the window, the batch size and the macro panel's own numbers --
clipped to the months the window can reach, because the panel is live FRED data and a month
published above the cut cannot have entered a reading that stops below it. So rule 2's four runs
share **six** readings, two causes by three samples, where they used to pay twelve; and a run
that stops now picks up at the fit it was on instead of at the reading, which on a job that
takes a day matters more than the minutes.

**And the key is the covering set, which is why two numbers are right.** The table above reads
152,565 calendar keys for the *selected* model's six macro columns and the selection's reading
finds **286,387**, on the same cells: a selection has to cover all fifteen candidates before the
first specification exists, and two of the ones it drops -- `volatility_change` and
`inflation_change` -- are changes since origination, so they split a calendar key the selected
model leaves whole. 1.88x the key, for a reading that serves every candidate.

## Newton instead of the optimiser's long path

The cold fit's 142 SLSQP evaluations at 17 seconds each were 96% of its 43.79 minutes, while
the two Newton steps that finished it cost 39 seconds a Hessian. That ratio is the one the whole
engine used to be shaped around: a Hessian was 49 times a value, so a hundred of them was out
of the question and the optimiser's long path was the only option. At twice a
value-and-gradient the arithmetic changes sides. From lifelines' own seed, nothing else
changed:

| | time | evaluations | Newton steps | log-likelihood |
|---|---|---|---|---|
| SLSQP then the polish | 43.79 min | **142** | 2 | -10,691,177.6879 |
| **damped Newton** | **10.18 min** | **16** | 8 | -10,691,177.6879 |
| the nested candidate, from the seed | 10.91 min | 16 | 8 | -10,693,329.3806 |
| the same candidate, warm from its parent | 5.67 min | 6 | 5 | -10,693,329.3806 |

**4.3x on a cold fit**, to the same optimum, certified 3.31e-05 standard errors from it. The
path is worth reading: it begins **1,890** standard errors out and the first six evaluations are
refused as not a likelihood -- the damped step probing the region where lifelines' clipped
likelihood is unbounded below, which is exactly what the wall is for -- and then the damping
ladder walks it in: 1.26e3, 706, 423, 207, 58.4, 6.61, 0.107, 3.31e-05.

The last two rows say warm starts are still worth their keep: half the time, the same optimum
to four decimals of ten million.

The method chain stays behind it as the fallback, because this replaces the optimiser's path
and no fit that used to succeed may fail. `polish=False` keeps the optimiser untouched, because
that mode exists to reproduce lifelines exactly and is what the equivalence tests use.

## What a selection run costs now

Roughly:

    3 readings                   35 min   (0 where they are already on disk)
    the candidates' moments        0 min   (4.6 before they came off the keys)
    1 cold fit                    2 min   (was 44, then 10, then 4 on one thread)
    ~29 warm candidates          32 min   (~1.1 min each: 5 Hessians at 4.2 s, 6 values at 2.5)
                                ------
                                1.2 hours, or 0.6 on a table already read

against the **10.5 hours** the four recorded runs averaged, and 42.0 hours for all four. Rule 2
needs four runs and a reading does not depend on the family, so the two families share one and
the two causes do not -- and with the readings now written to disk, a whole campaign pays two of
them and nothing on a re-run.

The reading is no longer the problem and the optimiser's path is no longer the problem; what is
left is the **curvature**. A warm candidate's minutes are five Hessians and six evaluations, so
it is Hessian-bound, and a Hessian over the whole half was 29.69 s of which the jet was 67% --
spent on allocation and dispatch rather than on arithmetic, which is what a fused loop takes and
Python cannot. It is 9.74 s now.

## The last pass over the parquet, and what is left after it

**Steps 5 and 6 were the last thing in a selection that read the cell file for itself.** They
need a weighted covariance of the 19 continuous candidates, and `weighted_moments` took it off
the expanded episodes. It comes off the key frames instead, by the argument the whole encoding
rests on: every candidate is a function of the loan combination or of the calendar key, so a sum
over rows is a sum over combinations times the weight they carry. The weighted count per
combination is one scatter-add over the mapped rows; a pair on the same side is then a dot
product over 3,001 or 286,387 entries; and a pair across the sides is `A.T @ (W B)`, one gather
and one scatter-add per calendar candidate, which is the only part that touches 72.7 million
numbers. **Nothing new is stored** -- the rows are already on disk and the keys already carry the
values.

Measured on the production half, 19 candidates over 72,671,500 rows:

| | off the keys | off the cells |
|---|---|---|
| time | **19.68 s** | **273.39 s** |
| rows, loan-months | 72,671,500 and 2,112,532,468 | the same |
| covariance | max relative **1.06e-11** | |
| correlation | max relative **1.07e-11** | |
| the deviations the design is scaled by | max relative **8.89e-14** | |

**14x**, and the two agree to the digits a sum over 72.7 million terms taken in two orders can.
What the steps read off it are thresholds on a correlation and on a variance inflation factor, so
1e-11 cannot move a decision; what it can move is the last digits of two committed tables, and
that is the whole cost.

**And the figure that justified it was wrong, by a factor of four.** The pass was described in a
comment as "twenty minutes of parquet", which is where that number came from in this report and
in CLAUDE.md -- a stale figure carried forward, not a measurement anybody took. It is **4.6
minutes**. So the saving is 4.2 minutes a selection and about **17 minutes a campaign**, not the
80 published here this morning. It is kept because it is done, verified and simpler -- the
command loses a step and a pass -- but it is the smallest of the three options and not the
largest, and the ordering below is corrected with it.

### What is left, priced

Two, and the arithmetic is per *campaign* -- four selections, two causes by two families --
because that is the unit a decision is taken in. A reading is cached on disk and does not depend
on the family, so a campaign pays **six** readings once per table and nothing on a re-run; what
recurs is what nothing caches.

**1. The exact row merge, which needs the table sorted** -- 1.34x on every evaluation, so a warm
candidate from about 2.5 minutes to 1.9: **70 minutes a campaign, every campaign**, and the
largest of the two. Measured above: 1.11x unsorted, 1.34x sorted, and a per-quarter sort recovers
all of it. Cost: a re-aggregation of the whole book, the fit cache, the cached readings, and the
row count every report prints as its sample size.

**2. Step 9 reads the book twice more than it needs to** -- **46 minutes, once per table**, not
per campaign, because those two readings are cached like any other. The two origination-parity
halves are subsets of the first reading, but a parity is not recoverable from an encoded row: the
row carries `(i, j, age, event, weight)` and the origination month is in neither key. Carrying it
on the calendar key *frame*, or one byte of parity on the row, makes each half a boolean mask.
Cost: masking cuts the blocks differently, so step 9's two fits differ in their last digits from a
freshly-read pair and the published stability table moves with them.

What is *not* on the list, and why. Rule 7 was re-priced on 9 October and neither given-up
extension fits, so there is no rebuild of the cell table to carry the sort for free. Step 7's
one-step screen was abandoned with the number that says why: 75 minutes before the compiled
kernel, fifteen after, and it is the only item that would touch the declared procedure. And the
compiled kernel is done, at 2.80x and an exception declared beside rule 13.

## Threads, and the gate they were measured against

Rule 13 declared the gate before any of this was written: a value, a gradient and a Hessian over
the whole training half at least **twice** as fast as the single-threaded 9.74 s -- 4.87 s or
better -- on this machine's four physical cores, or abandoned with the number published. And the
summation stays deterministic: the rows go into a **fixed** number of contiguous parts, each part
sums its own in its own order, and the partials are added in the parts' own order rather than as
they finish.

| threads | value+gradient | with the Hessian | ns a row |
|---|---|---|---|
| 1 | 8.69 s | 11.34 s | 156 |
| 2 | 4.65 | 6.81 | 94 |
| **4** | **2.45** | **4.22** | **58** |
| 6 | 2.29 | 3.77 | 52 |
| 8 | 2.20 | 3.53 | 49 |

**It passes, and both readings of the gate agree this time**: 4.22 s is inside the declared 4.87,
and 11.34 / 4.22 = **2.69x within the run**, inside which the single-threaded figure was measured
alongside. Against the NumPy path in the same conditions -- 31.18 s -- four threads are **7.4x**.
The process holds 471 MB, unchanged: the threads share the mapped rows and each keeps 2.5 MB of
accumulators.

**The machine drifted 13 to 18% during the session, which is why nothing here is a ratio against
an older number.** The single-threaded kernel read 9.74 s this morning and 11.34 this evening on
the same source, and the restructuring looked like a 15% regression until the NumPy baseline was
re-measured and had drifted with it, 27.4 s to 31.2. Hours of sustained load on a laptop; the
only defensible comparison is one taken inside a run.

**The objective is identical across thread counts, not merely close**, and that is the
compensated sum: Neumaier recovers the same total whatever the grouping. The gradient and the
curvature accumulate into per-combination bins, which are grouped differently, and agree to
6.8e-14. Two runs at the same count agree **bit for bit** at every count tried, including counts
that do not divide the rows and one larger than the machine's cores.

And the fit is the same fit. The published model's cached optimum, re-fitted warm through the
**four-thread** kernel on the whole half: 0.35 minutes, one evaluation, certifying it 7.73e-05
standard errors out at a log-likelihood of -10,688,088.42593586 against the cached
-10,688,088.42593586 -- a difference of exactly **0.0**, the same numbers the single-threaded run
produced.

**Four, and declared rather than discovered.** `config.KERNEL_THREADS` is the count, and it is a
constant rather than `os.cpu_count()` because the count is part of what determines the answer: a
number read off the hardware would make a fit's last digits a property of the machine, which is
the mistake a pooled fit in this project already made once. Six and eight reach 3.77 and 3.53,
and half a second is not worth a count that has stopped meaning anything. Every fit records what
summed it.

## The gate for compiling anything

The compiled kernel is measured **against the NumPy above**, not against autograd. Measuring it
against autograd would credit a compiled language with removing a tape that NumPy already
removed.

And what is left is not arithmetic. Of a Hessian's 409 ns a row the jet is 67%, the
scatter-adds 18% and the gathers 4% -- and the jet's 178 array operations a chunk should cost
about 5 ms by the memory they move where they cost 19, the difference being a fresh array per
operation and numpy's per-call dispatch. That is also why removing a quarter of those operations
bought 12% and not 25%, and why the chunk sweep has a floor in the middle rather than at an end:
small chunks pay the dispatch, large ones pay the memory. The primitives themselves are 1.3 ns a
row for an exponential, 0.3 for a multiply, 1.7 for a gather and 3.1 for a scatter-add. A fused
loop holds the six jet components in registers, lets the compiler delete the structural zeros
outright, and traverses the fifteen bytes of a row once -- so what it removes is exactly the
part that is not arithmetic.

So the gate, declared before anything is written: **value+gradient+Hessian over the whole
training half -- 72,671,500 rows at 26 parameters -- in at most 9.90 s against the 29.69
measured here, with resident memory no higher than 1.31 GB.** Three times, on the same rows,
the same parameters and this machine. If it does not pass, it is abandoned and the number is
published here.

**It is third in the queue, not first, and the reason is still the arithmetic above.** A
selection is now three readings and thirty fits, and a fit is a handful of Hessians: compiling
the evaluation takes a warm candidate's 5.67 minutes to perhaps two and a selection's 3.5 hours
to about 1.6. Worth having, after the two changes that cost nothing and cannot fail.

### A whole selection, end to end, on a short window

Every piece above was measured on its own. This is the rehearsal that runs them together: a
full `creditsurv select` on the window up to **2002-12** -- 3,246,878 cells over 103,080,649
loan-months -- with the reports redirected so nothing published was touched.

| | |
|---|---|
| all ten steps | correlation, inflation, screening, backward elimination, stability, materiality |
| time | **18.1 minutes**, peak **1.28 GB** |
| fits | **33**: 24 fitted, **9 served from the cache** |
| readings written | **three** -- the window, and step 9's two origination-parity halves |
| outcome | 13 covariates eliminated, a formula selected, ten report tables written |

The window is a tenth of the production one, so the minutes do not translate; what the rehearsal
is for is the **three**. That is the number the whole design rests on -- a reading per *sample*
rather than per candidate -- and it is what a run actually produces, not what a docstring claims.
The nine cached fits are the other half of it: the cache is hit inside a run, not only between
runs.

It also found a gap in `prune-encodings`, which is what rehearsals are for. The sweep reasons by
the cell table's identity, so a reading of the table **on disk** at a reporting date nobody will
ask about again -- these three -- is current and useless at the same time, and no rule can tell
which. The audit now prints the reporting date and `--name` takes one by name.

## The compiled kernel, measured against that gate

`crates/creditsurv-kernel` is the same arithmetic as a fused loop: one `#[pyfunction]`, numpy
arrays of fixed dtype across the boundary, and the six jet components in registers instead of
six arrays over a chunk. It is built by `uv sync --extra kernel`, optional at import, and
`models/kernel/terms.py` stays normative -- the suite runs the equivalence tests against each
backend that is present and passes with neither.

Three readings on the whole training half, 72,671,500 rows at 26 parameters, one thread:

| | NumPy | compiled | ratio |
|---|---|---|---|
| value+gradient | 12.17, 12.18, 12.25 s | **7.79, 7.72, 7.72 s** | **1.57x** |
| with the Hessian | 26.86, 27.55, 27.44 s | **9.73, 9.76, 9.74 s** | **2.80x** |
| resident | 0.47 GB | **0.47 GB** | ceiling 1.31 |

The compiled figure is stable to **0.15%** across runs; the NumPy baseline is not, and that
matters below.

**The verdict is that it does not pass, and the reason the verdict needs a paragraph is a defect
in how the gate was written.** It was declared two ways in the same sentence -- "in at most
**9.90 s** against the **29.69** the NumPy kernel is measured at" and "**three times**, on the
same rows and this machine" -- and the two disagree: 9.74 s is inside 9.90, and 2.80x is not
three. What settles it is that the **absolute** number is not reproducible. The same NumPy code,
unchanged, measures 26.86 to 29.69 s across this session's runs, an 8% spread with machine state;
the ratio measured inside one run does not move. So the meaningful quantity is the ratio, the
ratio is 2.80x with a spread of 0.05, and three is outside it. Reading the 9.74 as a pass would
be choosing the thermometer that suits: the final run's baseline was the fastest of six.

**It is kept, and the exception is declared in rule 13.** The condition the owner attached was
that the logic be airtight rather than that the number be three, so what follows is what that
cost and what it bought.

**The two backends on the production table, element by element.** 72.7 million rows at the
published specification, both families, at the seed and at a point in the data:

| | agreement |
|---|---|
| the objective | **1.2e-15** relative, and 1.97e-16 on one point |
| the gradient, 26 entries | **1.1e-13** at worst |
| the curvature, 676 entries | **1.7e-13** of the largest entry |

The curvature's worst *elementwise* relative figure is 4.9e-10, and it is on an entry of 6.19
against a largest entry of 8.09e+07 -- seven orders below the matrix's scale, where a relative
measure means nothing. The absolute spread over the whole Hessian is 1.35e-05.

**And every fit on disk was unreadable until this was found.** The engine was one module before
this branch, and a pickle resolves a class by importing the module it was written from: all 176
cached fits name `creditsurv.models.blocks.BlockFit`, which the split into `models/engine/` had
removed. `load_fit` turns an unreadable pickle into a **miss** -- deliberately, because a pickle
is tied to the versions that wrote it -- so the cache went silently empty and a run would have
started cold. `creditsurv/models/blocks.py` keeps the old name importable, a test holds it to the
class so nobody tidies it away as the decoy it looks like, and `load_fit`'s warning now names the
exception, because a `ModuleNotFoundError` wants a shim where a version skew wants a refit. It
was found by trying to load a real fit, not by a test.

**Those figures are two orders better than the first ones, and the reason is a summation order.**
Three of the accumulators are scalars over every row, and 72.7 million sequential additions into
one `f64` is the worst order there is: the error grows with the count, where NumPy's `bincount`
and `dot` sum pairwise and it grows with its square root. The two backends read 4.0e-11 apart on
the objective, 7.9e-11 on the gradient -- against the **1.97e-16** that 443 blocks and 800
reproduce, which is this project's own standard for two orderings of one sum. Neumaier's
compensation on those three sums closes it for a handful of flops a row, inside the noise of the
evaluation, and it is deterministic: fixed order, no reassociation, the same bits every run. The
gradient and the curvature need none of it -- they accumulate into 3,001 and 286,387 bins, so
each sums about 24,000 terms rather than 72.7 million.

**And the proof that settles it is a fit.** The published model's optimum in the cache was found
by the NumPy kernel; re-fitted warm from its own coefficients through the **compiled** one, on
the whole training half:

| | |
|---|---|
| time | **0.30 minutes**, 1 evaluation, 0 Newton steps |
| where it began, and ended | **7.73e-05 standard errors** from the optimum |
| log-likelihood | **-10,688,088.42593586**, against the cached -10,688,088.42593586 |
| the difference | **0.0** |
| coefficients | moved at most **7.1e-15 standard errors** |
| standard errors | differ by at most **3.5e-13** relative |
| resident | 0.47 GB |

So the compiled kernel certifies the NumPy kernel's optimum as the optimum, on the production
table, with the polish's own measure. That is the equivalence that matters: not that two arrays
of numbers are close, but that a fit lands in the same place.

**One difference is declared rather than fixed.** Outside the data the log-logistic's Hessian
overflows and the two reach a different flavour of non-finite in the same entries: `-inf` from
the compiled loop where the NumPy reads `nan`. It is structural. The Python chain carries a
structural zero as the literal `0.0` and drops the term, so `0 * inf` never arises there, while
an array element that is merely numerically zero gets no such treatment and poisons the sum; the
compiled loop takes the shortcut on the value instead. What the engine reads is the objective,
which agrees there to the last bit, and a step to a non-finite curvature is refused by the polish
either way -- so the test holds the claim that decides a fit: identical wherever the curvature is
a number, and not a number wherever the other is not.

**Where the remaining 20% is, measured rather than assumed.** The obvious guess is libm -- a row
needs three cumulative hazards, two survivals and one logarithm, six transcendentals, and no loop
engineering removes them. It is wrong. Rebuilt with every `exp` and `ln` replaced by an affine
expression, which keeps the control flow and destroys the answer, the compiled Hessian reads
**92 ns a row against 137**: the transcendentals are **33%** and the jet chain is the other two
thirds. There is room, and it is not in libm.

Two things that were tried and are recorded because they look like they should work:

* **`-C target-cpu=native` is a regression**, 11.71 s against 10.18 on the same source. The row
  loop is scalar and dependent, so there is nothing to vectorise, and the instruction selection
  it chooses instead is worse. Removed.
* **Panics unwind rather than abort**, which was not the first setting. `panic = "abort"` is
  the faster one in principle and what it does here is kill the interpreter with no traceback in
  the middle of a fit that may be an hour old; unwinding lets pyo3 raise a Python exception the
  engine can see. Measured both ways in the same conditions: **10.19 s against 10.27**, 0.8%,
  inside the noise. An earlier reading of 6.5% was machine state and is retracted.
* **Unchecked indexing bought 2% and cost a segmentation fault.** Replacing the tables' bounds
  checks with `get_unchecked` took a Hessian from 10.18 s to 9.96, and the first thing it did was
  turn a wrong accumulator length into a crash **inside an ordinary fit**: a model with no
  calendar covariate hands over a table of *n* rows and **zero** columns, and the length had been
  derived by dividing the flat slice by the column count, which gives zero rows. The
  accumulators were allocated empty and the loop wrote past them. A bounds check would have named
  the array and the index; two per cent is not what this project pays for that. The checks are
  back, the length comes from the shape, and `tests/test_kernel.py` fits a loan-only model through
  both backends.
