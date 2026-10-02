# How the code is arranged

One page, because the arrangement is simple and the useful part is the rule rather than the
diagram. `tests/test_architecture.py` enforces everything on this page; if the two ever
disagree, the test is right.

## The layers, and the one rule

The order is the one the data moves in. Raw archives become a book, the book becomes cells,
cells become fits, fits become views, views become a site, and the command line sits above all
of it.

| | layer | what lives there |
|---|---|---|
| 0 | `config`, `names`, `features` | declared constants, the variable registry, the covariate definitions |
| 1 | `data/` | the archives, the book's SQL, the cell table, the artefact store |
| 2 | `explore` | descriptive statistics over frames the data layer produces |
| 3 | `models/` | the likelihood, the fit engine, the selection procedure |
| 4 | `backtest/` | the windows, the metrics, the acceptance criteria |
| 5 | `portfolio`, `profiling` | descriptions of the book and prices for the cell key |
| 6 | `views/`, `reporting/` | the aggregates the site reads and the documents it shows |
| 7 | `site/` | the build-time hook that places figures and numbers on pages |
| 8 | `cli` | the commands |

**A module may import from its own layer or any deeper one, and never from a shallower one.**
That is the whole rule. It is not a style preference: it is what makes the pieces replaceable.
A `models/` that could read `reporting/` would be a model whose answer depends on how it is
written up.

The rule is read at **both scopes**, module level and inside functions. This package uses
function-level imports heavily and deliberately — importing `data.panel` at module scope put
**0.85 s** on every `creditsurv --help` — so a graph built from the top of each file would miss
most of `cli.py`.

## The one cycle, and why it is allowed

`data.ingest` reads the record layout from `data.freddiemac`, which needs the event definition
from `data.book`, which needs the manifest from `data.ingest`. The last edge is taken inside the
function that needs it, with the reason written where it is taken. Every other cycle is a
mistake, and the test says so by name.

## The two ports

Two places are deliberately narrow, because something is expected to be swapped behind them.

**The objective, as the optimiser sees it.** `models/blocks.py` defines what an evaluation means —
a value and a gradient, a curvature on request, the penalty, the wall below which the objective
is not a likelihood, the floor a nested model may not cross, and the guards that end a hopeless
fit. Three implementations satisfy it: autograd over a stored design, the written-out
likelihood, and the pool that adds up what several processes computed. Everything above it is
algebra on a handful of numbers and does not know which one it has.

**The arithmetic, as the objective sees it.** `models/kernel` takes the encoded rows and the two
design tables and returns a scalar, a gradient and a curvature. It imports **nothing** from this
package — the test checks that — because that is what makes a compiled implementation a drop-in
rather than a fork. The moment it reads a declared constant it stops being a port and becomes
part of the model.

## The patterns, named where they are

Named rather than introduced: most of these were here before anyone wrote them down.

- **The ubiquitous language is a registry, not a convention.** `names.py` holds every variable
  the code can name, with its label, its former names and its levels, and four tests in
  `tests/test_names.py` refuse a covariate the registry does not know, a level the aggregation
  does not produce, and a name that is also another variable's former name. A shared vocabulary
  that nothing enforces is a glossary; this one fails the build.
- **Value objects everywhere.** `Specification`, `Encoding`, `Rows`, `Expanded`, `BlockFit`,
  `StoredColumn`, `Moments`, `Split` — frozen, compared by value, derived rather than mutated.
  There are no entities: nothing in this codebase has an identity that outlives a computation.
- **Strategy** at the two ports above.
- **Repository** for the artefacts a long run must not lose. `data/store.py` gives a fit a
  fingerprint of everything that determines it, an atomic write, a readable description beside
  the bytes, and a search over the descriptions. A fit on this book is minutes to hours; losing
  one to a crash while writing a report is how the cache came to exist.
- **Declared rules, held to a record by a test.** The strongest discipline here and the least
  like a pattern. `docs/rules.md` states what will be decided before the fit that decides it;
  `tests/` holds the configuration to the record the selection wrote, the master scale to the
  rule that declared it, and the give-up ladder to the order `docs/rules.md` gives. A rule that
  can be changed after seeing the result is not a rule.

## What is deliberately not here

No `domain/`, `application/`, `infrastructure/` directories: the packages above already group by
concern, and a reader looking for the cell table finds `data/`. No entities, aggregates, domain
events or buses: there is no mutable state with identity to put in them. No dependency-injection
container and no abstract base class where a `Protocol` does the job. And no repository, factory
or adapter introduced for its own sake — only where a second implementation exists or is known to
be coming.
