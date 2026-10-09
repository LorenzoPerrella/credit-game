"""Fit a lifelines AFT model block by block, so memory stops growing with the population.

lifelines evaluates the likelihood of every row at once and differentiates it with
autograd, which keeps every intermediate array for the backward pass. On this project's
panel a Weibull fit was measured at about **680 bytes per training row** above the data,
in the optimiser and again in the Hessian: three filtered copies of the design matrix
live on the tape, on top of the two full copies ``_fit`` makes before optimising. At
9.5 million rows the process reached a 13.45 GB footprint and swapped. The exact
calendar key puts the training half at 62 million rows, which is 45-50 GB.

The log-likelihood is a sum over rows, and so are its gradient and its Hessian. Added up
block by block they are the same numbers -- up to the order in which floating-point
additions happen -- so the optimiser takes the same steps to the same optimum while only
one block's tape exists at a time. Nothing is approximated and nothing is sampled.

What is mirrored, from lifelines 0.30:

* ``ParametericAFTRegressionFitter.fit_interval_censoring`` -- the regressors, the checks
  on the bounds, the stand-in for an infinite upper bound;
* ``ParametricRegressionFitter._fit`` -- the design, the column scaling, and the
  attributes ``summary``, ``AIC_`` and ``predict_cumulative_hazard`` read;
* ``ParametricRegressionFitter._fit_model`` -- the objective, the optimiser and its
  options, the Hessian;
* ``ParametericAFTRegressionFitter._create_initial_point`` -- the univariate fit that
  seeds the intercept. It reads the rows only through their bounds and weights, so it
  runs on the distinct bounds with summed weights: a few hundred rows in place of
  millions, and the same likelihood.

All four are private, which is why ``tests/test_blocks.py`` fits the same rows both ways
and compares coefficients, standard errors, log-likelihood and predictions. Re-run it
before trusting this module with a different lifelines.

A model fitted here does not carry the full-length copies of its training data lifelines
attaches -- ``lower_bound``, ``upper_bound``, ``event_observed``, ``entry``, ``weights``
-- nor the predicted medians behind ``concordance_index_`` or the central values behind
the partial-effects plots. Nothing in the project reads them, the cached model is smaller
for it, and anything that does try fails with an ``AttributeError`` rather than a wrong
number.

----

The module this package came from was 2,325 lines and seven concerns. They are now nine files,
and the dependency graph between them is a line:

* :mod:`.contract` -- what an evaluation *means*, and what ends a fit that cannot finish. The
  port: the bounds, the wall, the floor, the guards, and the protocol everything above obeys.
* :mod:`.storage` -- the rows, held at 30 bytes each and expanded only while evaluated.
* :mod:`.lifelines_glue` -- the four private lifelines calls this engine stands in for.
* :mod:`.scan` -- one pass over the rows, kept as blocks or encoded as keys.
* :mod:`.objective` -- the objective over this process's rows, by either arithmetic.
* :mod:`.workers` -- the same, split across processes, added in the parts' own order.
* :mod:`.polish` -- damped Newton, the certificate, and the method chain behind it.
* :mod:`.fit` -- the two entry points.
* :mod:`.cache` -- a reading written once and mapped back, so the eleven minutes are paid once.

What a caller needs is re-exported here, so nothing outside has to know which file a name is
in. What a *test* needs it should import from the file, because which concern a test names is
part of what the test says.
"""

from creditsurv.models.engine.cache import (
    encoding_fingerprint,
    load_encoding,
    save_encoding,
)
from creditsurv.models.engine.contract import (
    POLISH_TOLERANCE_SE,
    Pinned,
)
from creditsurv.models.engine.fit import (
    BlockFit,
    fit_encoded,
    fit_interval_censoring_in_blocks,
)
from creditsurv.models.engine.scan import Encoding, encode_blocks
from creditsurv.models.engine.storage import (
    DEFAULT_BLOCK_ROWS,
    INFINITY_STAND_IN,
    StoredColumn,
)

__all__ = [
    "DEFAULT_BLOCK_ROWS",
    "INFINITY_STAND_IN",
    "POLISH_TOLERANCE_SE",
    "BlockFit",
    "Encoding",
    "Pinned",
    "StoredColumn",
    "encode_blocks",
    "encoding_fingerprint",
    "fit_encoded",
    "fit_interval_censoring_in_blocks",
    "load_encoding",
    "save_encoding",
]
