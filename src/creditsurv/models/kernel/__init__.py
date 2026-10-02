"""The interval-censored likelihood and its derivatives, written out rather than traced.

This module is the second implementation of lifelines' likelihood that
:mod:`creditsurv.models.engine` was built to avoid, and it is here because the measurement
said so. At the 26 parameters rule 12 produced, a value-and-gradient on a 250,000-cell block
of the production table is 166.8 ms and a Hessian 1,373 -- so a cold fit is an hour and a
selection run is a day, and the finer bands the calibration needs cost a full re-selection for
two families. ``docs/reports/engine.md`` carries the attribution.

**What makes writing it out cheap is the shape of the data, not cleverness.** Measured on the
training window:

* every column of the design is a function of the **loan** combination (3,001 of them) or of
  the **calendar** key (153,309), and never of both, so the scale's linear predictor is
  ``eta = A[i] + B[j]`` -- two small tables rebuilt per evaluation for about a million flops,
  against expanding a 26-column design over 53 million rows;
* the interval is always ``[a, a+1]`` for an exit and ``[a+1, INFINITY_STAND_IN)`` for a
  survivor, the entry is always the age, and exact observations never occur, so every time a
  row needs is a lookup into a table of at most 361 ages;
* the shape is a single number -- no production fit gives it covariates -- so **a row depends
  on exactly two scalars**: its own ``eta`` and the shape's coefficient ``r``.

That last point is what this module is built around. The derivatives of a function of two
scalars are six numbers, whatever the number of parameters, and they are carried here through a
second-order forward chain (:class:`_Jet`) rather than derived by hand: the formulas below read
like the likelihood, and the chain makes their first and second derivatives exact by
construction. ``tests/test_kernel.py`` holds every one of them to autograd's own answer,
including inside the clipped regions, because a wall that moves by an epsilon moves
``blocks._outside_the_domain`` with it and changes what the guards do.

**Matching lifelines exactly means matching its clips, including where they are wrong.**
``safe_exp`` caps its argument and then reports the derivative of the *uncapped* exponential,
which is what autograd's custom VJP does; the Weibull's survival function is unclipped while
the log-logistic's is clipped to ``[1e-12, 1-1e-12]`` with a derivative of zero outside,
because the Weibull fitter overrides the method and the log-logistic does not; the interval
probability is clipped to ``[1e-25, 1-1e-25]`` in the likelihood itself, and the
left-truncation term is added to it **unclipped**, which is the whole reason the objective is
unbounded below and the reason this engine has a floor and a wall.

----

Three files, because a compiled implementation of the same arithmetic replaces exactly one of
them:

* :mod:`.likelihood` -- one row's log-likelihood and its derivatives, and lifelines' clips
  reproduced exactly;
* :mod:`.factorisation` -- the design as two small tables and the rows as two indices;
* :mod:`.terms` -- **the port**: rows and tables in, a scalar, a gradient and a curvature out.
"""

from creditsurv.models.kernel.factorisation import Expanded, Factorisation, Rows
from creditsurv.models.kernel.likelihood import (
    FAMILIES,
    INFINITY_STAND_IN,
    INTERVAL_CEILING,
    INTERVAL_FLOOR,
    MAX_EXPONENT,
    SURVIVAL_CEILING,
    SURVIVAL_FLOOR,
    TIME_FLOOR,
    log_times,
    row_likelihood,
)
from creditsurv.models.kernel.terms import Kernel

__all__ = [
    "FAMILIES",
    "INFINITY_STAND_IN",
    "INTERVAL_CEILING",
    "INTERVAL_FLOOR",
    "MAX_EXPONENT",
    "SURVIVAL_CEILING",
    "SURVIVAL_FLOOR",
    "TIME_FLOOR",
    "Expanded",
    "Factorisation",
    "Kernel",
    "Rows",
    "log_times",
    "row_likelihood",
]
