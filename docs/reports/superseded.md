# What was superseded, and when

This project keeps retracted reasoning on the record rather than deleting it -- the same reason
[the register](../rules_register.md) exists and the same reason the withdrawn arguments stay in
[variable selection](../variable_selection.md) under a notice. Deleting them would make the
repository look as though it had always been right. What this page is for is to say which
generated report was of which model, so that nobody reads one by accident.

## The three September reports: regenerated, and back in their own sections

[The methodology report](methodology.md), [the calibration report](calibration.md) and
[the backtest report](backtesting.md) were generated on **17 September 2026** and described a
model with the old covariate names (`fico_s`, `cltv_drift`, `nfci_lagged`, `has_mi`), no HARP
level, estimation to 2024-12, one backtest cut, the banded covariates read as a line through
their midpoints, and prepayment as independent censoring rather than a competing risk. The
second independent validation found them in the navigation while the main pages linked to them.

**They were regenerated on 19 October 2026** and now describe the configured model. They have
been returned to the sections whose current pages they sit beside, and this entry stays because
the gap was a month long and a reader who saw it is owed the explanation.

What closed it for good is not the regeneration. It is that a generated report now states its
own formula, its own shape and the date it was written, and that `tests/test_site.py` holds
every report a reporting module writes to the configuration -- with an exemption keyed to the
generation date, so it retires itself the moment a report is regenerated rather than needing
somebody to remember.

## Windows, anchoring and grades: still superseded

[That report](windows.md) carries its own notice. Every fit behind it has a **scalar shape**,
because the campaign never passed the ancillary formula to them, so its three regime cuts, its
anchoring multiplier and its master scale are of a model the configuration does not declare.
The third validation found it as N1 on 19 October 2026. It is regenerated once the corrected
campaign has run.

## The distribution family: still superseded

[That report](family.md) carries its own notice. It was run on **1 October 2026**, before
rule 12's bands and before rule 14's shape, so both families are compared on the linear form
with a scalar shape. Rule 2's comparison under the bands is owed and is blocked on the
prepayment model.

## The pattern, stated once

Every entry above is the same failure: **a generated artefact that is not tied by construction
to the model it describes.** The second validation named it, the third validation named it again
in a place the first fix had not reached, and what the fixes have in common is that they make
the tie a test rather than a habit.
