# Variables

Every variable the data, the model and the reports use, with what it is, its unit, its source
and the values it takes. The pages of this site show the **label**; the **name** is what the
code, the tables in `docs/tables` and the commands use. Where a variable was renamed in
September 2026, its **former** name is listed too, because the history on this site -- the
validation response, the decision log, the first selection run -- quotes the names as they
were at the time.

The table is generated from `creditsurv.names`, the registry the code reads its names from,
so it cannot describe a variable the model does not have.

## The loan, at origination

Measured once, when the loan was written. The continuous ones enter the cells at the midpoint
of their band.

<!-- table: variables_loan -->

## Macro covariates

Built after the collapse from the loan's origination month, its age and a FRED series, each
series read three months back. A **change since origination** is zero when the loan is written.

<!-- table: variables_macro -->

## The macro series

The monthly FRED series the covariates are built from.

<!-- table: variables_series -->

## The structure of the panel

<!-- table: variables_structure -->
