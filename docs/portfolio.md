# The portfolio

What the book looks like before any model is fitted to it: what was lent, when, to whom, and
how it performed. None of it is modelling. It is what makes the modelling legible, and it is
where several problems were found that no coefficient would have shown.

## In short

| | |
|---|---|
| Loans originated | **<!-- value: book.loans -->** |
| Amount originated | **<!-- value: book.amount -->** |
| Peak contracts outstanding | <!-- value: book.peak_contracts --> |
| Peak balance outstanding | <!-- value: book.peak_balance --> |
| Loan-months outstanding | <!-- value: book.loan_months --> |

The figures on this page read **the whole book**, including the loans the model leaves out
-- the HARP refinances and the other incomplete cases (see [Data](data.md#what-is-out-of-scope)).
The default is the model's own event: a moratorium is not one.

## New lending

<!-- figure: lending_volume -->

Two panels rather than two axes: a count and an amount share no scale. The series is the US
mortgage market's own history, and any model fitted across it is estimating across four
lending regimes:

- **2003**, the refinancing boom: September 2003 alone writes 576,869 loans, as the 30-year
  rate reaches its then-lowest point.
- **2004 to 2008**: volumes fall by two thirds as rates rise, then the crisis.
- **2009 to 2012**: a second refinancing wave, on much tighter underwriting.
- **2020 to 2021**: the largest by amount, at the lowest rates in the series; then **2022 to
  2024**, the sharpest contraction in the dataset.

That is the argument for time-varying covariates rather than a vintage dummy.

## What was written

The share of each year's loans, by segment. Pick one from the menu.

<!-- figure: lending_mix -->

??? example "A finding this view produced: `channel` cannot be used at four levels"
    Until 2008 roughly half of originations are coded `T`, third party not specified; from
    2009 the code vanishes and broker and correspondent absorb it entirely.

    | Share of new lending | 1999 | 2003 | 2008 | 2009 | 2021 |
    |---|---|---|---|---|---|
    | Retail | 53.8% | 58.1% | 50.4% | 58.2% | 57.9% |
    | Third party, not specified (`T`) | **46.1%** | **41.7%** | 31.2% | **0.0%** | **0.0%** |
    | Broker (`B`) | 0.0% | 0.1% | 7.2% | 16.6% | 13.4% |
    | Correspondent (`C`) | 0.1% | 0.1% | 11.2% | 25.2% | 28.6% |

    That is a change in how the field was coded, not in how loans were sold, and a model given
    four levels would read it as a risk effect. Retail's share is stable across the whole
    history, so the field means the same thing in every vintage only as **retail against
    third party**. The pooled frequencies look unremarkable: only the time series shows it.

## Underwriting, and how it drifted

<!-- figure: underwriting -->

The band is who was being lent to, the line the middle of the book. Credit scores move up and
their spread narrows after 2008: underwriting tightened, and the book being written changed
with it. That population shift is what makes an out-of-time backtest meaningful and an
in-sample fit flattering.

## The book outstanding

Loans outstanding each month, stacked by segment.

<!-- figure: outstanding -->

## How the book performed

**Default rate.** Defaults over loan-months, by calendar month, in basis points a month.

<!-- figure: default_rate -->

**Prepayment rate.** The conditional prepayment rate, `1 - (1 - SMM)^12`, where the single
monthly mortality is the share of the loans outstanding that prepay in the month -- counted in
loans, not in balance. Prepayment is the other way a loan leaves the book, and the model
treats it as censoring.

<!-- figure: prepayment_rate -->

??? note "Two spikes the moratorium fields do not remove"
    November 2005 and December 2017 stand out, at 11.1 and 6.8 basis points in the whole book,
    and they are in the modelled cells too. Each falls 90 days after a hurricane season --
    Katrina, then Harvey, Irma and Maria -- which is when disaster forbearance would first
    show as a 90-day delinquency. The disaster flag that marks such months is not set on
    those rows, so they count as defaults. That reading rests on the timing alone; it has not
    been measured loan by loan.

!!! note "2020 looks quiet, and that is the event definition speaking"
    Forbearance postponed the defaults it did not prevent, and the refinancing wave filled the
    book with new, low-hazard loans. With moratoria counted as defaults the May 2020 rate was
    90.4 basis points; see [Data](data.md#what-counts-as-a-default).

## Vintage curves

Cumulative default by loan age, one line per vintage year, estimated by Kaplan-Meier on each
vintage's own risk sets. A curve ends where fewer than <!-- value: figures.floor --> loan-months remain at risk.

<!-- figure: vintage_curves -->

!!! warning "Not the share of the vintage that defaulted"
    Kaplan-Meier treats a prepaid loan as censored, so one minus it is the probability of
    default for a loan that never prepays. Most loans do prepay, and those still in the book
    after ten years are the ones that could not refinance, so the curve runs well above the
    share of a vintage's loans that actually defaulted. Treating prepayment as a competing
    risk would give that share; the model does not yet.

The picture every mortgage report opens with, and the plainest statement of what the macro
covariates have to explain: vintages written into 2006 against those written into 2012.

## The economy the loans lived through

The monthly FRED series every macro covariate is built from, before the three-month lag.

<!-- figure: macro_series -->

See the [data dictionary](data_dictionary.md) for what each series is and how the covariates
are built from it.

??? info "Reproducing"
    ```bash
    uv run creditsurv fetch-macro
    uv run creditsurv ingest
    uv run creditsurv aggregate
    uv run creditsurv views --no-model   # the tables behind this page
    uv run creditsurv portfolio          # the static figures and portfolio_summary.json
    ```
