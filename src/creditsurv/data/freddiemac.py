"""The Freddie Mac Single-Family dataset's published record layout.

The project's data source. It sits behind a free but manual registration on Clarity, so it
cannot be fetched as part of a reproducible pipeline; what is public is the **layout**, and
that is what this module is. `data.ingest` reads these two field lists to parse the
pipe-delimited files into parquet, and nothing here touches the network or a file.

The lists were extracted from ``file_layout_july_2026.xlsx`` rather than transcribed by hand:
66 fields across two files is too many to copy reliably, and a single misplaced name shifts
every column after it while still parsing cleanly.

Each vintage comes as two pipe-delimited files with no header row:

``sample_orig_YYYY.txt``
    One row per loan, as underwritten.
``sample_svcg_YYYY.txt``
    One row per loan-month.

**The pandas loader that used to live here is now `tests/freddiemac_sample.py`.** It builds
the fixtures' canonical panel and nothing in the package calls it, and while it was here it
closed the one import cycle this codebase had: it needed the event definition from
`data.book`, which needs the layout from `data.ingest`, which needs this module.
"""

from __future__ import annotations

from typing import Final

#: Download page for the dataset. Registration is free but manual.
CLARITY_URL: Final = "https://claritydownload.fmapps.freddiemac.com/CRT/"

#: Origination file fields, in positional order, from the published layout.
ORIGINATION_COLUMNS: Final[tuple[str, ...]] = (
    "classic_fico",
    "first_payment_date",
    "first_time_homebuyer_indicator",
    "maturity_date",
    "msa",
    "mortgage_insurance_percentage",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_indicator",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_identifier",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantagescore_4",
)

#: Monthly performance file fields, in positional order.
PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "loan_identifier",
    "period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_of_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_loan_to_value",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "mortgage_insurance_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
)
