"""Backtest metrics: discrimination, calibration, accuracy and stability.

Discrimination and calibration answer different questions, and a model can be
excellent at one while useless at the other. A model that ranks every loan
correctly but predicts three times the observed default rate discriminates
perfectly and would still misprice the book. Both are reported, always.

Every metric here is computed **per loan**, never per loan-month. The episode panel
weights a loan by how long it survived, so a five-year loan would count sixty times
and a loan that defaulted in month three would count three times -- which is
precisely backwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

if TYPE_CHECKING:
    from collections.abc import Sequence


def discrimination(
    durations: pd.Series, events: pd.Series, predicted_pd: pd.Series
) -> dict[str, float]:
    """Harrell's concordance and the Gini coefficient derived from it.

    Concordance is computed against *negated* PD, because the index expects a
    predicted survival time: a higher probability of default must mean a shorter
    expected life. Getting that sign wrong yields a mirror-image result -- 0.3
    instead of 0.7 -- which reads as a broken model rather than a flipped sign.
    """
    index = float(
        concordance_index(durations, -predicted_pd.to_numpy(dtype=float), events.astype(bool))
    )
    return {"concordance": index, "gini": 2.0 * index - 1.0}


def brier_score(observed: pd.Series, predicted_pd: pd.Series) -> float:
    """Mean squared error of the predicted default probability."""
    outcome = observed.astype(float).to_numpy()
    prediction = predicted_pd.to_numpy(dtype=float)
    return float(np.mean((prediction - outcome) ** 2))


def calibration_table(
    observed: pd.Series,
    predicted_pd: pd.Series,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Predicted against realised default rate, by bucket of predicted PD.

    The actual-versus-expected view a credit committee reads. Buckets are quantiles
    of the prediction, so each holds a similar number of loans and the tail buckets
    -- where the model earns or loses its money -- are not one loan wide.
    """
    frame = pd.DataFrame(
        {
            "predicted": predicted_pd.to_numpy(dtype=float),
            "observed": observed.astype(float).to_numpy(),
        }
    )
    ranks = frame["predicted"].rank(method="first")
    frame["bucket"] = pd.qcut(ranks, q=min(n_buckets, len(frame)), labels=False, duplicates="drop")

    grouped = frame.groupby("bucket", observed=True).agg(
        loans=("observed", "size"),
        expected=("predicted", "mean"),
        actual=("observed", "mean"),
    )
    grouped["difference"] = grouped["actual"] - grouped["expected"]
    grouped["ratio"] = np.where(
        grouped["expected"] > 0, grouped["actual"] / grouped["expected"], np.nan
    )
    return grouped.reset_index()


def calibration_slope_intercept(observed: pd.Series, predicted_pd: pd.Series) -> dict[str, float]:
    """Regress realised outcomes on predicted log-odds.

    A perfectly calibrated model gives slope 1 and intercept 0. Slope below 1 means
    the predictions are spread too widely -- the model is more confident than the
    data supports; intercept away from 0 means a level bias, the whole book
    mispriced in one direction.
    """
    prediction = np.clip(predicted_pd.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(prediction / (1.0 - prediction))
    outcome = observed.astype(float).to_numpy()

    design = np.column_stack([np.ones_like(logit), logit])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)

    # Convert the linear fit back to a calibration statement on the logit scale.
    mean_prediction = float(prediction.mean())
    return {
        "slope": float(coefficients[1] / max(mean_prediction * (1 - mean_prediction), 1e-9)),
        "intercept": float(coefficients[0] - outcome.mean() + mean_prediction),
        "expected": mean_prediction,
        "actual": float(outcome.mean()),
        "actual_over_expected": float(outcome.mean() / mean_prediction)
        if mean_prediction > 0
        else float("nan"),
    }


def population_stability_index(
    reference: pd.Series, comparison: pd.Series, *, n_bins: int = 10
) -> float:
    """How far a covariate's distribution has moved between two samples.

    Standard credit monitoring, and the piece that explains *why* performance
    degrades rather than only reporting that it did. Convention: below 0.1 is
    stable, 0.1 to 0.25 warrants attention, above 0.25 is a material shift.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference.to_numpy(dtype=float), bins=edges)[0] / len(reference)
    comparison_share = np.histogram(comparison.to_numpy(dtype=float), bins=edges)[0] / len(
        comparison
    )

    # A zero share makes the log infinite, so empty bins are floored.
    floor = 1e-6
    reference_share = np.clip(reference_share, floor, None)
    comparison_share = np.clip(comparison_share, floor, None)

    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def stability_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    covariates: Sequence[str],
    *,
    time_varying: Sequence[str] = (),
) -> pd.DataFrame:
    """Population stability index per covariate, worst first.

    The usual reading -- below 0.1 stable, 0.1 to 0.25 monitor, above 0.25 shifted
    -- was devised for **application characteristics**: credit score, loan size,
    the mix of business being written. Those should look much the same this quarter
    as last, and a jump means something has changed about who is applying.

    It does not transfer to macro-driven covariates. ``cltv_drift`` and
    ``unemp_gap`` are *designed* to move with the economy, and train and test come
    from different calendar periods by construction, so their index is large
    whenever anything happened. This backtest returns 9.23 for ``cltv_drift``
    against 0.03 for ``fico_s``: the first number says the economy changed, which
    is not news and is not a defect, while the second says the book being written
    did not, which is the thing PSI was built to detect.

    Those covariates are therefore labelled rather than dropped, so the alarming
    figure is visible and correctly discounted instead of quietly triggering a
    model review.
    """
    varying = set(time_varying)
    rows = [
        {
            "covariate": name,
            "psi": population_stability_index(train[name], test[name]),
            "kind": "time-varying" if name in varying else "static",
        }
        for name in covariates
        if name in train.columns and name in test.columns
    ]
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    table["interpretation"] = np.where(
        table["kind"] == "time-varying",
        "expected to move",
        pd.cut(
            table["psi"],
            bins=[-np.inf, 0.1, 0.25, np.inf],
            labels=["stable", "monitor", "shifted"],
        ).astype(str),
    )
    return table
