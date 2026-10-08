"""Repayment forecasting.

Deliberately a transparent trend baseline rather than a black box: the
dashboard presents it as a planning aid, and a linear fit on monthly totals
is something a credit officer can sanity-check by eye.
"""
import numpy as np
import pandas as pd


def monthly_totals(df):
    d = df.copy()
    d["application_date"] = pd.to_datetime(d["application_date"], errors="coerce")
    d = d.dropna(subset=["application_date"])
    if d.empty:
        return pd.DataFrame(columns=["application_date", "amount_repaid"])
    return (
        d.set_index("application_date")
        .resample("MS")["amount_repaid"]
        .sum()
        .reset_index()
    )


def monthly_payment_forecast(df, periods=6):
    """Project `periods` months ahead with an OLS trend on monthly repayments.

    Returns an empty frame when there is too little history to fit a line,
    which the caller treats as "no forecast trace".
    """
    monthly = monthly_totals(df)
    if len(monthly) < 3 or periods < 1:
        return pd.DataFrame(columns=["application_date", "forecast_amount"])

    y = monthly["amount_repaid"].to_numpy(dtype="float64")
    x = np.arange(len(y), dtype="float64")
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(y), len(y) + periods, dtype="float64")
    forecast = np.clip(slope * future_x + intercept, 0, None)

    last = monthly["application_date"].iloc[-1]
    future_dates = pd.date_range(last, periods=periods + 1, freq="MS")[1:]

    return pd.DataFrame(
        {"application_date": future_dates, "forecast_amount": forecast}
    )
