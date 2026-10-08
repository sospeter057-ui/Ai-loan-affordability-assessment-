"""Loading, cleaning and feature derivation for borrower records.

`prepare_data` is the single entry point used by the dashboard. It is
idempotent: the dashboard round-trips prepared frames through a dcc.Store
and feeds them back in, so re-deriving columns must be safe.
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "borrower_id", "age", "gender", "region", "employment_type", "education",
    "monthly_income", "monthly_expenses", "existing_debt_payment",
    "loan_amount", "interest_rate", "term_months", "application_date",
    "payment_channel", "amount_due", "amount_repaid", "days_late",
    "previous_loans", "previous_defaults",
]

NUMERIC_COLUMNS = [
    "age", "monthly_income", "monthly_expenses", "existing_debt_payment",
    "loan_amount", "interest_rate", "term_months", "amount_due",
    "amount_repaid", "days_late", "previous_loans", "previous_defaults",
]


def amortized_payment(principal, annual_rate_pct, term_months):
    """Standard equal-instalment payment. Falls back to straight-line at 0%."""
    principal = np.asarray(principal, dtype="float64")
    term = np.asarray(term_months, dtype="float64")
    r = np.asarray(annual_rate_pct, dtype="float64") / 100.0 / 12.0
    term = np.where(term <= 0, 1.0, term)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (1 + r) ** term
        payment = np.where(r > 0, principal * r * factor / (factor - 1), principal / term)
    return np.nan_to_num(payment, nan=0.0, posinf=0.0, neginf=0.0)


def affordability_score(dti, loan_to_income, repayment_efficiency):
    """0-100 blend of debt burden, loan size and past repayment behaviour.

    Mirrors the heuristic the Loan Simulator applies to hypothetical loans so
    that a simulated borrower and a stored one are scored on the same scale.
    """
    score = 100 - 55 * dti - 18 * loan_to_income + 0.70 * (repayment_efficiency - 100)
    return np.clip(score, 0, 100)


def prepare_data(df):
    df = pd.DataFrame(df).copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "borrower_id" not in df.columns:
        df["borrower_id"] = [f"B{i:05d}" for i in range(1, len(df) + 1)]
    df["borrower_id"] = df["borrower_id"].astype(str)

    for col, default in [
        ("monthly_income", 0.0), ("monthly_expenses", 0.0),
        ("existing_debt_payment", 0.0), ("loan_amount", 0.0),
        ("interest_rate", 0.0), ("term_months", 1.0),
        ("amount_due", 0.0), ("amount_repaid", 0.0), ("days_late", 0.0),
        ("previous_loans", 0.0), ("previous_defaults", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["term_months"] = df["term_months"].clip(lower=1)
    df["monthly_income"] = df["monthly_income"].clip(lower=1)

    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    else:
        df["application_date"] = pd.NaT
    df["application_date"] = df["application_date"].fillna(pd.Timestamp("2026-01-01"))

    if "payment_channel" not in df.columns:
        df["payment_channel"] = "Unknown"
    df["payment_channel"] = df["payment_channel"].fillna("Unknown").astype(str)

    # --- derived affordability features -------------------------------------
    df["monthly_payment"] = amortized_payment(
        df["loan_amount"], df["interest_rate"], df["term_months"]
    )
    df["disposable_income"] = (
        df["monthly_income"] - df["monthly_expenses"] - df["existing_debt_payment"]
    )
    df["dti"] = (df["existing_debt_payment"] + df["monthly_payment"]) / df["monthly_income"]
    df["pti"] = df["monthly_payment"] / df["monthly_income"]
    df["loan_to_income"] = df["loan_amount"] / df["monthly_income"]

    due = df["amount_due"].replace(0, np.nan)
    # Sized as a bubble radius in the Overview scatter, so keep it non-negative.
    df["repayment_efficiency"] = (
        (df["amount_repaid"] / due * 100).fillna(0).clip(0, 150)
    )

    df["affordability_score"] = affordability_score(
        df["dti"], df["loan_to_income"], df["repayment_efficiency"]
    )

    if "defaulted" not in df.columns:
        df["defaulted"] = (
            (df["days_late"] > 90) | (df["repayment_efficiency"] < 60)
        ).astype(int)
    df["defaulted"] = pd.to_numeric(df["defaulted"], errors="coerce").fillna(0).astype(int)

    if "loan_status" not in df.columns:
        df["loan_status"] = np.select(
            [df["defaulted"] == 1, df["repayment_efficiency"] >= 99, df["days_late"] > 30],
            ["Defaulted", "Fully Repaid", "Late"],
            default="Current",
        )
    df["loan_status"] = df["loan_status"].astype(str)

    return df.reset_index(drop=True)
