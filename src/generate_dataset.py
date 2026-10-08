"""Generate the demo borrower dataset (data/borrowers.csv).

Synthetic, but shaped so the dashboard's relationships are visible:
default risk rises with debt burden and past defaults, and repayment
performance varies by channel.
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

REGIONS = ["Dar es Salaam", "Arusha", "Mwanza", "Dodoma", "Mbeya", "Tanga"]
EMPLOYMENT = ["Salaried", "Self-employed", "Informal trade", "Smallholder farmer"]
EDUCATION = ["Primary", "Secondary", "Vocational", "University"]
CHANNELS = ["M-Pesa", "Tigo Pesa", "Airtel Money", "Bank transfer", "Cash at branch"]
CHANNEL_QUALITY = {
    "M-Pesa": 1.04, "Tigo Pesa": 1.01, "Airtel Money": 0.99,
    "Bank transfer": 1.06, "Cash at branch": 0.88,
}


def generate(n=2400):
    employment = RNG.choice(EMPLOYMENT, n, p=[0.30, 0.34, 0.26, 0.10])
    income_base = {
        "Salaried": 1_250_000, "Self-employed": 980_000,
        "Informal trade": 620_000, "Smallholder farmer": 480_000,
    }
    monthly_income = np.array([
        RNG.lognormal(np.log(income_base[e]), 0.42) for e in employment
    ]).round(-3)

    expense_ratio = RNG.beta(5, 6, n) * 0.75 + 0.18
    monthly_expenses = (monthly_income * expense_ratio).round(-3)

    has_debt = RNG.random(n) < 0.45
    existing_debt_payment = np.where(
        has_debt, (monthly_income * RNG.beta(2, 9, n) * 0.9).round(-3), 0
    )

    term_months = RNG.choice([3, 6, 9, 12, 18, 24], n, p=[.14, .28, .18, .22, .11, .07])
    loan_multiple = RNG.gamma(3.2, 0.55, n)
    loan_amount = (monthly_income * loan_multiple).clip(150_000, 12_000_000).round(-3)
    interest_rate = (RNG.normal(18.5, 3.2, n)).clip(9, 32).round(1)

    previous_loans = RNG.poisson(1.6, n)
    previous_defaults = RNG.binomial(previous_loans, 0.11)

    r = interest_rate / 100 / 12
    factor = (1 + r) ** term_months
    monthly_payment = loan_amount * r * factor / (factor - 1)
    disposable = monthly_income - monthly_expenses - existing_debt_payment
    dti = (existing_debt_payment + monthly_payment) / monthly_income

    channel = RNG.choice(CHANNELS, n, p=[.34, .19, .17, .18, .12])
    quality = np.array([CHANNEL_QUALITY[c] for c in channel])

    # Repayment pressure: high DTI and prior defaults depress performance.
    strain = (
        1.9 * np.clip(dti - 0.35, 0, None)
        + 0.14 * previous_defaults
        + 0.9 * np.clip(-disposable / monthly_income, 0, None)
    )
    performance = np.clip(
        RNG.normal(1.0, 0.11, n) * quality - strain, 0.05, 1.22
    )

    application_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        RNG.integers(0, 640, n), unit="D"
    )
    amount_due = (monthly_payment * term_months).round(-3)
    amount_repaid = (amount_due * performance).round(-3)
    days_late = np.clip(
        ((1 - performance) * RNG.normal(130, 45, n)).round(), 0, 365
    ).astype(int)

    df = pd.DataFrame({
        "borrower_id": [f"B{i:05d}" for i in range(1, n + 1)],
        "age": RNG.integers(21, 62, n),
        "gender": RNG.choice(["Female", "Male"], n, p=[0.53, 0.47]),
        "region": RNG.choice(REGIONS, n),
        "employment_type": employment,
        "education": RNG.choice(EDUCATION, n, p=[0.21, 0.42, 0.24, 0.13]),
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "existing_debt_payment": existing_debt_payment,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "term_months": term_months,
        "application_date": application_date.strftime("%Y-%m-%d"),
        "payment_channel": channel,
        "amount_due": amount_due,
        "amount_repaid": amount_repaid,
        "days_late": days_late,
        "previous_loans": previous_loans,
        "previous_defaults": previous_defaults,
    })
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, "data", "borrowers.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    generate().to_csv(out, index=False)
    print(f"wrote {out}")
