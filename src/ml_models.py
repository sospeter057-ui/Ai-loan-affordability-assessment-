"""Default-risk and affordability models.

Two gradient-boosted classifiers over the same engineered feature block:
one predicts default, one predicts whether the borrower serviced the loan
comfortably. Models are persisted with joblib and trained on first use if
the models/ directory is empty.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "monthly_income", "monthly_expenses", "existing_debt_payment",
    "loan_amount", "interest_rate", "term_months", "monthly_payment",
    "disposable_income", "dti", "pti", "loan_to_income",
    "previous_loans", "previous_defaults", "age",
]

DEFAULT_MODEL = "default_risk.joblib"
AFFORDABILITY_MODEL = "affordability.joblib"


def _matrix(records):
    """Build the feature matrix, tolerating missing columns."""
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
    out = pd.DataFrame(index=df.index)
    for col in FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
    return out.fillna(0.0)


def _new_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )),
    ])


def train_models(df, model_dir=None):
    """Fit both classifiers. Persists to `model_dir` when given."""
    X = _matrix(df)

    y_default = df["defaulted"].astype(int)
    # "Affordable" = the instalment fitted inside disposable income and the
    # borrower actually kept up with it.
    y_afford = (
        (df["disposable_income"] >= df["monthly_payment"])
        & (df["dti"] <= 0.45)
        & (df["repayment_efficiency"] >= 85)
    ).astype(int)

    models = {}
    for name, y in [("default", y_default), ("affordability", y_afford)]:
        pipe = _new_pipeline()
        if y.nunique() < 2:
            # Degenerate target (e.g. a tiny uploaded CSV) - skip rather than crash.
            models[name] = None
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipe.fit(X_tr, y_tr)
        models[name] = pipe
        models[f"{name}_accuracy"] = float(pipe.score(X_te, y_te))

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        if models.get("default") is not None:
            joblib.dump(models["default"], os.path.join(model_dir, DEFAULT_MODEL))
        if models.get("affordability") is not None:
            joblib.dump(models["affordability"], os.path.join(model_dir, AFFORDABILITY_MODEL))

    return models


def load_models(model_dir):
    """Load persisted models, training them from the demo dataset if absent."""
    default_path = os.path.join(model_dir, DEFAULT_MODEL)
    afford_path = os.path.join(model_dir, AFFORDABILITY_MODEL)

    if os.path.exists(default_path) and os.path.exists(afford_path):
        return {
            "default": joblib.load(default_path),
            "affordability": joblib.load(afford_path),
        }

    base = os.path.dirname(os.path.abspath(model_dir.rstrip(os.sep)))
    csv = os.path.join(base, "data", "borrowers.csv")
    if not os.path.exists(csv):
        return {"default": None, "affordability": None}

    from src.data_processing import prepare_data
    return train_models(prepare_data(pd.read_csv(csv)), model_dir)


def predict_borrower(row, models):
    """Score a single borrower record (a dict from `df.to_dict()`).

    Returns an empty dict when no model is available; the dashboard then
    falls back to the rule-based affordability score.
    """
    if not models:
        return {}
    X = _matrix(pd.DataFrame([row]))
    result = {}

    default_model = models.get("default")
    if default_model is not None:
        result["default_probability"] = float(default_model.predict_proba(X)[0][1])

    afford_model = models.get("affordability")
    if afford_model is not None:
        result["affordability_probability"] = float(afford_model.predict_proba(X)[0][1])

    return result
